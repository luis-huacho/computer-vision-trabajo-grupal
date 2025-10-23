import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
import cv2
import os
from PIL import Image
import io
import warnings

warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="Segmentación de Personas - AISegment Optimizado",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# ARQUITECTURA DEL MODELO
# ============================================================================

class AttentionBlock(nn.Module):
    """Attention Gate para U-Net."""

    def __init__(self, gate_channels, in_channels, inter_channels):
        super(AttentionBlock, self).__init__()

        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.input_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_conv = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gate):
        gate_conv = self.gate_conv(gate)
        input_conv = self.input_conv(x)

        if gate_conv.shape[2:] != input_conv.shape[2:]:
            gate_conv = F.interpolate(gate_conv, size=input_conv.shape[2:], mode='bilinear', align_corners=False)

        combined = self.relu(gate_conv + input_conv)
        attention = self.sigmoid(self.bn(self.output_conv(combined)))

        return x * attention


class DoubleConv(nn.Module):
    """Bloque de doble convolución usado en U-Net."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    """Encoder path del U-Net con skip connections usando ResNet-50."""

    def __init__(self, pretrained=True):
        super(UNetEncoder, self).__init__()

        resnet = resnet50(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        self.bottleneck = DoubleConv(2048, 4096, dropout_rate=0.2)

    def forward(self, x):
        skip_connections = []

        x1 = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x1)

        x2 = self.maxpool(x1)

        x3 = self.layer1(x2)
        skip_connections.append(x3)

        x4 = self.layer2(x3)
        skip_connections.append(x4)

        x5 = self.layer3(x4)
        skip_connections.append(x5)

        x6 = self.layer4(x5)
        skip_connections.append(x6)

        x7 = self.bottleneck(x6)

        return x7, skip_connections


class UNetDecoder(nn.Module):
    """Decoder path del U-Net con Attention Gates para ResNet-50."""

    def __init__(self, use_attention=True):
        super(UNetDecoder, self).__init__()
        self.use_attention = use_attention

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)

        if self.use_attention:
            self.att1 = AttentionBlock(2048, 2048, 1024)
            self.att2 = AttentionBlock(1024, 1024, 512)
            self.att3 = AttentionBlock(512, 512, 256)
            self.att4 = AttentionBlock(256, 256, 128)
            self.att5 = AttentionBlock(64, 64, 32)

        # Decoder convolutions
        self.conv1 = DoubleConv(4096, 2048)
        self.conv2 = DoubleConv(2048, 1024)
        self.conv3 = DoubleConv(1024, 512)
        self.conv4 = DoubleConv(512, 256)
        self.conv5 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)

    def _match_tensor_size(self, x, target_tensor):
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, skip_connections):
        skips = skip_connections[::-1]

        x = self.up1(x)
        skip = skips[0]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att1(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        skip = skips[1]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att2(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        skip = skips[2]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att3(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        skip = skips[3]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att4(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        skip = skips[4]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att5(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv5(x)

        x = self.final_conv(x)

        rgb = torch.sigmoid(x[:, :3])
        alpha = torch.sigmoid(x[:, 3:4])

        return torch.cat([rgb, alpha], dim=1)


class UNetAutoencoder(nn.Module):
    """U-Net Autoencoder completo para remoción de fondo."""

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder, self).__init__()
        self.encoder = UNetEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded


# ============================================================================
# CLASE DE INFERENCIA
# ============================================================================

class SegmentationInference:
    """Clase para realizar inferencia con el modelo de segmentación."""

    def __init__(self, model_path, device='cpu', image_size=384):
        self.device = device
        self.image_size = image_size
        self.model = UNetAutoencoder(pretrained=False, use_attention=True)

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            self.model_loaded = False

    def process_image(self, image):
        """
        Procesa una imagen y retorna el resultado de segmentación.

        Args:
            image: PIL Image o numpy array

        Returns:
            dict con la imagen segmentada (RGBA) y la máscara
        """
        if not self.model_loaded:
            return None

        try:
            # Convertir PIL a numpy array si es necesario
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Asegurar formato RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = image[:, :, :3]
            else:
                st.error("Formato de imagen no soportado")
                return None

            original_size = image_rgb.shape[:2]

            # Redimensionar con padding para mantener proporciones
            h, w = image_rgb.shape[:2]
            scale = min(self.image_size / h, self.image_size / w)
            new_h, new_w = int(h * scale), int(w * scale)

            image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Calcular padding
            pad_h = (self.image_size - new_h) // 2
            pad_w = (self.image_size - new_w) // 2
            pad_h_bottom = self.image_size - new_h - pad_h
            pad_w_right = self.image_size - new_w - pad_w

            # Aplicar padding
            image_padded = cv2.copyMakeBorder(
                image_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
                cv2.BORDER_CONSTANT, value=0
            )

            # Normalizar
            image_normalized = image_padded.astype(np.float32) / 255.0

            # Convertir a tensor
            input_tensor = torch.FloatTensor(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            # Inferencia
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output.squeeze(0).cpu().numpy()

            # Extraer RGB y Alpha
            rgb_channels = output[:3].transpose(1, 2, 0)
            alpha_channel = output[3]

            # Remover padding
            rgb_unpadded = rgb_channels[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :]
            alpha_unpadded = alpha_channel[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

            # Restaurar al tamaño original
            rgb_restored = cv2.resize(rgb_unpadded, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha_restored = cv2.resize(alpha_unpadded, (w, h), interpolation=cv2.INTER_NEAREST)

            # Usar colores originales con máscara suavizada
            threshold = 0.4
            person_mask = alpha_restored > threshold

            original_normalized = image_rgb.astype(np.float32) / 255.0
            rgb_enhanced = np.where(
                person_mask[..., np.newaxis],
                original_normalized,
                rgb_restored
            )

            # Suavizar máscara
            alpha_smooth = cv2.bilateralFilter(
                (alpha_restored * 255).astype(np.uint8),
                5, 50, 50
            ).astype(np.float32) / 255.0

            # Crear resultado RGBA
            result_rgba = np.zeros((h, w, 4), dtype=np.float32)
            result_rgba[:, :, :3] = rgb_enhanced
            result_rgba[:, :, 3] = alpha_smooth
            result_rgba = (result_rgba * 255).astype(np.uint8)

            # Máscara en formato uint8
            mask = (alpha_smooth * 255).astype(np.uint8)

            return {
                'original_image': image_rgb,
                'segmented_rgba': result_rgba,
                'mask': mask,
                'original_size': original_size
            }

        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")
            return None


# ============================================================================
# CARGA DEL MODELO (CON CACHE)
# ============================================================================

@st.cache_resource
def load_model():
    """Cargar el modelo de segmentación (con cache para evitar recargas)."""
    model_path = 'checkpoints/aisegment_20pct_optimized/best_segmentation.pth'

    if not os.path.exists(model_path):
        st.error(f"❌ Modelo no encontrado en: {model_path}")
        st.info("Asegúrate de que el modelo esté entrenado y guardado en la ruta correcta.")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SegmentationInference(model_path, device, image_size=384)


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación Streamlit."""

    # Header
    st.title("✂️ Segmentación de Personas con IA")
    st.markdown("**Modelo:** AISegment 20% Optimizado - U-Net ResNet-50 + Attention Gates")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Modelo")
        st.markdown("""
        ### Modelo AISegment Optimizado

        **Arquitectura:**
        - U-Net con ResNet-50 como encoder
        - Attention Gates para mayor precisión
        - Salida RGBA (RGB + canal Alpha)

        **Entrenamiento:**
        - Dataset: AISegment Matting Human (20%)
        - ~6,885 imágenes de alta calidad
        - Optimizado para retratos de personas

        **Características:**
        - Segmentación precisa de personas
        - Máscara alpha suave y natural
        - Preservación de colores originales
        - Procesamiento optimizado CPU/GPU

        ### Tecnología:
        - PyTorch
        - OpenCV
        - Streamlit
        """)

        # Configuración
        st.header("⚙️ Configuración")
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Dispositivo: {device}")

        show_mask = st.checkbox("Mostrar máscara", value=True)
        show_original = st.checkbox("Mostrar imagen original", value=True)

    # Cargar modelo
    with st.spinner("Cargando modelo..."):
        segmenter = load_model()

    if segmenter is None:
        st.stop()

    st.success("✅ Modelo de segmentación cargado correctamente")

    # Opciones de entrada de imagen
    st.header("📸 Selecciona la fuente de imagen")

    input_method = st.radio(
        "¿Cómo quieres proporcionar la imagen?",
        ["📤 Subir archivo", "📷 Capturar con cámara"],
        horizontal=True
    )

    original_image = None
    image_source = None

    if input_method == "📤 Subir archivo":
        st.subheader("📤 Subir Imagen")
        uploaded_file = st.file_uploader(
            "Sube una imagen con persona(s)...",
            type=['png', 'jpg', 'jpeg'],
            help="Sube una imagen que contenga personas para segmentación"
        )

        if uploaded_file is not None:
            image_source = "archivo"
            original_image = Image.open(uploaded_file)

            # Mostrar información del archivo
            with st.expander("📋 Información del archivo"):
                file_info = {
                    "Nombre": uploaded_file.name,
                    "Tamaño": f"{uploaded_file.size / 1024:.2f} KB",
                    "Tipo": uploaded_file.type
                }
                for key, value in file_info.items():
                    st.write(f"- **{key}:** {value}")

    else:  # Capturar con cámara
        st.subheader("📷 Capturar con Cámara")
        st.info("📸 Permite el acceso a la cámara cuando el navegador lo solicite")

        camera_photo = st.camera_input("Toma una foto")

        if camera_photo is not None:
            image_source = "cámara"
            original_image = Image.open(camera_photo)

            # Mostrar información de la captura
            with st.expander("📋 Información de la captura"):
                capture_info = {
                    "Tamaño": f"{camera_photo.size / 1024:.2f} KB",
                    "Tipo": camera_photo.type,
                    "Origen": "Cámara web"
                }
                for key, value in capture_info.items():
                    st.write(f"- **{key}:** {value}")

    if original_image is not None:
        # Botón de procesamiento
        if st.button("🚀 Procesar Imagen", type="primary", use_container_width=True):

            # Crear barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()

            import time
            start_time = time.time()

            # Procesamiento
            status_text.text("🔄 Preparando imagen...")
            progress_bar.progress(20)

            status_text.text("🧠 Procesando con IA...")
            progress_bar.progress(40)

            # Procesar con el modelo
            results = segmenter.process_image(original_image)

            if results is not None:
                progress_bar.progress(100)
                status_text.text("✅ ¡Procesamiento completado!")

                elapsed_time = time.time() - start_time
                st.info(f"⏱️ Tiempo de procesamiento: {elapsed_time:.2f}s")

                # Mostrar resultados
                st.header("📊 Resultados")

                # Crear columnas para mostrar imágenes
                num_cols = sum([show_original, True, show_mask])  # True para imagen cortada siempre
                cols = st.columns(num_cols)
                col_idx = 0

                # Imagen original
                if show_original:
                    with cols[col_idx]:
                        st.subheader("📷 Original")
                        st.image(results['original_image'], use_container_width=True)
                        st.caption(f"Tamaño: {results['original_size']}")
                    col_idx += 1

                # Imagen segmentada (RGBA) - siempre se muestra
                with cols[col_idx]:
                    st.subheader("✂️ Imagen Cortada")
                    st.image(results['segmented_rgba'], use_container_width=True)
                    st.caption("Formato: RGBA con transparencia")
                col_idx += 1

                # Máscara
                if show_mask:
                    with cols[col_idx]:
                        st.subheader("🎭 Máscara")
                        st.image(results['mask'], use_container_width=True, clamp=True)
                        st.caption("Canal Alpha de segmentación")

                # Estadísticas
                st.header("📈 Estadísticas")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    mask = results['mask']
                    total_pixels = mask.size
                    person_pixels = np.sum(mask > 127)
                    coverage = (person_pixels / total_pixels) * 100
                    st.metric("Cobertura", f"{coverage:.1f}%")

                with col2:
                    st.metric("Píxeles de Persona", f"{person_pixels:,}")

                with col3:
                    mask_mean = np.mean(mask)
                    st.metric("Brillo Promedio", f"{mask_mean:.1f}")

                with col4:
                    mask_std = np.std(mask.astype(np.float32))
                    st.metric("Contraste", f"{mask_std:.1f}")

                # Análisis de calidad
                st.header("🔬 Análisis de Calidad")

                # Análisis de bordes
                sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
                edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                edge_strength = np.mean(edge_magnitude)

                quality_col1, quality_col2 = st.columns(2)

                with quality_col1:
                    st.metric("Definición de Bordes", f"{edge_strength:.1f}")

                    # Puntuación de calidad
                    quality_score = min(100, (coverage * 0.3 + mask_std * 0.4 + edge_strength * 0.3))
                    st.metric("Puntuación de Calidad", f"{quality_score:.0f}/100")

                with quality_col2:
                    # Recomendaciones
                    st.write("**💡 Evaluación:**")
                    if quality_score >= 85:
                        st.success("✅ Excelente calidad de segmentación")
                    elif quality_score >= 70:
                        st.info("✔️ Buena calidad de segmentación")
                    elif quality_score >= 50:
                        st.warning("⚠️ Calidad moderada")
                    else:
                        st.error("❌ Baja calidad, considera otra imagen")

                # Opciones de descarga
                st.header("📥 Descargar Resultados")

                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    # Descargar imagen cortada (RGBA)
                    result_pil = Image.fromarray(results['segmented_rgba'], 'RGBA')
                    buf_rgba = io.BytesIO()
                    result_pil.save(buf_rgba, format='PNG')

                    st.download_button(
                        label="📥 Descargar Imagen Cortada (PNG)",
                        data=buf_rgba.getvalue(),
                        file_name="persona_segmentada.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_col2:
                    # Descargar máscara
                    mask_pil = Image.fromarray(results['mask'], 'L')
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')

                    st.download_button(
                        label="📥 Descargar Máscara (PNG)",
                        data=buf_mask.getvalue(),
                        file_name="mascara.png",
                        mime="image/png",
                        use_container_width=True
                    )

                # Análisis avanzado (opcional)
                with st.expander("🔬 Análisis Avanzado"):
                    st.subheader("📊 Histograma de la Máscara")

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.hist(mask.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_xlabel('Valor de Píxel')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Distribución de Valores en la Máscara')
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close()

            else:
                progress_bar.progress(0)
                status_text.text("❌ Error en el procesamiento")
                st.error("❌ Error al procesar la imagen")

    # Footer
    st.markdown("---")
    with st.expander("ℹ️ Información Técnica del Modelo"):
        st.markdown("""
        ### 🏗️ Arquitectura del Modelo

        **Encoder:**
        - ResNet-50 pre-entrenado (ImageNet)
        - Capas: conv1 (64) → layer1 (256) → layer2 (512) → layer3 (1024) → layer4 (2048)
        - Bottleneck: 4096 canales

        **Decoder:**
        - 5 bloques de upsampling con ConvTranspose2d
        - Attention Gates en cada nivel
        - Skip connections desde el encoder

        **Salida:**
        - 4 canales: RGB + Alpha
        - Activación: Sigmoid
        - Tamaño de procesamiento: 384x384

        ### 📚 Dataset de Entrenamiento

        **AISegment Matting Human (20%):**
        - ~6,885 imágenes de retratos
        - Máscaras de matting profesional de alta calidad
        - Enfoque específico en personas de medio cuerpo

        ### 🚀 Optimizaciones

        - Padding para mantener proporciones
        - Preservación de colores originales
        - Suavizado bilateral de máscara
        - Threshold adaptativo (0.4)
        - Restauración exacta al tamaño original
        """)

    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            ✂️ Segmentación de Personas con IA<br>
            Modelo: AISegment 20% Optimizado - U-Net ResNet-50<br>
            Desarrollado con PyTorch, OpenCV y Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()