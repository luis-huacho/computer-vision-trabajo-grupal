import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import numpy as np
import cv2
import os
from PIL import Image
import io
import warnings

warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="Removedor de Fondo con IA - Debug Mode",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ImageProcessor:
    """
    Clase para manejar redimensionamiento consistente de imágenes.
    Mantiene proporciones y permite restaurar tamaño original.
    """

    @staticmethod
    def resize_with_padding(image, mask, target_size):
        """
        Redimensiona imagen y máscara manteniendo proporciones y agregando padding.

        Args:
            image: Imagen de entrada (H, W, C)
            mask: Máscara de entrada (H, W)
            target_size: Tamaño objetivo (int)

        Returns:
            tuple: (imagen_redimensionada, máscara_redimensionada, metadatos_para_restaurar)
        """
        h, w = image.shape[:2]

        # Calcular escala para mantener proporciones
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Redimensionar manteniendo proporciones
        if len(image.shape) == 3:
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if len(mask.shape) == 2:
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Calcular padding para centrar
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        pad_h_bottom = target_size - new_h - pad_h
        pad_w_right = target_size - new_w - pad_w

        # Aplicar padding
        if len(image.shape) == 3:
            image_padded = cv2.copyMakeBorder(
                image_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            image_padded = cv2.copyMakeBorder(
                image_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
                cv2.BORDER_CONSTANT, value=0
            )

        mask_padded = cv2.copyMakeBorder(
            mask_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
            cv2.BORDER_CONSTANT, value=0
        )

        # Metadatos para restaurar
        restore_metadata = {
            'original_size': (h, w),
            'scale': scale,
            'new_size': (new_h, new_w),
            'padding': (pad_h, pad_w, pad_h_bottom, pad_w_right)
        }

        return image_padded, mask_padded, restore_metadata

    @staticmethod
    def restore_original_size(image, mask, restore_metadata):
        """
        Restaura imagen y máscara a su tamaño original.

        Args:
            image: Imagen procesada (target_size, target_size, C)
            mask: Máscara procesada (target_size, target_size)
            restore_metadata: Metadatos del redimensionamiento

        Returns:
            tuple: (imagen_original, máscara_original)
        """
        pad_h, pad_w, pad_h_bottom, pad_w_right = restore_metadata['padding']
        new_h, new_w = restore_metadata['new_size']
        original_h, original_w = restore_metadata['original_size']

        # Remover padding
        if len(image.shape) == 3:
            image_unpadded = image[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :]
        else:
            image_unpadded = image[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        if len(mask.shape) == 2:
            mask_unpadded = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        else:
            mask_unpadded = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        # Redimensionar al tamaño original
        if len(image.shape) == 3:
            image_restored = cv2.resize(image_unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_restored = cv2.resize(image_unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        mask_restored = cv2.resize(mask_unpadded, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        return image_restored, mask_restored


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
    """Encoder path del U-Net con skip connections."""

    def __init__(self, pretrained=True):
        super(UNetEncoder, self).__init__()

        resnet = resnet34(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024, dropout_rate=0.2)

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
    """Decoder path del U-Net con Attention Gates."""

    def __init__(self, use_attention=True):
        super(UNetDecoder, self).__init__()
        self.use_attention = use_attention

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        if self.use_attention:
            self.att1 = AttentionBlock(512, 512, 256)
            self.att2 = AttentionBlock(256, 256, 128)
            self.att3 = AttentionBlock(128, 128, 64)
            self.att4 = AttentionBlock(64, 64, 32)
            self.att5 = AttentionBlock(64, 64, 32)

        self.conv1 = DoubleConv(1024, 512)
        self.conv2 = DoubleConv(512, 256)
        self.conv3 = DoubleConv(256, 128)
        self.conv4 = DoubleConv(128, 64)
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
# CLASE DE INFERENCIA CON DEBUG
# ============================================================================

class BackgroundRemoverDebug:
    """
    Clase para realizar inferencia con el modelo entrenado.
    VERSIÓN DEBUG: Retorna todas las etapas intermedias del procesamiento.
    """

    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = UNetAutoencoder(pretrained=False, use_attention=True)
        self.processor = ImageProcessor()

        try:
            # Cargar modelo entrenado
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            self.model_loaded = False

    def remove_background_debug(self, image, image_size=256):
        """
        Remueve el fondo de una imagen y retorna todas las etapas intermedias.

        Returns:
            dict con todas las etapas del procesamiento
        """
        if not self.model_loaded:
            return None

        try:
            # Convertir PIL a numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Asegurar que la imagen esté en formato RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = image[:, :, :3]  # Quitar canal alpha si existe
            else:
                st.error("Formato de imagen no soportado")
                return None

            original_size = image_rgb.shape[:2]

            # ETAPA 1: Redimensionamiento con padding
            dummy_mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
            image_processed, _, restore_metadata = self.processor.resize_with_padding(
                image_rgb, dummy_mask, image_size
            )

            # Normalizar
            image_normalized = image_processed.astype(np.float32) / 255.0

            # Convertir a tensor
            input_tensor = torch.FloatTensor(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            # Optimización para CPU
            torch.set_num_threads(2)

            # ETAPA 2: Inferencia del modelo
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output.squeeze(0).cpu().numpy()

            # ETAPA 3: Post-procesamiento
            rgb_channels = output[:3].transpose(1, 2, 0)
            alpha_channel = output[3]

            # Crear imagen RGBA procesada (antes de restaurar)
            result_processed = np.zeros((image_size, image_size, 4), dtype=np.float32)
            result_processed[:, :, :3] = rgb_channels
            result_processed[:, :, 3] = alpha_channel
            result_processed = (result_processed * 255).astype(np.uint8)

            # ETAPA 4: Restaurar al tamaño original
            rgb_restored, alpha_restored = self.processor.restore_original_size(
                rgb_channels, alpha_channel, restore_metadata
            )

            # Crear imagen RGBA final
            result_final = np.zeros((original_size[0], original_size[1], 4), dtype=np.float32)
            result_final[:, :, :3] = rgb_restored
            result_final[:, :, 3] = alpha_restored
            result_final = (result_final * 255).astype(np.uint8)

            # Retornar todas las etapas
            return {
                'original_image': image_rgb,
                'resized_image': (image_processed * 255).astype(np.uint8),
                'generated_mask': (alpha_channel * 255).astype(np.uint8),
                'result_processed': result_processed,
                'result_final': result_final,
                'metadata': restore_metadata,
                'original_size': original_size,
                'processed_size': (image_size, image_size)
            }

        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")
            return None


@st.cache_resource
def load_model():
    """Cargar el modelo (con cache para evitar recargas)."""
    model_path = 'checkpoints/best_model.pth'

    if not os.path.exists(model_path):
        st.error("❌ Modelo no encontrado en 'checkpoints/best_model.pth'")
        st.info("Asegúrate de que el modelo esté entrenado y guardado en la ruta correcta.")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return BackgroundRemoverDebug(model_path, device)


def display_image_comparison(images_dict, stage_names, cols_per_row=3):
    """
    Muestra múltiples imágenes en una grilla organizada.

    Args:
        images_dict: Diccionario con las imágenes
        stage_names: Lista de nombres de las etapas a mostrar
        cols_per_row: Número de columnas por fila
    """
    num_images = len(stage_names)
    num_rows = (num_images + cols_per_row - 1) // cols_per_row

    for row in range(num_rows):
        cols = st.columns(cols_per_row)

        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx

            if img_idx < num_images:
                stage_name = stage_names[img_idx]

                with cols[col_idx]:
                    if stage_name in images_dict and images_dict[stage_name] is not None:
                        image_data = images_dict[stage_name]

                        # Convertir a PIL Image si es numpy array
                        if isinstance(image_data, np.ndarray):
                            if len(image_data.shape) == 3:
                                if image_data.shape[2] == 4:
                                    pil_image = Image.fromarray(image_data, 'RGBA')
                                else:
                                    pil_image = Image.fromarray(image_data, 'RGB')
                            else:
                                # Imagen en escala de grises
                                pil_image = Image.fromarray(image_data, 'L')
                        else:
                            pil_image = image_data

                        st.image(pil_image, caption=stage_name, use_column_width=True)

                        # Mostrar información adicional
                        if isinstance(image_data, np.ndarray):
                            st.caption(f"Dimensiones: {image_data.shape}")
                    else:
                        st.write(f"❌ {stage_name} no disponible")


def main():
    """Función principal de la aplicación Streamlit."""

    # Header
    st.title("🎭 Removedor de Fondo con IA - Modo Debug")
    st.markdown("**Visualiza todas las etapas del procesamiento: desde la imagen original hasta el resultado final**")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Debug")
        st.markdown("""
        ### Etapas visualizadas:
        1. **Imagen Original** - Como se subió
        2. **Imagen Redimensionada** - Con padding para el modelo
        3. **Máscara Generada** - Alpha channel del modelo
        4. **Resultado Procesado** - Antes de restaurar dimensiones
        5. **Resultado Final** - Restaurado al tamaño original

        ### Características técnicas:
        - Modelo U-Net con ResNet-34 backbone
        - Attention Gates para mejor precisión
        - Preservación de proporciones con padding
        - Restauración exacta de dimensiones
        """)

        # Configuración
        st.header("⚙️ Configuración")
        processing_size = st.slider("Tamaño de procesamiento", 128, 512, 256, 32)
        show_technical_info = st.checkbox("Mostrar información técnica", value=True)

        # Información del sistema
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Usando: {device}")

    # Cargar modelo
    with st.spinner("Cargando modelo..."):
        bg_remover = load_model()

    if bg_remover is None:
        st.stop()

    st.success("✅ Modelo cargado correctamente")

    # Subir imagen
    st.header("📤 Subir Imagen")
    uploaded_file = st.file_uploader(
        "Elige una imagen...",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen que contenga personas para visualizar todo el proceso"
    )

    if uploaded_file is not None:
        # Mostrar información del archivo
        file_info = {
            "Nombre": uploaded_file.name,
            "Tamaño": f"{uploaded_file.size / 1024:.2f} KB",
            "Tipo": uploaded_file.type
        }

        with st.expander("📋 Información del archivo"):
            for key, value in file_info.items():
                st.write(f"**{key}:** {value}")

        # Cargar imagen original
        original_image = Image.open(uploaded_file)
        original_array = np.array(original_image)

        st.header("🔄 Procesamiento Paso a Paso")

        # Botón de procesamiento
        if st.button("🚀 Procesar y Analizar Imagen", type="primary", use_container_width=True):

            # Crear columnas para mostrar progreso
            progress_col, info_col = st.columns([3, 1])

            with progress_col:
                progress_bar = st.progress(0)
                status_text = st.empty()

            with info_col:
                timer_text = st.empty()

            import time
            start_time = time.time()

            # Etapa 1: Preparación
            status_text.text("🔄 Preparando imagen...")
            progress_bar.progress(20)
            timer_text.text(f"⏱️ {time.time() - start_time:.1f}s")

            # Etapa 2: Procesamiento
            status_text.text("🧠 Procesando con IA...")
            progress_bar.progress(60)

            # Procesar imagen
            results = bg_remover.remove_background_debug(original_image, image_size=processing_size)

            progress_bar.progress(80)
            status_text.text("📊 Generando visualizaciones...")

            if results is not None:
                progress_bar.progress(100)
                status_text.text("✅ ¡Procesamiento completado!")
                timer_text.text(f"⏱️ {time.time() - start_time:.1f}s")

                # Preparar imágenes para visualización
                images_for_display = {
                    "1. Imagen Original": results['original_image'],
                    "2. Imagen Redimensionada": results['resized_image'],
                    "3. Máscara Generada": results['generated_mask'],
                    "4. Resultado Procesado": results['result_processed'],
                    "5. Resultado Final": results['result_final']
                }

                # Mostrar todas las etapas
                st.header("📊 Resultados del Procesamiento")

                # Visualización en grilla
                stage_names = list(images_for_display.keys())
                display_image_comparison(images_for_display, stage_names, cols_per_row=3)

                # Información técnica detallada
                if show_technical_info:
                    st.header("🔬 Análisis Técnico Detallado")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("📏 Dimensiones")
                        st.write(f"**Original:** {results['original_size']}")
                        st.write(f"**Procesado:** {results['processed_size']}")
                        st.write(f"**Escala aplicada:** {results['metadata']['scale']:.3f}")

                        padding = results['metadata']['padding']
                        st.write(f"**Padding aplicado:**")
                        st.write(f"- Superior: {padding[0]}px")
                        st.write(f"- Izquierdo: {padding[1]}px")
                        st.write(f"- Inferior: {padding[2]}px")
                        st.write(f"- Derecho: {padding[3]}px")

                    with col2:
                        st.subheader("📊 Estadísticas de la Máscara")
                        mask = results['generated_mask']
                        total_pixels = mask.size
                        person_pixels = np.sum(mask > 127)
                        background_pixels = total_pixels - person_pixels
                        coverage = (person_pixels / total_pixels) * 100

                        st.metric("Cobertura de Personas", f"{coverage:.1f}%")
                        st.metric("Píxeles de Persona", f"{person_pixels:,}")
                        st.metric("Píxeles de Fondo", f"{background_pixels:,}")
                        st.metric("Píxeles Totales", f"{total_pixels:,}")

                    with col3:
                        st.subheader("🎯 Calidad del Resultado")

                        # Análisis de la máscara
                        mask_std = np.std(mask.astype(np.float32))
                        mask_mean = np.mean(mask.astype(np.float32))

                        # Análisis de bordes (gradiente)
                        sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
                        sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
                        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                        edge_strength = np.mean(edge_magnitude)

                        st.metric("Contraste de Máscara", f"{mask_std:.1f}")
                        st.metric("Brillo Promedio", f"{mask_mean:.1f}")
                        st.metric("Definición de Bordes", f"{edge_strength:.1f}")

                        # Indicador de calidad general
                        quality_score = min(100, (coverage * 0.3 + mask_std * 0.4 + edge_strength * 0.3))
                        st.metric("Puntuación de Calidad", f"{quality_score:.0f}/100")

                # Comparación lado a lado
                st.header("🔍 Comparación Original vs Resultado")
                comp_col1, comp_col2 = st.columns(2)

                with comp_col1:
                    st.subheader("📷 Imagen Original")
                    st.image(results['original_image'], use_column_width=True)

                with comp_col2:
                    st.subheader("✨ Resultado Final")
                    st.image(results['result_final'], use_column_width=True)

                # Opciones de descarga
                st.header("📥 Descargar Resultados")

                download_col1, download_col2, download_col3 = st.columns(3)

                with download_col1:
                    # Descargar resultado final
                    result_pil = Image.fromarray(results['result_final'], 'RGBA')
                    buf_final = io.BytesIO()
                    result_pil.save(buf_final, format='PNG')

                    st.download_button(
                        label="📥 Resultado Final (PNG)",
                        data=buf_final.getvalue(),
                        file_name="imagen_sin_fondo_final.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_col2:
                    # Descargar máscara generada
                    mask_pil = Image.fromarray(results['generated_mask'], 'L')
                    buf_mask = io.BytesIO()
                    mask_pil.save(buf_mask, format='PNG')

                    st.download_button(
                        label="📥 Máscara (PNG)",
                        data=buf_mask.getvalue(),
                        file_name="mascara_generada.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_col3:
                    # Descargar resultado procesado (antes de restaurar)
                    processed_pil = Image.fromarray(results['result_processed'], 'RGBA')
                    buf_processed = io.BytesIO()
                    processed_pil.save(buf_processed, format='PNG')

                    st.download_button(
                        label="📥 Resultado Procesado",
                        data=buf_processed.getvalue(),
                        file_name="resultado_procesado.png",
                        mime="image/png",
                        use_container_width=True
                    )

                # Análisis avanzado opcional
                with st.expander("🔬 Análisis Avanzado"):
                    st.subheader("🧮 Histograma de la Máscara")

                    # Crear histograma
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.hist(results['generated_mask'].flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_xlabel('Valor de Píxel')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Distribución de Valores en la Máscara Generada')
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close()

                    # Análisis por zonas
                    st.subheader("📍 Análisis por Zonas")

                    h, w = results['generated_mask'].shape

                    # Dividir en 9 zonas (3x3)
                    zones = {}
                    zone_names = [
                        "Superior Izq", "Superior Centro", "Superior Der",
                        "Centro Izq", "Centro Centro", "Centro Der",
                        "Inferior Izq", "Inferior Centro", "Inferior Der"
                    ]

                    for i, name in enumerate(zone_names):
                        row = i // 3
                        col = i % 3

                        start_h = row * h // 3
                        end_h = (row + 1) * h // 3
                        start_w = col * w // 3
                        end_w = (col + 1) * w // 3

                        zone_mask = results['generated_mask'][start_h:end_h, start_w:end_w]
                        zone_coverage = np.mean(zone_mask > 127) * 100
                        zones[name] = zone_coverage

                    # Mostrar en 3 columnas
                    zone_cols = st.columns(3)
                    for i, (zone_name, coverage) in enumerate(zones.items()):
                        with zone_cols[i % 3]:
                            st.metric(zone_name, f"{coverage:.1f}%")

                    # Mapa de calor de las zonas
                    st.subheader("🗺️ Mapa de Calor por Zonas")

                    # Crear matriz 3x3 con los valores
                    heatmap_data = np.array([[zones[zone_names[i * 3 + j]] for j in range(3)] for i in range(3)])

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

                    # Agregar etiquetas
                    for i in range(3):
                        for j in range(3):
                            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                                           ha="center", va="center", color="black", fontweight='bold')

                    ax.set_xticks(range(3))
                    ax.set_yticks(range(3))
                    ax.set_xticklabels(['Izquierda', 'Centro', 'Derecha'])
                    ax.set_yticklabels(['Superior', 'Centro', 'Inferior'])
                    ax.set_title('Cobertura de Personas por Zona (%)')

                    # Colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Porcentaje de Cobertura')

                    st.pyplot(fig)
                    plt.close()

                # Consejos para mejorar resultados
                st.header("💡 Consejos para Mejores Resultados")

                advice_col1, advice_col2 = st.columns(2)

                with advice_col1:
                    st.subheader("✅ Mejores Prácticas")
                    advice_good = [
                        "🎯 Usar imágenes con personas claramente visibles",
                        "💡 Buena iluminación uniforme",
                        "🎨 Contraste claro entre persona y fondo",
                        "📐 Personas completas en el encuadre",
                        "🔍 Resolución mínima de 300x300 píxeles"
                    ]
                    for tip in advice_good:
                        st.write(tip)

                with advice_col2:
                    st.subheader("⚠️ Evitar")
                    advice_bad = [
                        "🌫️ Fondos muy similares al color de la piel/ropa",
                        "✂️ Personas parcialmente cortadas",
                        "👥 Múltiples personas superpuestas",
                        "🌈 Iluminación muy contrastada o sombras fuertes",
                        "📱 Imágenes muy pixeladas o borrosas"
                    ]
                    for warning in advice_bad:
                        st.write(warning)

                # Análisis automático de la imagen
                st.header("🤖 Análisis Automático de la Imagen")

                analysis_results = analyze_image_quality(results)

                analysis_col1, analysis_col2 = st.columns(2)

                with analysis_col1:
                    st.subheader("📊 Puntuaciones")
                    for metric, score in analysis_results['scores'].items():
                        color = "normal"
                        if score >= 80:
                            color = "inverse"
                        elif score <= 40:
                            color = "off"

                        st.metric(metric, f"{score}/100", delta=None)

                with analysis_col2:
                    st.subheader("💬 Recomendaciones")
                    for recommendation in analysis_results['recommendations']:
                        st.write(f"• {recommendation}")

                st.success("✅ ¡Procesamiento completado! Revisa todos los resultados arriba.")

            else:
                progress_bar.progress(0)
                status_text.text("❌ Error en el procesamiento")
                st.error("❌ Error al procesar la imagen")

    # Footer con información adicional
    st.markdown("---")
    with st.expander("ℹ️ Información Técnica del Modelo"):
        st.markdown("""
        ### 🏗️ Arquitectura del Modelo
        - **Base**: U-Net con encoder ResNet-34 pre-entrenado
        - **Attention Gates**: Mecanismo de atención para mejorar precisión
        - **Entrada**: 3 canales RGB
        - **Salida**: 4 canales RGBA (RGB + canal Alpha para transparencia)

        ### 🔄 Proceso de Entrenamiento
        - **Dataset**: Supervisely Persons
        - **Aumentación**: Flip horizontal, rotaciones, cambios de brillo/contraste
        - **Pérdida**: Combinación de BCE, Dice Loss, Perceptual Loss y Edge Loss
        - **Optimización**: Adam con Cosine Annealing scheduler

        ### ⚙️ Detalles de Implementación
        - **Redimensionamiento**: Mantiene proporciones con padding
        - **Restauración**: Exacta al tamaño original
        - **Procesamiento**: Optimizado para CPU y GPU
        - **Precisión**: Float32 para mayor calidad
        """)

    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🎭 Removedor de Fondo con IA - Modo Debug<br>
            Desarrollado con ❤️ usando Streamlit, PyTorch y OpenCV<br>
            Modelo U-Net con ResNet-34 y Attention Gates
        </div>
        """,
        unsafe_allow_html=True
    )


def analyze_image_quality(results):
    """
    Analiza la calidad de la imagen procesada y genera recomendaciones.

    Args:
        results: Diccionario con los resultados del procesamiento

    Returns:
        dict: Análisis con puntuaciones y recomendaciones
    """
    scores = {}
    recommendations = []

    # Analizar cobertura de personas
    mask = results['generated_mask']
    total_pixels = mask.size
    person_pixels = np.sum(mask > 127)
    coverage = (person_pixels / total_pixels) * 100

    # Puntuación de cobertura
    if coverage > 15:
        scores['Cobertura de Personas'] = min(100, coverage * 4)
    else:
        scores['Cobertura de Personas'] = coverage * 2
        recommendations.append("La persona ocupa poco espacio en la imagen. Considera usar un encuadre más cerrado.")

    # Analizar contraste de la máscara
    mask_std = np.std(mask.astype(np.float32))
    contrast_score = min(100, mask_std * 0.8)
    scores['Contraste de Máscara'] = contrast_score

    if contrast_score < 60:
        recommendations.append("Bajo contraste en la segmentación. Prueba con mejor iluminación.")

    # Analizar definición de bordes
    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_strength = np.mean(edge_magnitude)
    edge_score = min(100, edge_strength * 2)
    scores['Definición de Bordes'] = edge_score

    if edge_score < 50:
        recommendations.append(
            "Bordes poco definidos. Asegúrate de que haya buen contraste entre la persona y el fondo.")

    # Analizar distribución en la imagen
    original_h, original_w = results['original_size']
    aspect_ratio = original_w / original_h

    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        scores['Proporción de Imagen'] = 60
        recommendations.append(
            "Proporción de imagen extrema. Las imágenes más cuadradas suelen dar mejores resultados.")
    else:
        scores['Proporción de Imagen'] = 90

    # Analizar resolución
    resolution_score = min(100, (original_h * original_w) / 90000 * 100)  # Normalizado para 300x300
    scores['Resolución'] = resolution_score

    if resolution_score < 70:
        recommendations.append("Resolución baja. Imágenes de mayor resolución producen mejores resultados.")

    # Puntuación general
    overall_score = np.mean(list(scores.values()))
    scores['Puntuación General'] = overall_score

    # Recomendaciones generales
    if overall_score >= 85:
        recommendations.append("¡Excelente calidad! La imagen es ideal para remoción de fondo.")
    elif overall_score >= 70:
        recommendations.append("Buena calidad. Resultados satisfactorios.")
    elif overall_score >= 50:
        recommendations.append("Calidad moderada. Considera las sugerencias para mejorar.")
    else:
        recommendations.append("Calidad baja. Se recomienda usar una imagen diferente.")

    if not recommendations:
        recommendations.append("La imagen tiene buena calidad para el procesamiento.")

    return {
        'scores': {k: int(v) for k, v in scores.items()},
        'recommendations': recommendations
    }


if __name__ == "__main__":
    main()