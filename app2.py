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
from harmonization import HarmonizationInference

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
            alpha_restored = cv2.resize(alpha_unpadded, (w, h), interpolation=cv2.INTER_LINEAR)

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
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_edge_strength(mask):
    """Calcular fuerza de bordes en máscara."""
    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.mean(edge_magnitude)


def calculate_mask_similarity(mask1, mask2):
    """Calcular similitud entre dos máscaras (IoU)."""
    mask1_binary = (mask1 > 127).astype(np.uint8)
    mask2_binary = (mask2 > 127).astype(np.uint8)

    intersection = np.sum(mask1_binary & mask2_binary)
    union = np.sum(mask1_binary | mask2_binary)

    if union == 0:
        return 0.0

    iou = (intersection / union) * 100
    return iou


def compose_with_background(foreground_rgba, background_rgb, scale_person=0.33, align="bottom"):
    """
    Compone foreground sobre background con escala y alineación.

    Args:
        foreground_rgba: numpy array (H, W, 4) - RGBA
        background_rgb: numpy array (H, W, 3) - RGB
        scale_person: Proporción del foreground respecto al background (default: 0.33 = 1:3)
        align: "bottom", "center", "top" - Alineación vertical

    Returns:
        composite_rgb: numpy array (H, W, 3) - RGB compuesto
        metadata: dict con información de la composición
    """
    fg_h, fg_w = foreground_rgba.shape[:2]

    # Calcular tamaño del background (3x el foreground por defecto)
    bg_h = int(fg_h / scale_person)
    bg_w = int(fg_w / scale_person)

    # Redimensionar background manteniendo aspect ratio
    bg_aspect = background_rgb.shape[1] / background_rgb.shape[0]
    target_aspect = bg_w / bg_h

    if bg_aspect > target_aspect:
        # Background más ancho, ajustar por altura
        new_h = bg_h
        new_w = int(bg_h * bg_aspect)
    else:
        # Background más alto, ajustar por ancho
        new_w = bg_w
        new_h = int(bg_w / bg_aspect)

    background_resized = cv2.resize(background_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Recortar al tamaño objetivo (crop central)
    crop_x = (new_w - bg_w) // 2
    crop_y = (new_h - bg_h) // 2
    background_cropped = background_resized[crop_y:crop_y+bg_h, crop_x:crop_x+bg_w]

    # Separar canales RGB y Alpha
    fg_rgb = foreground_rgba[:, :, :3].astype(np.float32) / 255.0
    fg_alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0
    bg_float = background_cropped.astype(np.float32) / 255.0

    # Expandir alpha a 3 canales
    alpha_3ch = fg_alpha[:, :, np.newaxis]

    # Calcular posición según alineación
    if align == "bottom":
        y_offset = bg_h - fg_h  # Base inferior
        x_offset = (bg_w - fg_w) // 2  # Centrado horizontal
    elif align == "top":
        y_offset = 0
        x_offset = (bg_w - fg_w) // 2
    else:  # center
        y_offset = (bg_h - fg_h) // 2
        x_offset = (bg_w - fg_w) // 2

    # Alpha blending
    composite = bg_float.copy()
    composite[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w] = (
        fg_rgb * alpha_3ch +
        bg_float[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w] * (1 - alpha_3ch)
    )

    # Convertir de vuelta a uint8
    composite_rgb = (composite * 255).astype(np.uint8)

    metadata = {
        'fg_size': (fg_h, fg_w),
        'bg_size': (bg_h, bg_w),
        'fg_position': (y_offset, x_offset),
        'scale': scale_person
    }

    return composite_rgb, metadata


# ============================================================================
# CARGA DE MODELOS (CON CACHE)
# ============================================================================

@st.cache_resource
def load_model_aisegment():
    """Cargar modelo AISegment (retratos profesionales)."""
    model_path = 'checkpoints/aisegment_full_optimized_20gb_aggressive/best_segmentation.pth'

    if not os.path.exists(model_path):
        st.error(f"❌ Modelo AISegment no encontrado en: {model_path}")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SegmentationInference(model_path, device, image_size=384)


@st.cache_resource
def load_model_coco():
    """Cargar modelo COCO (personas en contextos variados)."""
    model_path = 'checkpoints/resnet50/best_segmentation.pth'

    if not os.path.exists(model_path):
        st.error(f"❌ Modelo COCO no encontrado en: {model_path}")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SegmentationInference(model_path, device, image_size=384)


def get_available_vram_gb():
    """Detectar VRAM disponible en GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0


@st.cache_resource
def load_models(load_both=False):
    """
    Cargar modelo(s) según configuración.

    Args:
        load_both: Si True, intenta cargar ambos modelos si hay VRAM suficiente

    Returns:
        dict con modelos cargados y estado
    """
    vram_gb = get_available_vram_gb()

    if load_both and vram_gb >= 12:
        # VRAM suficiente, cargar ambos
        return {
            'aisegment': load_model_aisegment(),
            'coco': load_model_coco(),
            'both_loaded': True
        }
    else:
        # VRAM limitada o modo único, solo cargar AISegment por defecto
        return {
            'aisegment': load_model_aisegment(),
            'coco': None,
            'both_loaded': False
        }


@st.cache_resource
def load_harmonizer():
    """Cargar modelo de armonización."""
    model_path = 'checkpoints/best_harmonizer.pth'

    if not os.path.exists(model_path):
        st.warning(f"⚠️ Modelo de armonización no encontrado en: {model_path}")
        st.info("La composición estará disponible sin armonización IA")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        harmonizer = HarmonizationInference(model_path, device)
        return harmonizer
    except Exception as e:
        st.error(f"Error cargando armonizador: {str(e)}")
        return None


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación Streamlit."""

    # Header
    st.title("✂️ Segmentación de Personas con IA - Comparación de Modelos")
    st.markdown("**Comparación:** AISegment (Retratos) vs COCO (General) | U-Net ResNet-50 + Attention Gates")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información de Modelos")

        with st.expander("📋 Detalles de los Modelos", expanded=False):
            st.markdown("""
            ### 1. AISegment (Retratos)
            - **Dataset:** 34,425 retratos profesionales
            - **Matting:** Alta calidad con canal alpha suave
            - **Mejor para:** Selfies, fotos de personas, retratos
            - **Entrenamiento:** Aggressive (33 epochs, batch 24)

            ### 2. COCO (General)
            - **Dataset:** 118K+ personas en contextos variados
            - **Escenas:** Naturales, poses diversas
            - **Mejor para:** Cualquier contexto, escenas generales
            - **Métricas:** Val IoU 0.805, Val Dice 0.890

            ### Arquitectura Común:
            - U-Net con ResNet-50 (457 capas)
            - Attention Gates para mayor precisión
            - Salida: RGBA (RGB + canal Alpha)
            - Procesamiento: 384x384 píxeles

            ### Tecnología:
            - PyTorch
            - OpenCV
            - Streamlit
            """)

        #  Selección de Modelo
        st.header("🎯 Selección de Modelo")

        operation_mode = st.radio(
            "Modo de operación:",
            ["🎯 Modelo único", "🆚 Comparar ambos"],
            help="Usar un modelo o comparar ambos lado a lado"
        )

        selected_model = None
        if operation_mode == "🎯 Modelo único":
            selected_model = st.radio(
                "Modelo a usar:",
                ["AISegment (Retratos)", "COCO (General)"],
                index=0,
                help="Selecciona el modelo que mejor se ajuste a tu imagen"
            )

        # Advertencia de VRAM para comparación
        if operation_mode == "🆚 Comparar ambos":
            vram = get_available_vram_gb()
            if vram > 0:
                if vram < 12:
                    st.warning(f"⚠️ VRAM detectada: {vram:.1f}GB. Se recomienda 12GB+ para comparación simultánea.")
                    st.info("💡 Los modelos se cargarán secuencialmente.")
                else:
                    st.success(f"✅ VRAM: {vram:.1f}GB - Suficiente para ambos modelos")
            else:
                st.info("💻 Modo CPU detectado")

        # Configuración
        st.header("⚙️ Configuración")
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Dispositivo: {device}")

        show_mask = st.checkbox("Mostrar máscara", value=True)
        show_original = st.checkbox("Mostrar imagen original", value=True)

    # Cargar modelo(s)
    with st.spinner("Cargando modelo(s)..."):
        compare_mode = operation_mode == "🆚 Comparar ambos"
        models = load_models(load_both=compare_mode)

        segmenter_aisegment = models['aisegment']
        segmenter_coco = models['coco']
        both_loaded = models['both_loaded']

    # Validar disponibilidad de modelos
    if segmenter_aisegment is None and segmenter_coco is None:
        st.error("❌ No se pudo cargar ningún modelo. Verifica los checkpoints.")
        st.stop()

    # Mostrar estado de modelos cargados
    if segmenter_aisegment and segmenter_aisegment.model_loaded:
        st.success("✅ Modelo AISegment cargado correctamente")
    else:
        st.error("❌ Modelo AISegment no disponible")

    if compare_mode or selected_model == "COCO (General)":
        if segmenter_coco and segmenter_coco.model_loaded:
            st.success("✅ Modelo COCO cargado correctamente")
        else:
            st.warning("⚠️ Modelo COCO no disponible (se cargará bajo demanda)")
            if selected_model == "COCO (General)" and segmenter_coco is None:
                with st.spinner("Cargando modelo COCO..."):
                    segmenter_coco = load_model_coco()
                    if segmenter_coco and segmenter_coco.model_loaded:
                        st.success("✅ Modelo COCO cargado correctamente")
                    else:
                        st.error("❌ No se pudo cargar el modelo COCO")
                        st.stop()

    if compare_mode and not both_loaded:
        st.info("ℹ️ Modo comparación secuencial activado (VRAM limitada)")

    # ========================================
    # SECCIÓN 1: UPLOADERS (LADO A LADO)
    # ========================================
    st.header("📸 Selecciona las imágenes")

    col_person, col_background = st.columns(2)

    # ---------- COLUMNA: IMAGEN DE PERSONA ----------
    with col_person:
        st.subheader("👤 Imagen de Persona")

        input_method = st.radio(
            "¿Cómo quieres proporcionar la imagen?",
            ["📤 Subir archivo", "📷 Capturar con cámara"],
            horizontal=True,
            key="person_input_method"
        )

        original_image = None
        image_source = None

        if input_method == "📤 Subir archivo":
            uploaded_file = st.file_uploader(
                "Sube una imagen con persona(s)...",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Sube una imagen que contenga personas para segmentación",
                key="person_uploader"
            )

            if uploaded_file is not None:
                image_source = "archivo"
                original_image = Image.open(uploaded_file)

                # Mostrar preview
                st.image(original_image, caption="Imagen seleccionada", use_container_width=True)

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
            st.info("📸 Permite el acceso a la cámara cuando el navegador lo solicite")

            camera_photo = st.camera_input("Toma una foto", key="camera_person")

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

    # ---------- COLUMNA: IMAGEN DE FONDO ----------
    with col_background:
        st.subheader("🖼️ Imagen de Fondo (Opcional)")

        st.info("💡 Si subes un fondo, se aplicará composición y armonización automáticamente")

        background_file = st.file_uploader(
            "Sube una imagen de fondo...",
            type=['png', 'jpg', 'jpeg', 'webp'],
            key="background_uploader",
            help="Opcional: Sube el fondo sobre el cual quieres fusionar la persona segmentada"
        )

        background_image = None
        background_rgb = None

        if background_file is not None:
            background_image = Image.open(background_file)
            background_rgb = np.array(background_image.convert('RGB'))

            # Mostrar preview
            st.image(background_image, caption="Fondo seleccionado", use_container_width=True)

            # Mostrar información del archivo
            with st.expander("📋 Información del fondo"):
                bg_info = {
                    "Nombre": background_file.name,
                    "Tamaño": f"{background_file.size / 1024:.2f} KB",
                    "Tipo": background_file.type,
                    "Dimensiones": f"{background_image.size[0]}x{background_image.size[1]}"
                }
                for key, value in bg_info.items():
                    st.write(f"- **{key}:** {value}")

    # ========================================
    # SECCIÓN 2: CONFIGURACIÓN UNIFICADA
    # ========================================
    if original_image is not None:
        st.markdown("---")
        st.header("⚙️ Configuración del Procesamiento")

        # Configuraciones en expander
        with st.expander("🎛️ Configurar parámetros", expanded=background_image is not None):
            config_col1, config_col2 = st.columns(2)

            with config_col1:
                st.subheader("🎯 Composición")

                # Configuración solo si hay background
                if background_image is not None:
                    # Selección de modelo para composición
                    if operation_mode == "🆚 Comparar ambos":
                        model_for_composition = st.radio(
                            "Usar resultado de:",
                            ["AISegment", "COCO"],
                            index=0,
                            help="Elige qué resultado de segmentación usar para la composición"
                        )
                    else:
                        model_for_composition = "AISegment" if selected_model == "AISegment (Retratos)" else "COCO"
                        st.info(f"Se usará resultado de: **{model_for_composition}**")

                    # Configuración de escala
                    scale_person = st.slider(
                        "Tamaño de la persona",
                        min_value=0.1,
                        max_value=0.8,
                        value=0.33,
                        step=0.05,
                        help="0.33 = persona ocupa 1/3 del fondo (recomendado)"
                    )

                    # Alineación
                    alignment = st.selectbox(
                        "Alineación vertical",
                        ["bottom", "center", "top"],
                        index=0,
                        help="Posición vertical de la persona en el fondo"
                    )
                else:
                    st.info("⏭️ Sube una imagen de fondo para activar opciones de composición")
                    model_for_composition = None
                    scale_person = 0.33
                    alignment = "bottom"

            with config_col2:
                st.subheader("🎨 Armonización")

                if background_image is not None:
                    # Factor de armonización
                    blend_factor = st.slider(
                        "Intensidad armonización",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="0.0 = sin armonización, 1.0 = completa"
                    )

                    preserve_sharpness = st.checkbox(
                        "Preservar nitidez",
                        value=True,
                        help="Aplicar filtro de nitidez tras armonización"
                    )
                else:
                    st.info("⏭️ La armonización se aplicará automáticamente si subes un fondo")
                    blend_factor = 0.7
                    preserve_sharpness = True

        # ========================================
        # BOTÓN ÚNICO DE PROCESAMIENTO
        # ========================================
        st.markdown("---")

        # Texto dinámico del botón
        if background_image is not None:
            button_text = "🚀 PROCESAR TODO (Segmentación + Composición + Armonización)"
            button_help = "Procesará la segmentación, compondrá con el fondo y aplicará armonización automática"
        else:
            button_text = "🚀 PROCESAR SEGMENTACIÓN"
            button_help = "Procesará solo la segmentación. Sube un fondo para activar composición y armonización"

        if st.button(button_text, type="primary", use_container_width=True, help=button_help):

            # ========================================
            # PIPELINE UNIFICADO DE PROCESAMIENTO
            # ========================================
            progress_bar = st.progress(0)
            status_text = st.empty()

            import time
            start_time = time.time()

            # Variables para resultados
            results_aisegment = None
            results_coco = None
            composition_simple = None
            composition_harmonized = None
            composition_metadata = None

            # ========== PASO 1: SEGMENTACIÓN ==========
            status_text.text("🔄 Paso 1/3: Preparando imagen...")
            progress_bar.progress(10)

            # MODO COMPARACIÓN
            if operation_mode == "🆚 Comparar ambos":
                status_text.text("🧠 Paso 1/3: Segmentación con AISegment...")
                progress_bar.progress(15)

                if segmenter_aisegment:
                    results_aisegment = segmenter_aisegment.process_image(original_image)

                progress_bar.progress(25)
                status_text.text("🧠 Paso 1/3: Segmentación con COCO...")

                if segmenter_coco:
                    results_coco = segmenter_coco.process_image(original_image)
                elif not both_loaded and segmenter_coco is None:
                    with st.spinner("Cargando modelo COCO..."):
                        segmenter_coco = load_model_coco()
                        if segmenter_coco:
                            results_coco = segmenter_coco.process_image(original_image)

                progress_bar.progress(40)

            # MODO ÚNICO
            else:
                status_text.text("🧠 Paso 1/3: Segmentación con IA...")
                progress_bar.progress(20)

                if selected_model == "AISegment (Retratos)":
                    results_aisegment = segmenter_aisegment.process_image(original_image)
                else:  # COCO
                    results_coco = segmenter_coco.process_image(original_image)

                progress_bar.progress(40)

            status_text.text("✅ Paso 1/3: Segmentación completada")

            # ========== PASO 2: COMPOSICIÓN (SI HAY BACKGROUND) ==========
            if background_image is not None:
                progress_bar.progress(45)
                status_text.text("🎨 Paso 2/3: Componiendo con fondo...")

                # Seleccionar resultado según configuración
                if model_for_composition == "AISegment":
                    segmented_result = results_aisegment
                elif model_for_composition == "COCO":
                    segmented_result = results_coco
                else:
                    # Por defecto usar el disponible
                    segmented_result = results_aisegment if results_aisegment else results_coco

                if segmented_result is not None:
                    foreground_rgba = segmented_result['segmented_rgba']

                    # Composición simple
                    composition_simple, composition_metadata = compose_with_background(
                        foreground_rgba,
                        background_rgb,
                        scale_person=scale_person,
                        align=alignment
                    )

                    progress_bar.progress(60)
                    status_text.text("✅ Paso 2/3: Composición completada")

                    # ========== PASO 3: ARMONIZACIÓN ==========
                    progress_bar.progress(65)
                    status_text.text("✨ Paso 3/3: Aplicando armonización...")

                    # Cargar harmonizer si es necesario
                    harmonizer = load_harmonizer()

                    if harmonizer:
                        composition_harmonized = harmonizer.harmonize_composition(
                            composition_simple,
                            blend_factor=blend_factor,
                            preserve_sharpness=preserve_sharpness
                        )
                        progress_bar.progress(90)
                        status_text.text("✅ Paso 3/3: Armonización completada")
                    else:
                        st.warning("⚠️ No se pudo cargar el harmonizer. Solo se mostrará composición simple.")
                        composition_harmonized = None
                        progress_bar.progress(90)
                else:
                    st.error("❌ No hay resultado de segmentación disponible para composición")
                    progress_bar.progress(90)
            else:
                # Sin background, solo segmentación
                progress_bar.progress(90)
                status_text.text("⏭️ Pasos 2-3 omitidos (sin fondo)")

            progress_bar.progress(100)
            status_text.text("✅ ¡Pipeline completado!")

            elapsed_time = time.time() - start_time
            st.success(f"⏱️ Tiempo total: {elapsed_time:.2f}s")

            # Verificar que al menos un resultado esté disponible
            if results_aisegment is None and results_coco is None:
                st.error("❌ Error al procesar la imagen con ambos modelos")
                progress_bar.progress(0)
                status_text.text("")
            else:

                # ========================================
                # SECCIÓN 3: RESULTADOS EN TABS
                # ========================================
                st.markdown("---")
                st.header("📊 Resultados del Procesamiento")

                # Crear tabs dinámicamente según el contenido disponible
                tab_names = []
                tab_names.append("✂️ Segmentación")
                if background_image is not None and composition_simple is not None:
                    tab_names.append("🎨 Composición")
                tab_names.append("📈 Estadísticas")
                tab_names.append("📥 Descargas")

                tabs = st.tabs(tab_names)
                tab_idx = 0

                # ========== TAB 1: SEGMENTACIÓN ==========
                with tabs[tab_idx]:
                    tab_idx += 1

                    # MODO COMPARACIÓN
                    if operation_mode == "🆚 Comparar ambos":
                        st.subheader("🆚 Comparación: AISegment vs COCO")

                        # Fila 1: Imágenes segmentadas
                        col_orig, col_aiseg, col_coco = st.columns(3)

                        with col_orig:
                            st.markdown("**📷 Original**")
                            st.image(original_image, use_container_width=True)

                        with col_aiseg:
                            st.markdown("**✂️ AISegment (Retratos)**")
                            if results_aisegment:
                                st.image(results_aisegment['segmented_rgba'], use_container_width=True)

                                # Métricas AISegment
                                mask_ai = results_aisegment['mask']
                                coverage_ai = (np.sum(mask_ai > 127) / mask_ai.size) * 100
                                edge_ai = calculate_edge_strength(mask_ai)

                                st.metric("Cobertura", f"{coverage_ai:.1f}%")
                                st.metric("Definición", f"{edge_ai:.1f}")
                            else:
                                st.error("No disponible")

                        with col_coco:
                            st.markdown("**✂️ COCO (General)**")
                            if results_coco:
                                st.image(results_coco['segmented_rgba'], use_container_width=True)

                                # Métricas COCO
                                mask_coco = results_coco['mask']
                                coverage_coco = (np.sum(mask_coco > 127) / mask_coco.size) * 100
                                edge_coco = calculate_edge_strength(mask_coco)

                                st.metric("Cobertura", f"{coverage_coco:.1f}%")
                                st.metric("Definición", f"{edge_coco:.1f}")
                            else:
                                st.error("No disponible")

                        # Máscaras (si está habilitado)
                        if show_mask:
                            st.markdown("---")
                            st.markdown("**🎭 Máscaras de Segmentación**")

                            col_mask_ai, col_mask_coco = st.columns(2)

                            with col_mask_ai:
                                if results_aisegment:
                                    st.image(results_aisegment['mask'], use_container_width=True, clamp=True, caption="Máscara AISegment")

                            with col_mask_coco:
                                if results_coco:
                                    st.image(results_coco['mask'], use_container_width=True, clamp=True, caption="Máscara COCO")

                    # MODO ÚNICO
                    else:
                        active_result = results_aisegment if selected_model == "AISegment (Retratos)" else results_coco

                        # Crear columnas para mostrar imágenes
                        num_cols = sum([show_original, True, show_mask])
                        cols = st.columns(num_cols)
                        col_idx = 0

                        # Imagen original
                        if show_original:
                            with cols[col_idx]:
                                st.markdown("**📷 Original**")
                                st.image(active_result['original_image'], use_container_width=True)
                                st.caption(f"Tamaño: {active_result['original_size']}")
                            col_idx += 1

                        # Imagen segmentada (RGBA)
                        with cols[col_idx]:
                            st.markdown("**✂️ Imagen Cortada**")
                            st.image(active_result['segmented_rgba'], use_container_width=True)
                            st.caption(f"Modelo: {selected_model}")
                        col_idx += 1

                        # Máscara
                        if show_mask:
                            with cols[col_idx]:
                                st.markdown("**🎭 Máscara**")
                                st.image(active_result['mask'], use_container_width=True, clamp=True)
                                st.caption("Canal Alpha de segmentación")

                # ========== TAB 2: COMPOSICIÓN (si hay background) ==========
                if background_image is not None and composition_simple is not None:
                    with tabs[tab_idx]:
                        tab_idx += 1

                        st.subheader("🎨 Resultados de Composición y Armonización")

                        # Mostrar composiciones
                        col_comp, col_harm = st.columns(2)

                        with col_comp:
                            st.markdown("**🖼️ Composición Simple**")
                            st.image(composition_simple, use_container_width=True)
                            st.caption("Fusión básica sin armonización")

                        with col_harm:
                            st.markdown("**✨ Composición Armonizada**")
                            if composition_harmonized is not None:
                                st.image(composition_harmonized, use_container_width=True)
                                st.caption(f"Con armonización IA (intensidad: {blend_factor})")
                            else:
                                st.info("⚠️ Armonización no aplicada")

                        # Información de la composición
                        if composition_metadata:
                            with st.expander("📋 Detalles de la composición"):
                                st.write(f"- **Tamaño foreground**: {composition_metadata['fg_size']}")
                                st.write(f"- **Tamaño background**: {composition_metadata['bg_size']}")
                                st.write(f"- **Posición**: {composition_metadata['fg_position']}")
                                st.write(f"- **Escala**: 1:{1/composition_metadata['scale']:.1f}")
                                st.write(f"- **Alineación**: {alignment}")

                # ========== TAB 3: ESTADÍSTICAS ==========
                with tabs[tab_idx if background_image is not None and composition_simple is not None else tab_idx]:
                    if background_image is not None and composition_simple is not None:
                        tab_idx += 1

                    st.subheader("📈 Análisis Estadístico")

                    # MODO COMPARACIÓN
                    if operation_mode == "🆚 Comparar ambos":
                        if results_aisegment and results_coco:
                            mask_ai = results_aisegment['mask']
                            mask_coco = results_coco['mask']
                            coverage_ai = (np.sum(mask_ai > 127) / mask_ai.size) * 100
                            coverage_coco = (np.sum(mask_coco > 127) / mask_coco.size) * 100
                            edge_ai = calculate_edge_strength(mask_ai)
                            edge_coco = calculate_edge_strength(mask_coco)

                            st.markdown("**🆚 Análisis Comparativo**")

                            col_a1, col_a2, col_a3, col_a4 = st.columns(4)

                            with col_a1:
                                better_coverage = "AISegment" if coverage_ai > coverage_coco else "COCO"
                                diff_coverage = abs(coverage_ai - coverage_coco)
                                st.metric(
                                    "Mejor Cobertura",
                                    better_coverage,
                                    f"+{diff_coverage:.1f}%"
                                )

                            with col_a2:
                                better_edges = "AISegment" if edge_ai > edge_coco else "COCO"
                                diff_edges = abs(edge_ai - edge_coco)
                                st.metric(
                                    "Mejor Definición",
                                    better_edges,
                                    f"+{diff_edges:.1f}"
                                )

                            with col_a3:
                                # Calcular similitud entre máscaras
                                similarity = calculate_mask_similarity(mask_ai, mask_coco)
                                st.metric(
                                    "Similitud IoU",
                                    f"{similarity:.1f}%",
                                    help="Similitud entre ambas máscaras"
                                )

                            with col_a4:
                                # Puntuación de calidad combinada
                                quality_ai = (coverage_ai * 0.5 + edge_ai * 0.5)
                                quality_coco = (coverage_coco * 0.5 + edge_coco * 0.5)
                                winner = "AISegment" if quality_ai > quality_coco else "COCO"
                                st.metric(
                                    "Mejor Overall",
                                    winner
                                )

                            # Recomendación
                            st.markdown("💡 **Recomendación:**")
                            if quality_ai > quality_coco + 5:
                                st.info(f"**AISegment** muestra mejor resultado ({quality_ai - quality_coco:.1f} puntos de ventaja)")
                            elif quality_coco > quality_ai + 5:
                                st.info(f"**COCO** muestra mejor resultado ({quality_coco - quality_ai:.1f} puntos de ventaja)")
                            else:
                                st.info("Ambos modelos tienen resultados similares")

                    # MODO ÚNICO
                    else:
                        active_result = results_aisegment if selected_model == "AISegment (Retratos)" else results_coco

                        if active_result:
                            mask = active_result['mask']
                            total_pixels = mask.size
                            person_pixels = np.sum(mask > 127)
                            coverage = (person_pixels / total_pixels) * 100
                            mask_mean = np.mean(mask)
                            mask_std = np.std(mask.astype(np.float32))
                            edge_strength = calculate_edge_strength(mask)

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Cobertura", f"{coverage:.1f}%")

                            with col2:
                                st.metric("Píxeles de Persona", f"{person_pixels:,}")

                            with col3:
                                st.metric("Brillo Promedio", f"{mask_mean:.1f}")

                            with col4:
                                st.metric("Contraste", f"{mask_std:.1f}")

                            # Análisis de calidad
                            st.markdown("---")
                            st.markdown("**🔬 Análisis de Calidad**")

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

                            # Histograma (opcional)
                            with st.expander("📊 Histograma de la Máscara"):
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(10, 4))

                                ax.hist(mask.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                                ax.set_xlabel('Valor de Píxel')
                                ax.set_ylabel('Frecuencia')
                                ax.set_title('Distribución de Valores en la Máscara')
                                ax.grid(True, alpha=0.3)

                                st.pyplot(fig)
                                plt.close()

                # ========== TAB 4: DESCARGAS ==========
                with tabs[-1]:  # Último tab
                    st.subheader("📥 Descargar Resultados")

                    # MODO COMPARACIÓN: Descargas de ambos modelos
                    if operation_mode == "🆚 Comparar ambos":
                        st.markdown("**Descargas de Segmentación:**")
                        download_col1, download_col2, download_col3, download_col4 = st.columns(4)

                        with download_col1:
                            if results_aisegment:
                                result_pil = Image.fromarray(results_aisegment['segmented_rgba'], 'RGBA')
                                buf = io.BytesIO()
                                result_pil.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 AISegment (PNG)",
                                    data=buf.getvalue(),
                                    file_name="aisegment_segmentada.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                        with download_col2:
                            if results_aisegment:
                                mask_pil = Image.fromarray(results_aisegment['mask'], 'L')
                                buf = io.BytesIO()
                                mask_pil.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 Máscara AI",
                                    data=buf.getvalue(),
                                    file_name="aisegment_mascara.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                        with download_col3:
                            if results_coco:
                                result_pil = Image.fromarray(results_coco['segmented_rgba'], 'RGBA')
                                buf = io.BytesIO()
                                result_pil.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 COCO (PNG)",
                                    data=buf.getvalue(),
                                    file_name="coco_segmentada.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                        with download_col4:
                            if results_coco:
                                mask_pil = Image.fromarray(results_coco['mask'], 'L')
                                buf = io.BytesIO()
                                mask_pil.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 Máscara COCO",
                                    data=buf.getvalue(),
                                    file_name="coco_mascara.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                    # MODO ÚNICO: Descargas del modelo seleccionado
                    else:
                        active_result = results_aisegment if selected_model == "AISegment (Retratos)" else results_coco

                        if active_result:
                            st.markdown("**Descargas de Segmentación:**")
                            download_col1, download_col2 = st.columns(2)

                            with download_col1:
                                # Descargar imagen cortada (RGBA)
                                result_pil = Image.fromarray(active_result['segmented_rgba'], 'RGBA')
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
                                mask_pil = Image.fromarray(active_result['mask'], 'L')
                                buf_mask = io.BytesIO()
                                mask_pil.save(buf_mask, format='PNG')

                                st.download_button(
                                    label="📥 Descargar Máscara (PNG)",
                                    data=buf_mask.getvalue(),
                                    file_name="mascara.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                    # Descargas de composición (si están disponibles)
                    if background_image is not None and composition_simple is not None:
                        st.markdown("---")
                        st.markdown("**Descargas de Composición:**")

                        download_col1, download_col2 = st.columns(2)

                        with download_col1:
                            # Composición simple
                            comp_pil = Image.fromarray(composition_simple, 'RGB')
                            buf_comp = io.BytesIO()
                            comp_pil.save(buf_comp, format='PNG')

                            st.download_button(
                                label="📥 Descargar Composición Simple",
                                data=buf_comp.getvalue(),
                                file_name="composicion_simple.png",
                                mime="image/png",
                                use_container_width=True
                            )

                        with download_col2:
                            # Composición armonizada
                            if composition_harmonized is not None:
                                harm_pil = Image.fromarray(composition_harmonized, 'RGB')
                                buf_harm = io.BytesIO()
                                harm_pil.save(buf_harm, format='PNG')

                                st.download_button(
                                    label="📥 Descargar Composición Armonizada",
                                    data=buf_harm.getvalue(),
                                    file_name="composicion_armonizada.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            else:
                                st.info("⏭️ Armonización no disponible")

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