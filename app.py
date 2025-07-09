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
from smartComposition import smart_composite_arrays

warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="Procesador de Imágenes IA - Segmentación, Composición y Harmonización",
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

    @staticmethod
    def composite_foreground_background(foreground_rgba, background_rgb):
        """
        Compone una imagen RGBA sobre un fondo RGB.
        
        Args:
            foreground_rgba: Imagen RGBA del foreground (H, W, 4)
            background_rgb: Imagen RGB del background (H, W, 3)
            
        Returns:
            Imagen RGB compuesta (H, W, 3)
        """
        # Asegurar que las imágenes tengan el mismo tamaño
        if foreground_rgba.shape[:2] != background_rgb.shape[:2]:
            background_rgb = cv2.resize(background_rgb, (foreground_rgba.shape[1], foreground_rgba.shape[0]))
        
        # Extraer canales
        foreground_rgb = foreground_rgba[:, :, :3].astype(np.float32) / 255.0
        alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0
        background_rgb = background_rgb.astype(np.float32) / 255.0
        
        # Normalizar alpha a [0, 1] y expandir dimensiones
        alpha = alpha[:, :, np.newaxis]
        
        # Composición alpha blending
        composite = foreground_rgb * alpha + background_rgb * (1 - alpha)
        
        # Convertir de vuelta a uint8
        composite = (composite * 255).astype(np.uint8)
        
        return composite

    @staticmethod
    def advanced_composite(foreground_rgba, background_rgb, 
                          depth_distance=1.0, scale_factor=1.0, vertical_position=0.0,
                          blur_background=False, blur_strength=5):
        """
        Composición avanzada con controles de profundidad y perspectiva.
        
        Args:
            foreground_rgba: Imagen RGBA del foreground (H, W, 4)
            background_rgb: Imagen RGB del background (H, W, 3)
            depth_distance: Factor de distancia percibida (1.0 = normal)
            scale_factor: Factor de escala del sujeto
            vertical_position: Posición vertical (-1.0 a 1.0)
            blur_background: Si aplicar blur al fondo
            blur_strength: Intensidad del blur (mayor = más blur)
            
        Returns:
            Imagen RGB compuesta (H, W, 3)
        """
        original_bg_shape = background_rgb.shape[:2]
        
        # Aplicar blur al fondo si está habilitado
        if blur_background:
            # Usar blur gaussiano para simular profundidad de campo
            ksize = max(1, int(blur_strength * 2 + 1))
            if ksize % 2 == 0:
                ksize += 1
            background_rgb = cv2.GaussianBlur(background_rgb, (ksize, ksize), blur_strength/3)
        
        # Simular efectos de distancia en el foreground
        if depth_distance != 1.0:
            # A mayor distancia, menos contraste y más blur sutil
            if depth_distance > 1.0:
                # Reducir contraste para simular distancia
                foreground_rgba[:, :, :3] = ImageProcessor._adjust_contrast(
                    foreground_rgba[:, :, :3], 
                    1.0 - (depth_distance - 1.0) * 0.3
                )
                # Blur muy sutil para objetos lejanos
                if depth_distance > 1.5:
                    blur_fg = max(1, int((depth_distance - 1.0) * 2))
                    if blur_fg % 2 == 0:
                        blur_fg += 1
                    for i in range(3):
                        foreground_rgba[:, :, i] = cv2.GaussianBlur(
                            foreground_rgba[:, :, i], (blur_fg, blur_fg), blur_fg/3
                        )
        
        # Ajustar escala del foreground
        if scale_factor != 1.0:
            new_h = int(foreground_rgba.shape[0] * scale_factor)
            new_w = int(foreground_rgba.shape[1] * scale_factor)
            foreground_rgba = cv2.resize(foreground_rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Asegurar que el background tenga el tamaño correcto
        if foreground_rgba.shape[:2] != background_rgb.shape[:2]:
            background_rgb = cv2.resize(background_rgb, (foreground_rgba.shape[1], foreground_rgba.shape[0]))
        
        # Ajustar posición vertical
        if vertical_position != 0.0:
            fg_h, fg_w = foreground_rgba.shape[:2]
            bg_h, bg_w = background_rgb.shape[:2]
            
            # Crear canvas del tamaño del background
            canvas_fg = np.zeros((bg_h, bg_w, 4), dtype=foreground_rgba.dtype)
            canvas_bg = background_rgb.copy()
            
            # Calcular offset vertical
            vertical_offset = int(vertical_position * bg_h * 0.3)  # Max 30% del alto
            
            # Calcular posiciones de colocación
            start_y = max(0, (bg_h - fg_h) // 2 + vertical_offset)
            end_y = min(bg_h, start_y + fg_h)
            start_x = max(0, (bg_w - fg_w) // 2)
            end_x = min(bg_w, start_x + fg_w)
            
            # Ajustar tamaños si es necesario
            crop_start_y = max(0, -start_y)
            crop_end_y = crop_start_y + (end_y - start_y)
            crop_start_x = max(0, -start_x)
            crop_end_x = crop_start_x + (end_x - start_x)
            
            # Colocar foreground en canvas
            canvas_fg[start_y:end_y, start_x:end_x] = foreground_rgba[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            
            foreground_rgba = canvas_fg
            background_rgb = canvas_bg
        
        # Aplicar composición básica
        foreground_rgb = foreground_rgba[:, :, :3].astype(np.float32) / 255.0
        alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0
        background_rgb = background_rgb.astype(np.float32) / 255.0
        
        # Normalizar alpha y expandir dimensiones
        alpha = alpha[:, :, np.newaxis]
        
        # Composición alpha blending
        composite = foreground_rgb * alpha + background_rgb * (1 - alpha)
        
        # Convertir de vuelta a uint8
        composite = (composite * 255).astype(np.uint8)
        
        return composite
    
    @staticmethod
    def _adjust_contrast(image, factor):
        """Ajusta el contraste de una imagen."""
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip(mean + factor * (image - mean), 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_color_adjustments(image, brightness=0, contrast=1.0, saturation=1.0, opacity=1.0):
        """
        Aplica ajustes de color y transparencia a una imagen.
        
        Args:
            image: Imagen RGB (H, W, 3)
            brightness: Ajuste de brillo (-100 a 100)
            contrast: Factor de contraste (0.1 a 3.0)
            saturation: Factor de saturación (0.0 a 3.0)
            opacity: Opacidad final (0.0 a 1.0)
            
        Returns:
            Imagen ajustada como RGBA (H, W, 4)
        """
        # Convertir a float para cálculos
        image_float = image.astype(np.float32)
        
        # Ajuste de brillo
        if brightness != 0:
            image_float = np.clip(image_float + brightness, 0, 255)
        
        # Ajuste de contraste
        if contrast != 1.0:
            # Punto medio para el contraste
            mid_point = 127.5
            image_float = np.clip(mid_point + contrast * (image_float - mid_point), 0, 255)
        
        # Ajuste de saturación
        if saturation != 1.0:
            # Convertir a HSV para ajustar saturación
            image_hsv = cv2.cvtColor(image_float.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation, 0, 255)
            image_float = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Convertir de vuelta a uint8
        image_adjusted = np.clip(image_float, 0, 255).astype(np.uint8)
        
        # Crear canal alpha basado en opacidad
        alpha_channel = np.full((image_adjusted.shape[0], image_adjusted.shape[1], 1), 
                               int(opacity * 255), dtype=np.uint8)
        
        # Combinar RGB + Alpha
        result_rgba = np.concatenate([image_adjusted, alpha_channel], axis=2)
        
        return result_rgba


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
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
            # result_final[:, :, :3] = rgb_restored
            # result_final[:, :, 3] = alpha_restored
            threshold = 0.4
            person_mask = alpha_restored > threshold

            # Usar imagen original donde hay persona detectada
            original_normalized = image_rgb.astype(np.float32) / 255.0
            rgb_enhanced = np.where(
                person_mask[..., np.newaxis],
                original_normalized,  # Colores originales
                rgb_restored  # RGB del modelo
            )

            # Suavizar máscara para transiciones naturales
            alpha_smooth = cv2.bilateralFilter(
                (alpha_restored * 255).astype(np.uint8),
                5, 50, 50
            ).astype(np.float32) / 255.0

            result_final[:, :, :3] = rgb_enhanced
            result_final[:, :, 3] = alpha_smooth

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
def load_segmentation_model():
    """Cargar el modelo de segmentación (con cache para evitar recargas)."""
    model_path = 'checkpoints/resnet34/best_segmentation.pth'

    if not os.path.exists(model_path):
        st.error("❌ Modelo de segmentación no encontrado en 'checkpoints/best_segmentation.pth'")
        st.info("Asegúrate de que el modelo esté entrenado y guardado en la ruta correcta.")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return BackgroundRemoverDebug(model_path, device)

@st.cache_resource
def load_harmonization_model():
    """Cargar el modelo de harmonización (con cache para evitar recargas)."""
    model_path = 'checkpoints/best_harmonizer.pth'

    if not os.path.exists(model_path):
        st.error("❌ Modelo de harmonización no encontrado en 'checkpoints/best_harmonizer.pth'")
        st.info("Asegúrate de que el modelo esté entrenado y guardado en la ruta correcta.")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        from harmonization import HarmonizationInference
        return HarmonizationInference(model_path, device)
    except ImportError as e:
        st.warning(f"⚠️ No se pudo importar el módulo de harmonización: {str(e)}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error cargando modelo de harmonización: {str(e)}")
        return None


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

                        st.image(pil_image, caption=stage_name, use_container_width=True)

                        # Mostrar información adicional
                        if isinstance(image_data, np.ndarray):
                            st.caption(f"Dimensiones: {image_data.shape}")
                    else:
                        st.write(f"❌ {stage_name} no disponible")


def main():
    """Función principal de la aplicación Streamlit."""

    # Header
    st.title("🎭 Estudio de Composición Digital con IA")
    st.markdown("**Pipeline profesional: Segmentación → Composición 3D → Harmonización → Post-procesamiento**")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Procesamiento")
        st.markdown("""
        ### Pipeline Profesional:
        1. **Segmentación IA** - Extracción precisa con U-Net
        2. **Composición 3D** - Perspectiva, escala y profundidad
        3. **Harmonización** - Ajuste inteligente de iluminación
        4. **Post-procesamiento** - Color, contraste y efectos finales
        
        ### Controles Avanzados:
        
        **🎭 Composición:**
        - Distancia percibida (perspectiva)
        - Escala del sujeto (tamaño relativo)
        - Posición vertical (composición)
        - Blur de fondo (profundidad de campo)
        
        **🎨 Harmonización:**
        - Intensidad de corrección
        - Preservación de nitidez
        - Mezcla inteligente
        
        **🎛️ Post-procesamiento:**
        - Brillo y contraste
        - Saturación de colores
        - Opacidad final

        ### Tecnología:
        - **Segmentación**: U-Net + ResNet-34 + Attention Gates
        - **Harmonización**: U-Net especializado en color
        - **Composición**: Alpha blending con efectos 3D
        - **Calidad**: Interpolación cúbica y filtros de nitidez
        """)

        # Configuración
        st.header("⚙️ Configuración")
        processing_size = st.slider("Tamaño de procesamiento", 128, 512, 256, 32)
        show_technical_info = st.checkbox("Mostrar información técnica", value=True)
        
        # Configuración de Composición
        with st.expander("🎭 Composición Avanzada"):
            # Controles de perspectiva y profundidad
            st.write("**Perspectiva y Profundidad:**")
            depth_distance = st.slider("Distancia percibida", 0.1, 3.0, 1.0, 0.1,
                                     help="1.0 = Normal, <1.0 = Más cerca, >1.0 = Más lejos")
            scale_factor = st.slider("Escala del sujeto", 0.3, 1.5, 1.0, 0.05,
                                   help="Ajusta el tamaño relativo del sujeto")
            vertical_position = st.slider("Posición vertical", -0.3, 0.3, 0.0, 0.05,
                                        help="Ajusta la posición vertical del sujeto")
            blur_background = st.checkbox("Desenfocar fondo (bokeh)", value=False,
                                        help="Simula profundidad de campo")
            blur_strength = st.slider("Intensidad del bokeh", 1, 15, 5, 1,
                                    help="Mayor valor = más desenfoque") if blur_background else 5
        
        # Configuración de Harmonización
        with st.expander("🎨 Harmonización"):
            harmonization_enabled = st.checkbox("Aplicar harmonización", value=True)
            blend_factor = st.slider("Intensidad de harmonización", 0.0, 1.0, 0.7, 0.1,
                                    help="0.0 = Solo composición, 1.0 = Solo harmonización")
            preserve_sharpness = st.checkbox("Preservar nitidez", value=True,
                                           help="Aplica filtros para mantener la calidad de imagen")
        
        # Ajustes de color y transparencia
        with st.expander("🎛️ Ajustes Finales"):
            col1_adj, col2_adj = st.columns(2)
            with col1_adj:
                brightness_adjust = st.slider("Brillo", -50, 50, 0, 5,
                                            help="Ajuste de brillo general")
                contrast_adjust = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1,
                                          help="Ajuste de contraste")
            with col2_adj:
                saturation_adjust = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1,
                                            help="Ajuste de saturación de colores")
                final_opacity = st.slider("Opacidad final", 0.1, 1.0, 1.0, 0.05,
                                        help="Transparencia del resultado final")

        # Información del sistema
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Usando: {device}")

    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        col1, col2 = st.columns(2)
        with col1:
            bg_remover = load_segmentation_model()
        with col2:
            harmonizer = load_harmonization_model()

    if bg_remover is None:
        st.stop()

    models_status = []
    if bg_remover is not None:
        models_status.append("✅ Modelo de segmentación cargado")
    if harmonizer is not None:
        models_status.append("✅ Modelo de harmonización cargado")
    else:
        models_status.append("⚠️ Modelo de harmonización no disponible")
    
    for status in models_status:
        st.success(status)

    # Subir imágenes
    st.header("📤 Subir Imágenes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Imagen Principal")
        uploaded_file = st.file_uploader(
            "Elige una imagen con personas...",
            type=['png', 'jpg', 'jpeg'],
            help="Sube una imagen que contenga personas para segmentación",
            key="main_image"
        )
    
    with col2:
        st.subheader("🖼️ Imagen de Fondo")
        background_file = st.file_uploader(
            "Elige un fondo (opcional)...",
            type=['png', 'jpg', 'jpeg'],
            help="Sube una imagen de fondo para composición y harmonización",
            key="background_image"
        )

    if uploaded_file is not None:
        # Mostrar información de los archivos
        with st.expander("📋 Información de los archivos"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Imagen Principal:**")
                file_info = {
                    "Nombre": uploaded_file.name,
                    "Tamaño": f"{uploaded_file.size / 1024:.2f} KB",
                    "Tipo": uploaded_file.type
                }
                for key, value in file_info.items():
                    st.write(f"- **{key}:** {value}")
            
            with col2:
                if background_file is not None:
                    st.write("**Imagen de Fondo:**")
                    bg_info = {
                        "Nombre": background_file.name,
                        "Tamaño": f"{background_file.size / 1024:.2f} KB",
                        "Tipo": background_file.type
                    }
                    for key, value in bg_info.items():
                        st.write(f"- **{key}:** {value}")
                else:
                    st.write("**Imagen de Fondo:** No seleccionada")

        # Cargar imágenes
        original_image = Image.open(uploaded_file)
        original_array = np.array(original_image)
        
        background_image = None
        background_array = None
        if background_file is not None:
            background_image = Image.open(background_file)
            background_array = np.array(background_image)

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

            # Procesar imagen con segmentación
            results = bg_remover.remove_background_debug(original_image, image_size=processing_size)
            
            progress_bar.progress(70)
            status_text.text("🎨 Procesando composición...")
            
            # Composición con fondo si está disponible
            composition_result = None
            harmonized_result = None
            
            if results is not None and background_array is not None:
                # Crear composición usando el resultado de segmentación
                foreground_rgba = results['result_final']  # Imagen RGBA segmentada
                
                # Redimensionar fondo al tamaño de la imagen principal
                bg_resized = cv2.resize(background_array, (foreground_rgba.shape[1], foreground_rgba.shape[0]))
                
                # Composición avanzada con controles de profundidad
                processor = ImageProcessor()
                composition_rgb = processor.advanced_composite(
                    foreground_rgba, bg_resized,
                    depth_distance=depth_distance,
                    scale_factor=scale_factor,
                    vertical_position=vertical_position,
                    blur_background=blur_background,
                    blur_strength=blur_strength
                )

                # Añadimos composición con IC LIGHT
                composition_rgb = smart_composite_arrays(
                    foreground_rgb=foreground_rgba,
                    background_rgb=bg_resized, 
                    model_quality="best",  # "fast", "quality", o "best"
                    shadow_intensity=0.15,
                )

                composition_result = composition_rgb
                
                progress_bar.progress(85)
                status_text.text("✨ Aplicando harmonización...")
                
                # Harmonización si el modelo está disponible y está habilitada
                if harmonizer is not None and harmonization_enabled:
                    try:
                        harmonized_rgb = harmonizer.harmonize_composition(
                            composition_rgb, 
                            blend_factor=blend_factor,
                            preserve_sharpness=preserve_sharpness
                        )
                        harmonized_result = harmonized_rgb
                    except Exception as e:
                        st.warning(f"Error en harmonización: {str(e)}")
                        harmonized_result = composition_rgb
                else:
                    harmonized_result = composition_rgb

                progress_bar.progress(95)
                status_text.text("🎨 Aplicando ajustes finales...")
                
                # Aplicar ajustes de color y transparencia al resultado final
                if harmonized_result is not None and (brightness_adjust != 0 or contrast_adjust != 1.0 or 
                                                    saturation_adjust != 1.0 or final_opacity != 1.0):
                    processor = ImageProcessor()
                    harmonized_result_rgba = processor.apply_color_adjustments(
                        harmonized_result,
                        brightness=brightness_adjust,
                        contrast=contrast_adjust,
                        saturation=saturation_adjust,
                        opacity=final_opacity
                    )
                    # Si no hay transparencia completa, convertir de vuelta a RGB
                    if final_opacity == 1.0:
                        harmonized_result = harmonized_result_rgba[:, :, :3]
                    else:
                        harmonized_result = harmonized_result_rgba

            progress_bar.progress(90)
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
                
                # Añadir imágenes de composición y harmonización si están disponibles
                if background_array is not None:
                    images_for_display["6. Fondo Original"] = background_array
                    if composition_result is not None:
                        images_for_display["7. Composición"] = composition_result
                    if harmonized_result is not None:
                        images_for_display["8. Resultado Harmonizado"] = harmonized_result

                # Mostrar todas las etapas
                st.header("📊 Resultados del Procesamiento")

                # Visualización en grilla
                stage_names = list(images_for_display.keys())
                display_image_comparison(images_for_display, stage_names, cols_per_row=3)

                # Información técnica detallada
                if show_technical_info:
                    st.header("🔬 Análisis Técnico Detallado")

                    col1, col2, col3, col4 = st.columns(4)

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

                        # Análisis de nitidez si hay composición
                        if composition_result is not None:
                            comp_sharpness = calculate_sharpness(composition_result)
                            st.metric("Nitidez Composición", f"{comp_sharpness:.1f}")
                            
                            if harmonized_result is not None and harmonization_enabled:
                                harm_sharpness = calculate_sharpness(harmonized_result)
                                st.metric("Nitidez Harmonizada", f"{harm_sharpness:.1f}")
                                
                                # Diferencia de nitidez
                                sharpness_diff = harm_sharpness - comp_sharpness
                                color = "normal" if sharpness_diff >= -10 else "off"
                                st.metric("Diferencia Nitidez", f"{sharpness_diff:+.1f}", 
                                        help="Positivo = se mantiene/mejora, Negativo = se degrada")

                        # Indicador de calidad general
                        quality_score = min(100, (coverage * 0.3 + mask_std * 0.4 + edge_strength * 0.3))
                        st.metric("Puntuación de Calidad", f"{quality_score:.0f}/100")

                    with col4:
                        st.subheader("🎛️ Efectos Aplicados")
                        
                        # Efectos de composición
                        if composition_result is not None:
                            st.write("**Composición:**")
                            if depth_distance != 1.0:
                                st.write(f"• Distancia: {depth_distance:.1f}x")
                            if scale_factor != 1.0:
                                st.write(f"• Escala: {scale_factor:.1f}x")
                            if vertical_position != 0.0:
                                st.write(f"• Posición V: {vertical_position:+.2f}")
                            if blur_background:
                                st.write(f"• Blur fondo: {blur_strength}")
                        
                        # Efectos de color
                        color_effects = []
                        if brightness_adjust != 0:
                            color_effects.append(f"Brillo: {brightness_adjust:+d}")
                        if contrast_adjust != 1.0:
                            color_effects.append(f"Contraste: {contrast_adjust:.1f}")
                        if saturation_adjust != 1.0:
                            color_effects.append(f"Saturación: {saturation_adjust:.1f}")
                        if final_opacity != 1.0:
                            color_effects.append(f"Opacidad: {final_opacity:.1f}")
                        
                        if color_effects:
                            st.write("**Ajustes de Color:**")
                            for effect in color_effects:
                                st.write(f"• {effect}")
                        
                        if not any([depth_distance != 1.0, scale_factor != 1.0, vertical_position != 0.0, 
                                  blur_background, brightness_adjust != 0, contrast_adjust != 1.0, 
                                  saturation_adjust != 1.0, final_opacity != 1.0]):
                            st.write("Sin efectos aplicados")

                # Comparación lado a lado
                st.header("🔍 Comparación de Resultados")
                
                if background_array is not None and harmonized_result is not None:
                    # Mostrar 4 columnas cuando hay composición y harmonización
                    comp_cols = st.columns(4)
                    
                    with comp_cols[0]:
                        st.subheader("📷 Original")
                        st.image(results['original_image'], use_container_width=True)
                    
                    with comp_cols[1]:
                        st.subheader("✂️ Segmentado")
                        st.image(results['result_final'], use_container_width=True)
                    
                    with comp_cols[2]:
                        st.subheader("🎨 Composición")
                        st.image(composition_result, use_container_width=True)
                        if composition_result is not None:
                            st.caption(f"Nitidez: {calculate_sharpness(composition_result):.2f}")
                            # Mostrar parámetros de composición aplicados
                            if depth_distance != 1.0 or scale_factor != 1.0 or vertical_position != 0.0 or blur_background:
                                params = []
                                if depth_distance != 1.0:
                                    params.append(f"Dist: {depth_distance:.1f}")
                                if scale_factor != 1.0:
                                    params.append(f"Escala: {scale_factor:.1f}")
                                if vertical_position != 0.0:
                                    params.append(f"Pos: {vertical_position:+.2f}")
                                if blur_background:
                                    params.append(f"Blur: {blur_strength}")
                                st.caption(" | ".join(params))
                    
                    with comp_cols[3]:
                        if harmonization_enabled and harmonized_result is not None:
                            st.subheader("✨ Harmonizado")
                            st.image(harmonized_result, use_container_width=True)
                            if len(harmonized_result.shape) == 3 and harmonized_result.shape[2] == 3:
                                st.caption(f"Nitidez: {calculate_sharpness(harmonized_result):.2f}")
                            st.caption(f"Factor: {blend_factor:.1f}")
                            # Mostrar ajustes aplicados
                            adjustments = []
                            if brightness_adjust != 0:
                                adjustments.append(f"B:{brightness_adjust:+d}")
                            if contrast_adjust != 1.0:
                                adjustments.append(f"C:{contrast_adjust:.1f}")
                            if saturation_adjust != 1.0:
                                adjustments.append(f"S:{saturation_adjust:.1f}")
                            if final_opacity != 1.0:
                                adjustments.append(f"O:{final_opacity:.1f}")
                            if adjustments:
                                st.caption(" | ".join(adjustments))
                        else:
                            st.subheader("⚠️ Harmonización")
                            st.write("Deshabilitada")
                else:
                    # Mostrar 2 columnas cuando solo hay segmentación
                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.subheader("📷 Imagen Original")
                        st.image(results['original_image'], use_container_width=True)

                    with comp_col2:
                        st.subheader("✨ Resultado Final")
                        st.image(results['result_final'], use_container_width=True)

                # Opciones de descarga
                st.header("📥 Descargar Resultados")

                if background_array is not None and harmonized_result is not None:
                    download_cols = st.columns(4)
                else:
                    download_cols = st.columns(3)

                with download_cols[0]:
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

                with download_cols[1]:
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

                with download_cols[2]:
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

                # Descargar resultado harmonizado si está disponible
                if len(download_cols) == 4 and harmonized_result is not None:
                    with download_cols[3]:
                        harmonized_pil = Image.fromarray(harmonized_result, 'RGB')
                        buf_harmonized = io.BytesIO()
                        harmonized_pil.save(buf_harmonized, format='PNG')

                        st.download_button(
                            label="📥 Resultado Harmonizado",
                            data=buf_harmonized.getvalue(),
                            file_name="resultado_harmonizado.png",
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

                advice_col1, advice_col2, advice_col3 = st.columns(3)

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

                with advice_col3:
                    st.subheader("🎨 Composición Avanzada")
                    composition_tips = [
                        "📏 Escala <1.0 para sujetos lejanos, >1.0 para cercanos",
                        "🌫️ Activa blur de fondo para profundidad realista",
                        "📍 Ajusta posición vertical para mejor composición",
                        "🎯 Distancia >1.5 reduce contraste automáticamente",
                        "🎨 Combina efectos gradualmente para naturalidad"
                    ]
                    for tip in composition_tips:
                        st.write(tip)
                        
                # Nueva sección de consejos específicos
                st.subheader("💡 Consejos Específicos por Efecto")
                
                effect_col1, effect_col2, effect_col3 = st.columns(3)
                
                with effect_col1:
                    st.write("**🎛️ Ajustes de Color:**")
                    color_tips = [
                        "🔆 Brillo: ±20 para ajustes sutiles",
                        "🎭 Contraste: 0.8-1.2 para looks naturales", 
                        "🌈 Saturación: 0.8-1.3 según el estilo deseado",
                        "👻 Opacidad: <1.0 para efectos fantasmales"
                    ]
                    for tip in color_tips:
                        st.write(tip)
                
                with effect_col2:
                    st.write("**🎨 Harmonización:**")
                    harmony_tips = [
                        "🔧 Factor 0.7 = balance natural",
                        "✨ Siempre preservar nitidez activo",
                        "🖼️ Fondos similares en iluminación",
                        "⚡ Desactivar si empeora resultado"
                    ]
                    for tip in harmony_tips:
                        st.write(tip)
                
                with effect_col3:
                    st.write("**📐 Perspectiva:**")
                    perspective_tips = [
                        "🏠 Interiores: distancia 0.8-1.2",
                        "🌅 Paisajes: distancia 1.5-2.5",
                        "👤 Retratos: escala 0.9-1.1",
                        "🎬 Cinematográfico: blur fondo 8-12"
                    ]
                    for tip in perspective_tips:
                        st.write(tip)

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
        ### 🏗️ Arquitectura de los Modelos
        
        **Modelo de Segmentación:**
        - **Base**: U-Net con encoder ResNet-34 pre-entrenado
        - **Attention Gates**: Mecanismo de atención para mejorar precisión
        - **Entrada**: 3 canales RGB
        - **Salida**: 4 canales RGBA (RGB + canal Alpha para transparencia)
        
        **Modelo de Harmonización:**
        - **Base**: U-Net especializado en corrección de color
        - **Entrada**: 3 canales RGB (imagen compuesta)
        - **Salida**: 3 canales RGB (imagen harmonizada)
        - **Pérdidas**: MSE, Perceptual, Color Consistency, Style

        ### 🔄 Proceso de Entrenamiento
        - **Dataset Segmentación**: Supervisely Persons
        - **Dataset Harmonización**: Composiciones sintéticas
        - **Aumentación**: Flip horizontal, rotaciones, cambios de brillo/contraste
        - **Optimización**: Adam con Cosine Annealing scheduler

        ### ⚙️ Detalles de Implementación
        - **Pipeline completo**: Segmentación → Composición → Harmonización
        - **Redimensionamiento**: Mantiene proporciones con padding
        - **Restauración**: Exacta al tamaño original
        - **Composición**: Alpha blending avanzado
        - **Procesamiento**: Optimizado para CPU y GPU
        - **Precisión**: Float32 para mayor calidad
        """)

    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🎭 Procesador de Imágenes con IA - Segmentación, Composición y Harmonización<br>
            Desarrollado con ❤️ usando Streamlit, PyTorch y OpenCV<br>
            Pipeline completo: U-Net Segmentación + U-Net Harmonización + Alpha Blending
        </div>
        """,
        unsafe_allow_html=True
    )


def calculate_sharpness(image):
    """
    Calcula la nitidez de una imagen usando la varianza del Laplaciano.
    
    Args:
        image: Imagen RGB como numpy array
        
    Returns:
        float: Valor de nitidez (mayor = más nítida)
    """
    if len(image.shape) == 3:
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calcular Laplaciano
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Retornar varianza del Laplaciano
    return laplacian.var()


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