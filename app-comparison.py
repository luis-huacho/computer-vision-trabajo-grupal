import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50
import numpy as np
import cv2
import os
from PIL import Image
import io
import warnings
import sys
from smartComposition import smart_composite_arrays

warnings.filterwarnings('ignore')

# Añadir el directorio raíz del proyecto al sys.path para encontrar el módulo 'models'
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Importar la arquitectura correcta de ResNet50
from models import UNetAutoencoder as UNetAutoencoder50_correct
from utils import ImageProcessor as ImageProcessorUtils

# Configurar la página
st.set_page_config(
    page_title="Comparación ResNet34 vs ResNet50 - Segmentación y Composición",
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


class UNetEncoder34(nn.Module):
    """Encoder path del U-Net con ResNet-34."""

    def __init__(self, pretrained=True):
        super(UNetEncoder34, self).__init__()

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


class UNetEncoder50(nn.Module):
    """Encoder path del U-Net con ResNet-50."""

    def __init__(self, pretrained=True):
        super(UNetEncoder50, self).__init__()

        resnet = resnet50(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.bottleneck = DoubleConv(2048, 2048, dropout_rate=0.2)

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


class UNetDecoder34(nn.Module):
    """Decoder path del U-Net para ResNet-34."""

    def __init__(self, use_attention=True):
        super(UNetDecoder34, self).__init__()
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


class UNetDecoder50(nn.Module):
    """Decoder path del U-Net para ResNet-50."""

    def __init__(self, use_attention=True):
        super(UNetDecoder50, self).__init__()
        self.use_attention = use_attention

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        if self.use_attention:
            self.att1 = AttentionBlock(1024, 2048, 512)
            self.att2 = AttentionBlock(512, 1024, 256)
            self.att3 = AttentionBlock(256, 512, 128)
            self.att4 = AttentionBlock(128, 256, 64)
            self.att5 = AttentionBlock(64, 64, 32)

        self.conv1 = DoubleConv(3072, 1024)  # 2048 + 1024
        self.conv2 = DoubleConv(1536, 512)   # 1024 + 512
        self.conv3 = DoubleConv(768, 256)    # 512 + 256
        self.conv4 = DoubleConv(384, 128)    # 256 + 128
        self.conv5 = DoubleConv(192, 64)     # 128 + 64

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


class UNetAutoencoder34(nn.Module):
    """U-Net Autoencoder completo para remoción de fondo con ResNet-34."""

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder34, self).__init__()
        self.encoder = UNetEncoder34(pretrained=pretrained)
        self.decoder = UNetDecoder34(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded


class UNetAutoencoder50(nn.Module):
    """U-Net Autoencoder completo para remoción de fondo con ResNet-50."""

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder50, self).__init__()
        self.encoder = UNetEncoder50(pretrained=pretrained)
        self.decoder = UNetDecoder50(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded


# ============================================================================
# CLASE DE INFERENCIA CON DEBUG - VERSIÓN DUAL
# ============================================================================

class BackgroundRemoverDual:
    """
    Clase para realizar inferencia con ambos modelos (ResNet34 y ResNet50).
    Permite comparar resultados y usar ResNet50 como modelo principal.
    """

    def __init__(self, model_path_34, model_path_50, device='cpu'):
        self.device = device
        self.processor = ImageProcessor()
        
        # Cargar ResNet34
        self.model_34 = UNetAutoencoder34(pretrained=False, use_attention=True)
        self.model_34_loaded = self._load_model(self.model_34, model_path_34, "ResNet34")
        
        # Cargar ResNet50 usando la arquitectura correcta del módulo models.py
        self.model_50 = UNetAutoencoder50_correct(pretrained=False, use_attention=True)
        self.model_50_loaded = self._load_model(self.model_50, model_path_50, "ResNet50")

    def _load_model(self, model, model_path, model_name):
        """Cargar un modelo específico."""
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                return True
            else:
                st.warning(f"❌ Modelo {model_name} no encontrado en '{model_path}'")
                return False
        except Exception as e:
            st.error(f"Error al cargar el modelo {model_name}: {str(e)}")
            return False

    def remove_background_comparison(self, image, image_size=256):
        """
        Remueve el fondo usando ambos modelos y retorna comparación.
        
        Returns:
            dict con resultados de ambos modelos
        """
        results = {
            'resnet34': None,
            'resnet50': None,
            'comparison_available': False
        }
        
        try:
            # Convertir PIL a numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Asegurar formato RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = image[:, :, :3]
            else:
                st.error("Formato de imagen no soportado")
                return results

            original_size = image_rgb.shape[:2]

            # Preprocesamiento común
            dummy_mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
            image_processed, _, restore_metadata = self.processor.resize_with_padding(
                image_rgb, dummy_mask, image_size
            )

            # Normalizar
            image_normalized = image_processed.astype(np.float32) / 255.0
            input_tensor = torch.FloatTensor(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            # Optimización para CPU
            torch.set_num_threads(2)

            # Procesamiento con ResNet34
            if self.model_34_loaded:
                results['resnet34'] = self._process_single_model(
                    self.model_34, input_tensor, image_rgb, restore_metadata, image_size, original_size
                )

            # Procesamiento con ResNet50
            if self.model_50_loaded:
                results['resnet50'] = self._process_single_model(
                    self.model_50, input_tensor, image_rgb, restore_metadata, image_size, original_size
                )

            results['comparison_available'] = self.model_34_loaded and self.model_50_loaded
            results['original_image'] = image_rgb
            
            return results

        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")
            return results

    def _process_single_model(self, model, input_tensor, image_rgb, restore_metadata, image_size, original_size):
        """Procesar imagen con un modelo específico."""
        
        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze(0).cpu().numpy()

        # Post-procesamiento
        rgb_channels = output[:3].transpose(1, 2, 0)
        alpha_channel = output[3]

        # Crear imagen RGBA procesada
        result_processed = np.zeros((image_size, image_size, 4), dtype=np.float32)
        result_processed[:, :, :3] = rgb_channels
        result_processed[:, :, 3] = alpha_channel
        result_processed = (result_processed * 255).astype(np.uint8)

        # Restaurar al tamaño original
        rgb_restored, alpha_restored = self.processor.restore_original_size(
            rgb_channels, alpha_channel, restore_metadata
        )

        # Crear imagen RGBA final
        result_final = np.zeros((original_size[0], original_size[1], 4), dtype=np.float32)
        threshold = 0.4
        person_mask = alpha_restored > threshold

        # Usar imagen original donde hay persona detectada
        original_normalized = image_rgb.astype(np.float32) / 255.0
        rgb_enhanced = np.where(
            person_mask[..., np.newaxis],
            original_normalized,
            rgb_restored
        )

        # Suavizar máscara para transiciones naturales
        alpha_smooth = cv2.bilateralFilter(
            (alpha_restored * 255).astype(np.uint8),
            5, 50, 50
        ).astype(np.float32) / 255.0

        result_final[:, :, :3] = rgb_enhanced
        result_final[:, :, 3] = alpha_smooth
        result_final = (result_final * 255).astype(np.uint8)

        return {
            'resized_image': (image_size, image_size),
            'generated_mask': (alpha_channel * 255).astype(np.uint8),
            'result_processed': result_processed,
            'result_final': result_final,
            'metadata': restore_metadata,
            'original_size': original_size,
            'processed_size': (image_size, image_size)
        }

    def get_main_model_result(self, comparison_results):
        """Retorna el resultado del modelo principal (SIEMPRE ResNet50)."""
        # FORZAR el uso de ResNet50 en el pipeline principal
        if comparison_results['resnet50'] is not None:
            return comparison_results['resnet50'], "ResNet50"
        else:
            st.error("❌ ResNet50 no disponible. El pipeline requiere ResNet50 específicamente.")
            return None, None
    
    def get_resnet50_result(self, comparison_results):
        """Retorna ÚNICAMENTE el resultado de ResNet50."""
        return comparison_results['resnet50'] if comparison_results['resnet50'] is not None else None


@st.cache_resource
def load_dual_segmentation_model():
    """Cargar ambos modelos de segmentación."""
    model_path_34 = 'checkpoints/resnet34/best_segmentation.pth'
    model_path_50 = 'checkpoints/resnet50/best_segmentation.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return BackgroundRemoverDual(model_path_34, model_path_50, device)


@st.cache_resource
def load_harmonization_model():
    """Cargar el modelo de harmonización."""
    model_path = 'checkpoints/best_harmonizer.pth'

    if not os.path.exists(model_path):
        st.error("❌ Modelo de harmonización no encontrado en 'checkpoints/best_harmonizer.pth'")
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
    """Muestra múltiples imágenes en una grilla organizada."""
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
    st.title("🔍 Comparación ResNet34 vs ResNet50 - Segmentación y Composición")
    st.markdown("**Pipeline profesional con comparación: ResNet34 vs ResNet50 → Composición 3D → Harmonización**")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Sistema")
        st.markdown("""
        ### 🔬 Comparación de Modelos:
        - **ResNet34**: Más rápido, menos parámetros
        - **ResNet50**: Más preciso, mayor capacidad
        - **Pipeline Principal**: Usa ResNet50 para mejores resultados
        
        ### 📊 Análisis Visual:
        1. **Comparación lado a lado** de ambos modelos
        2. **Métricas de calidad** automáticas
        3. **Pipeline completo** con el modelo seleccionado
        
        ### 🎭 Pipeline Completo:
        1. **Segmentación** - ResNet50 (principal)
        2. **Composición 3D** - Perspectiva y efectos
        3. **Harmonización** - Ajuste inteligente
        4. **Post-procesamiento** - Efectos finales
        """)

        # Configuración
        st.header("⚙️ Configuración")
        processing_size = st.slider("Tamaño de procesamiento", 128, 512, 256, 32)
        show_comparison = st.checkbox("Mostrar comparación ResNet34 vs ResNet50", value=True)
        show_technical_info = st.checkbox("Mostrar información técnica", value=True)
        
        # Configuración de Composición
        with st.expander("🎭 Composición Avanzada"):
            st.write("**Perspectiva y Profundidad:**")
            depth_distance = st.slider("Distancia percibida", 0.1, 3.0, 1.0, 0.1)
            scale_factor = st.slider("Escala del sujeto", 0.3, 1.5, 1.0, 0.05)
            vertical_position = st.slider("Posición vertical", -0.3, 0.3, 0.0, 0.05)
            blur_background = st.checkbox("Desenfocar fondo (bokeh)", value=False)
            blur_strength = st.slider("Intensidad del bokeh", 1, 15, 5, 1) if blur_background else 5
        
        # Configuración de Harmonización
        with st.expander("🎨 Harmonización"):
            harmonization_enabled = st.checkbox("Aplicar harmonización", value=True)
            blend_factor = st.slider("Intensidad de harmonización", 0.0, 1.0, 0.7, 0.1)
            preserve_sharpness = st.checkbox("Preservar nitidez", value=True)
        
        # Ajustes de color
        with st.expander("🎛️ Ajustes Finales"):
            col1_adj, col2_adj = st.columns(2)
            with col1_adj:
                brightness_adjust = st.slider("Brillo", -50, 50, 0, 5)
                contrast_adjust = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
            with col2_adj:
                saturation_adjust = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1)
                final_opacity = st.slider("Opacidad final", 0.1, 1.0, 1.0, 0.05)

        # Información del sistema
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Usando: {device}")

    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        col1, col2 = st.columns(2)
        with col1:
            bg_remover = load_dual_segmentation_model()
        with col2:
            harmonizer = load_harmonization_model()

    # Mostrar estado de los modelos
    models_status = []
    if bg_remover.model_34_loaded:
        models_status.append("✅ ResNet34 cargado correctamente")
    else:
        models_status.append("❌ ResNet34 no disponible")
    
    if bg_remover.model_50_loaded:
        models_status.append("✅ ResNet50 cargado correctamente")
    else:
        models_status.append("❌ ResNet50 no disponible")
    
    if harmonizer is not None:
        models_status.append("✅ Modelo de harmonización cargado")
    else:
        models_status.append("⚠️ Modelo de harmonización no disponible")
    
    for status in models_status:
        if "✅" in status:
            st.success(status)
        elif "❌" in status:
            st.error(status)
        else:
            st.warning(status)

    # Verificar que ResNet50 esté cargado (REQUERIDO para el pipeline)
    if not bg_remover.model_50_loaded:
        st.error("❌ ResNet50 no disponible. Este modelo es REQUERIDO para el pipeline principal.")
        if not bg_remover.model_34_loaded:
            st.error("❌ Ningún modelo disponible. La aplicación no puede continuar.")
            st.stop()
        else:
            st.warning("⚠️ Solo ResNet34 disponible. La comparación será limitada y el pipeline no funcionará.")

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
        # Cargar imágenes
        original_image = Image.open(uploaded_file)
        original_array = np.array(original_image)
        
        background_image = None
        background_array = None
        if background_file is not None:
            background_image = Image.open(background_file)
            background_array = np.array(background_image)

        st.header("🔄 Procesamiento con Comparación de Modelos")

        # Botón de procesamiento
        if st.button("🚀 Procesar y Comparar Modelos", type="primary", use_container_width=True):

            # Crear progreso
            progress_col, info_col = st.columns([3, 1])
            with progress_col:
                progress_bar = st.progress(0)
                status_text = st.empty()
            with info_col:
                timer_text = st.empty()

            import time
            start_time = time.time()

            # Procesamiento con comparación
            status_text.text("🔄 Preparando imágenes...")
            progress_bar.progress(10)
            timer_text.text(f"⏱️ {time.time() - start_time:.1f}s")

            status_text.text("🧠 Procesando con ambos modelos...")
            progress_bar.progress(30)

            # Procesar con ambos modelos
            comparison_results = bg_remover.remove_background_comparison(original_image, image_size=processing_size)
            
            progress_bar.progress(60)
            
            if comparison_results['resnet34'] is not None or comparison_results['resnet50'] is not None:
                
                # MOSTRAR SIEMPRE la comparación de segmentación
                st.header("🆚 VERSUS: Comparación de Segmentación ResNet34 vs ResNet50")
                
                if comparison_results['comparison_available']:
                    # Sección de comparación principal
                    st.subheader("🔍 Resultados de Segmentación Lado a Lado")
                    
                    comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 1])
                    
                    with comp_col1:
                        st.write("**📷 Imagen Original**")
                        st.image(comparison_results['original_image'], 
                               caption="Imagen de entrada", use_container_width=True)
                    
                    with comp_col2:
                        st.write("**🏃 ResNet34 - Rápido**")
                        if comparison_results['resnet34'] is not None:
                            st.image(comparison_results['resnet34']['result_final'], 
                                   caption="Segmentación ResNet34", use_container_width=True)
                            
                            # Métricas ResNet34
                            mask_34 = comparison_results['resnet34']['generated_mask']
                            coverage_34 = np.sum(mask_34 > 127) / mask_34.size * 100
                            sharpness_34 = calculate_sharpness(comparison_results['resnet34']['result_final'][:,:,:3])
                            
                            st.metric("Cobertura", f"{coverage_34:.1f}%")
                            st.metric("Nitidez", f"{sharpness_34:.1f}")
                        else:
                            st.error("ResNet34 no disponible")
                    
                    with comp_col3:
                        st.write("**🎯 ResNet50 - Preciso**")
                        if comparison_results['resnet50'] is not None:
                            st.image(comparison_results['resnet50']['result_final'], 
                                   caption="Segmentación ResNet50", use_container_width=True)
                            
                            # Métricas ResNet50
                            mask_50 = comparison_results['resnet50']['generated_mask']
                            coverage_50 = np.sum(mask_50 > 127) / mask_50.size * 100
                            sharpness_50 = calculate_sharpness(comparison_results['resnet50']['result_final'][:,:,:3])
                            
                            st.metric("Cobertura", f"{coverage_50:.1f}%")
                            st.metric("Nitidez", f"{sharpness_50:.1f}")
                        else:
                            st.error("ResNet50 no disponible")
                    
                    # Análisis comparativo detallado
                    if comparison_results['resnet34'] is not None and comparison_results['resnet50'] is not None:
                        st.subheader("📊 Análisis Comparativo Detallado")
                        
                        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                        
                        with analysis_col1:
                            st.write("**🎯 Calidad de Segmentación**")
                            # Diferencia de cobertura
                            coverage_diff = coverage_50 - coverage_34
                            st.metric("Diferencia Cobertura", f"{coverage_diff:+.1f}%", 
                                    help="Positivo = ResNet50 detecta más área")
                            
                            # Diferencia de nitidez
                            sharpness_diff = sharpness_50 - sharpness_34
                            st.metric("Diferencia Nitidez", f"{sharpness_diff:+.1f}", 
                                    help="Positivo = ResNet50 más nítido")
                        
                        with analysis_col2:
                            st.write("**⚡ Rendimiento**")
                            st.write("• **ResNet34**: ~21M parámetros")
                            st.write("• **ResNet50**: ~25M parámetros")
                            st.write("• **Velocidad**: ResNet34 ~20% más rápido")
                            st.write("• **Precisión**: ResNet50 mayor capacidad")
                        
                        with analysis_col3:
                            st.write("**🏆 Recomendación**")
                            if coverage_50 > coverage_34 and sharpness_50 > sharpness_34:
                                st.success("🎯 **ResNet50 GANA** - Mejor en ambas métricas")
                            elif coverage_50 > coverage_34:
                                st.info("🔍 **ResNet50** mejor cobertura")
                            elif sharpness_50 > sharpness_34:
                                st.info("✨ **ResNet50** mayor nitidez")
                            else:
                                st.warning("⚖️ **Resultados mixtos** - Depende del caso")
                            
                            st.write("**Pipeline usa**: ResNet50")
                    
                    # Comparación de máscaras
                    st.subheader("🎭 Comparación de Máscaras Generadas")
                    
                    mask_col1, mask_col2 = st.columns(2)
                    
                    with mask_col1:
                        if comparison_results['resnet34'] is not None:
                            st.image(comparison_results['resnet34']['generated_mask'], 
                                   caption="Máscara ResNet34", use_container_width=True)
                    
                    with mask_col2:
                        if comparison_results['resnet50'] is not None:
                            st.image(comparison_results['resnet50']['generated_mask'], 
                                   caption="Máscara ResNet50", use_container_width=True)
                
                else:
                    st.warning("⚠️ Solo un modelo disponible - No se puede hacer comparación completa")
                    if comparison_results['resnet34'] is not None:
                        st.info("Mostrando solo resultado de ResNet34")
                        st.image(comparison_results['resnet34']['result_final'], 
                               caption="Segmentación ResNet34", use_container_width=True)
                    if comparison_results['resnet50'] is not None:
                        st.info("Mostrando solo resultado de ResNet50")
                        st.image(comparison_results['resnet50']['result_final'], 
                               caption="Segmentación ResNet50", use_container_width=True)
                
                # FORZAR el uso de ResNet50 en el pipeline principal
                main_result, model_name = bg_remover.get_main_model_result(comparison_results)
                
                if main_result is not None and model_name == "ResNet50":
                    st.success("🎯 **PIPELINE PRINCIPAL**: Usando ResNet50 para máxima calidad")
                    progress_bar.progress(70)
                    status_text.text("🎨 Procesando composición...")
                    
                    # Usar el modelo principal para el pipeline completo
                    composition_result = None
                    harmonized_result = None
                    
                    if background_array is not None:
                        # Composición con fondo
                        foreground_rgba = main_result['result_final']
                        
                        # Redimensionar fondo
                        bg_resized = cv2.resize(background_array, (foreground_rgba.shape[1], foreground_rgba.shape[0]))
                        
                        # Composición avanzada
                        processor = ImageProcessor()
                        composition_rgb = processor.advanced_composite(
                            foreground_rgba, bg_resized,
                            depth_distance=depth_distance,
                            scale_factor=scale_factor,
                            vertical_position=vertical_position,
                            blur_background=blur_background,
                            blur_strength=blur_strength
                        )

                        # Composición con IC LIGHT
                        composition_rgb = smart_composite_arrays(
                            foreground_rgb=foreground_rgba,
                            background_rgb=bg_resized, 
                            model_quality="best",
                            shadow_intensity=0.15,
                        )

                        composition_result = composition_rgb
                        
                        progress_bar.progress(85)
                        status_text.text("✨ Aplicando harmonización...")
                        
                        # Harmonización
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
                        
                        # Ajustes de color finales
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
                            if final_opacity == 1.0:
                                harmonized_result = harmonized_result_rgba[:, :, :3]
                            else:
                                harmonized_result = harmonized_result_rgba

                    progress_bar.progress(100)
                    status_text.text("✅ ¡Procesamiento completado!")
                    timer_text.text(f"⏱️ {time.time() - start_time:.1f}s")

                    # Mostrar resultados del pipeline completo
                    st.header("📊 Resultados del Pipeline Completo")
                    
                    # Preparar imágenes para visualización
                    images_for_display = {
                        "1. Imagen Original": comparison_results['original_image'],
                        "2. Modelo Principal": main_result['result_final']
                    }
                    
                    # model_name ya está definido arriba
                    st.info(f"🎯 **Modelo Principal del Pipeline:** {model_name} (FORZADO)")
                    
                    if model_name != "ResNet50":
                        st.error("❌ ADVERTENCIA: El pipeline requiere ResNet50 específicamente")
                    
                    # Añadir resultados de composición
                    if background_array is not None:
                        images_for_display["3. Fondo Original"] = background_array
                        if composition_result is not None:
                            images_for_display["4. Composición"] = composition_result
                        if harmonized_result is not None:
                            images_for_display["5. Resultado Final"] = harmonized_result
                    
                    # Mostrar grilla de resultados
                    stage_names = list(images_for_display.keys())
                    display_image_comparison(images_for_display, stage_names, cols_per_row=3)

                    # Comparación final lado a lado
                    if harmonized_result is not None:
                        st.header("🔍 Comparación Final")
                        
                        final_cols = st.columns(4)
                        
                        with final_cols[0]:
                            st.subheader("📷 Original")
                            st.image(comparison_results['original_image'], use_container_width=True)
                        
                        with final_cols[1]:
                            st.subheader(f"✂️ {model_name}")
                            st.image(main_result['result_final'], use_container_width=True)
                        
                        with final_cols[2]:
                            st.subheader("🎨 Composición")
                            st.image(composition_result, use_container_width=True)
                        
                        with final_cols[3]:
                            st.subheader("✨ Final")
                            st.image(harmonized_result, use_container_width=True)

                    # Información técnica
                    if show_technical_info:
                        st.header("🔬 Análisis Técnico")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("📏 Información General")
                            st.write(f"**Modelo Principal:** {model_name}")
                            st.write(f"**Tamaño Original:** {main_result['original_size']}")
                            st.write(f"**Tamaño Procesado:** {main_result['processed_size']}")
                            st.write(f"**Escala:** {main_result['metadata']['scale']:.3f}")
                        
                        with col2:
                            st.subheader("📊 Métricas de Calidad")
                            mask = main_result['generated_mask']
                            coverage = np.sum(mask > 127) / mask.size * 100
                            st.metric("Cobertura", f"{coverage:.1f}%")
                            
                            if composition_result is not None:
                                comp_sharp = calculate_sharpness(composition_result)
                                st.metric("Nitidez Composición", f"{comp_sharp:.1f}")
                            
                            if harmonized_result is not None:
                                harm_sharp = calculate_sharpness(harmonized_result)
                                st.metric("Nitidez Final", f"{harm_sharp:.1f}")
                        
                        with col3:
                            st.subheader("🎛️ Efectos Aplicados")
                            effects = []
                            if depth_distance != 1.0:
                                effects.append(f"Distancia: {depth_distance:.1f}")
                            if scale_factor != 1.0:
                                effects.append(f"Escala: {scale_factor:.1f}")
                            if blur_background:
                                effects.append(f"Blur: {blur_strength}")
                            if brightness_adjust != 0:
                                effects.append(f"Brillo: {brightness_adjust:+d}")
                            
                            if effects:
                                for effect in effects:
                                    st.write(f"• {effect}")
                            else:
                                st.write("Sin efectos aplicados")

                    # Opciones de descarga
                    st.header("📥 Descargar Resultados")
                    
                    download_cols = st.columns(4)
                    
                    with download_cols[0]:
                        # Resultado principal
                        result_pil = Image.fromarray(main_result['result_final'], 'RGBA')
                        buf_main = io.BytesIO()
                        result_pil.save(buf_main, format='PNG')
                        
                        st.download_button(
                            label=f"📥 Resultado {model_name}",
                            data=buf_main.getvalue(),
                            file_name=f"segmentacion_{model_name.lower()}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with download_cols[1]:
                        # Máscara
                        mask_pil = Image.fromarray(main_result['generated_mask'], 'L')
                        buf_mask = io.BytesIO()
                        mask_pil.save(buf_mask, format='PNG')
                        
                        st.download_button(
                            label="📥 Máscara",
                            data=buf_mask.getvalue(),
                            file_name="mascara.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with download_cols[2]:
                        # Composición
                        if composition_result is not None:
                            comp_pil = Image.fromarray(composition_result, 'RGB')
                            buf_comp = io.BytesIO()
                            comp_pil.save(buf_comp, format='PNG')
                            
                            st.download_button(
                                label="📥 Composición",
                                data=buf_comp.getvalue(),
                                file_name="composicion.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    with download_cols[3]:
                        # Resultado final
                        if harmonized_result is not None:
                            final_pil = Image.fromarray(harmonized_result, 'RGB')
                            buf_final = io.BytesIO()
                            final_pil.save(buf_final, format='PNG')
                            
                            st.download_button(
                                label="📥 Resultado Final",
                                data=buf_final.getvalue(),
                                file_name="resultado_final.png",
                                mime="image/png",
                                use_container_width=True
                            )

                    st.success("✅ ¡Procesamiento completado con éxito!")
                
                else:
                    st.error("❌ No se pudo procesar la imagen con ningún modelo")
            
            else:
                st.error("❌ Error al procesar la imagen")

    # Footer
    st.markdown("---")
    with st.expander("ℹ️ Información Técnica"):
        st.markdown("""
        ### 🔍 Comparación de Arquitecturas
        
        **ResNet34:**
        - 34 capas, ~21M parámetros
        - Procesamiento más rápido
        - Menor uso de memoria
        - Ideal para aplicaciones en tiempo real
        
        **ResNet50:**
        - 50 capas, ~25M parámetros
        - Mayor precisión en segmentación
        - Mejor detección de detalles finos
        - Modelo principal del pipeline
        
        ### 🎯 Pipeline Optimizado
        - **Modelo Principal**: ResNet50 para máxima calidad
        - **Comparación Visual**: Ambos modelos lado a lado
        - **Composición Avanzada**: Efectos 3D y profundidad
        - **Harmonización**: Ajuste inteligente de colores
        
        ### 🚀 Rendimiento
        - Optimizado para CPU y GPU
        - Procesamiento en paralelo
        - Cache de modelos para velocidad
        - Restoration exacta de dimensiones originales
        """)

    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🔍 Comparación ResNet34 vs ResNet50 - Segmentación y Composición<br>
            Pipeline completo con el mejor modelo disponible<br>
            Desarrollado con ❤️ usando Streamlit, PyTorch y OpenCV
        </div>
        """,
        unsafe_allow_html=True
    )


def calculate_sharpness(image):
    """Calcula la nitidez de una imagen usando la varianza del Laplaciano."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


if __name__ == "__main__":
    main()