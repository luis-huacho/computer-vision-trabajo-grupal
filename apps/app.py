import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50, resnet34
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import sys
import os
import time
import cv2
import warnings
from typing import Tuple, Optional

# Suprimir warnings
warnings.filterwarnings('ignore')

# Intentar importar MiDaS
try:
    import timm
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False

# Añadir el directorio raíz del proyecto al sys.path para encontrar el módulo 'models'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models import UNetAutoencoder as UNetAutoencoder_ResNet50
from utils import ImageProcessor

st.set_page_config(page_title="Segmentación y Composición Realista con IA", layout="wide")

st.title("🤖 Segmentación y Composición Realista con IA")
st.write(
    "Esta aplicación combina **segmentación de personas** con **composición realista** usando estimación de profundidad. "
    "Sube una imagen principal y opcionalmente un fondo para crear composiciones con escala automática basada en profundidad."
)

st.info(
    "🎯 **Nuevo**: Redimensionamiento inteligente que analiza la profundidad del fondo para "
    "escalar automáticamente la persona de manera realista. Soporta método heurístico (rápido) y MiDaS (preciso)."
)

# --- Definición del modelo ResNet34 ---

class AttentionBlock(nn.Module):
    """
    Attention Gate para U-Net.
    Permite al modelo enfocarse en regiones importantes (personas).
    """

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

        # Asegurar mismas dimensiones usando interpolación
        if gate_conv.shape[2:] != input_conv.shape[2:]:
            gate_conv = F.interpolate(gate_conv, size=input_conv.shape[2:], mode='bilinear', align_corners=False)

        combined = self.relu(gate_conv + input_conv)
        attention = self.sigmoid(self.bn(self.output_conv(combined)))

        return x * attention


class DoubleConv(nn.Module):
    """
    Bloque de doble convolución usado en U-Net.
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    """

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


class UNetEncoder_ResNet34(nn.Module):
    """
    Encoder path del U-Net con skip connections.
    Utiliza ResNet-34 pre-entrenado como backbone para extracción de características.
    """

    def __init__(self, pretrained=True):
        super(UNetEncoder_ResNet34, self).__init__()

        # Usar ResNet-34 pre-entrenado como backbone
        resnet = resnet34(pretrained=pretrained)

        # Extraer capas del ResNet
        self.conv1 = resnet.conv1  # 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels (ResNet-34 BasicBlock)
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Capas adicionales para el bottleneck - actualizado para ResNet-34
        self.bottleneck = DoubleConv(512, 1024, dropout_rate=0.2)

    def forward(self, x):
        # Encoder path con skip connections
        skip_connections = []

        # Initial convolution
        x1 = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x1)  # Skip 1: 64 channels

        x2 = self.maxpool(x1)

        # ResNet layers (ResNet-34 con BasicBlock)
        x3 = self.layer1(x2)
        skip_connections.append(x3)  # Skip 2: 64 channels

        x4 = self.layer2(x3)
        skip_connections.append(x4)  # Skip 3: 128 channels

        x5 = self.layer3(x4)
        skip_connections.append(x5)  # Skip 4: 256 channels

        x6 = self.layer4(x5)
        skip_connections.append(x6)  # Skip 5: 512 channels

        # Bottleneck
        x7 = self.bottleneck(x6)

        return x7, skip_connections


class UNetDecoder_ResNet34(nn.Module):
    """
    Decoder path del U-Net con Attention Gates.
    Reconstruye la imagen enfocándose en las personas.
    """

    def __init__(self, use_attention=True):
        super(UNetDecoder_ResNet34, self).__init__()
        self.use_attention = use_attention

        # Upsampling layers - actualizado para ResNet-34
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Attention gates - actualizado para ResNet-34
        if self.use_attention:
            self.att1 = AttentionBlock(512, 512, 256)   # layer4: 512 channels
            self.att2 = AttentionBlock(256, 256, 128)   # layer3: 256 channels
            self.att3 = AttentionBlock(128, 128, 64)    # layer2: 128 channels
            self.att4 = AttentionBlock(64, 64, 32)      # layer1: 64 channels
            self.att5 = AttentionBlock(64, 64, 32)      # conv1: 64 channels

        # Convolution blocks - actualizado para ResNet-34
        self.conv1 = DoubleConv(1024, 512)  # 512 (up1) + 512 (skip[0]) = 1024
        self.conv2 = DoubleConv(512, 256)   # 256 (up2) + 256 (skip[1]) = 512
        self.conv3 = DoubleConv(256, 128)   # 128 (up3) + 128 (skip[2]) = 256
        self.conv4 = DoubleConv(128, 64)    # 64 (up4) + 64 (skip[3]) = 128
        self.conv5 = DoubleConv(128, 64)    # 64 (up5) + 64 (skip[4]) = 128

        # Output layer - 4 channels (RGB + Alpha)
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)

    def _match_tensor_size(self, x, target_tensor):
        """Ajusta el tamaño de x para que coincida con target_tensor usando interpolación."""
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, skip_connections):
        # Decoder path - las skip connections están en orden inverso
        skips = skip_connections[::-1]  # [512, 256, 128, 64, 64] (ResNet-34)

        # Up 1: 1024 -> 512
        x = self.up1(x)  # Upsample
        skip = skips[0]  # 512 channels (layer4)

        # Asegurar que las dimensiones coincidan
        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att1(skip, x)

        x = torch.cat([x, skip], dim=1)  # 512 + 512 = 1024
        x = self.conv1(x)  # 1024 -> 512

        # Up 2: 512 -> 256
        x = self.up2(x)
        skip = skips[1]  # 256 channels (layer3)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att2(skip, x)

        x = torch.cat([x, skip], dim=1)  # 256 + 256 = 512
        x = self.conv2(x)  # 512 -> 256

        # Up 3: 256 -> 128
        x = self.up3(x)
        skip = skips[2]  # 128 channels (layer2)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att3(skip, x)

        x = torch.cat([x, skip], dim=1)  # 128 + 128 = 256
        x = self.conv3(x)  # 256 -> 128

        # Up 4: 128 -> 64
        x = self.up4(x)
        skip = skips[3]  # 64 channels (layer1)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att4(skip, x)

        x = torch.cat([x, skip], dim=1)  # 64 + 64 = 128
        x = self.conv4(x)  # 128 -> 64

        # Up 5: 64 -> 64 (final upsampling)
        x = self.up5(x)
        skip = skips[4]  # 64 channels (conv1)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att5(skip, x)

        x = torch.cat([x, skip], dim=1)  # 64 + 64 = 128
        x = self.conv5(x)  # 128 -> 64

        # Final output
        x = self.final_conv(x)  # 64 -> 4 (RGBA)

        # Aplicar activaciones
        rgb = torch.sigmoid(x[:, :3])  # RGB channels
        alpha = torch.sigmoid(x[:, 3:4])  # Alpha channel

        return torch.cat([rgb, alpha], dim=1)


class UNetAutoencoder_ResNet34(nn.Module):
    """
    U-Net Autoencoder completo para remoción de fondo.
    Combina encoder y decoder para generar imágenes con personas sin fondo.
    """

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder_ResNet34, self).__init__()
        self.encoder = UNetEncoder_ResNet34(pretrained=pretrained)
        self.decoder = UNetDecoder_ResNet34(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded


# --- Sistema de Estimación de Profundidad ---

class DepthEstimator:
    """
    Sistema híbrido de estimación de profundidad que combina:
    1. Estimación heurística (rápida)
    2. MiDaS (precisa, si está disponible)
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.midas_model = None
        self.midas_transform = None
        
        # Intentar cargar MiDaS
        if MIDAS_AVAILABLE:
            self._load_midas()
    
    def _load_midas(self):
        """Carga el modelo MiDaS"""
        try:
            # Usar modelo pequeño para mejor rendimiento
            self.midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            self.midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform
            
            self.midas_model.to(self.device)
            self.midas_model.eval()
            
            st.success("✅ MiDaS cargado correctamente")
        except Exception as e:
            st.warning(f"⚠️ Error cargando MiDaS: {e}")
            self.midas_model = None
    
    def estimate_depth_heuristic(self, background_image: np.ndarray, person_position: Tuple[int, int]) -> float:
        """
        Estimación heurística de profundidad basada en:
        1. Posición vertical (objetos arriba = más lejos)
        2. Análisis de gradientes
        3. Detección de horizonte
        """
        h, w = background_image.shape[:2]
        pos_x, pos_y = person_position
        
        # 1. Factor de posición vertical (Y)
        vertical_factor = pos_y / h  # 0 = arriba, 1 = abajo
        
        # 2. Detectar línea de horizonte
        horizon_y = self._detect_horizon(background_image)
        
        # 3. Calcular distancia relativa al horizonte
        if horizon_y is not None:
            # Distancia al horizonte normalizada
            distance_to_horizon = abs(pos_y - horizon_y) / h
            # Objetos en el horizonte = más lejos
            horizon_factor = 1.0 - distance_to_horizon
        else:
            # Sin horizonte detectado, usar posición Y
            horizon_factor = 1.0 - vertical_factor
        
        # 4. Análisis de textura local
        texture_factor = self._analyze_local_texture(background_image, pos_x, pos_y)
        
        # 5. Combinar factores
        base_scale = 0.3 + (horizon_factor * 0.7)  # Rango: 0.3 - 1.0
        texture_adjustment = 1.0 + (texture_factor - 0.5) * 0.3  # Ajuste: ±15%
        
        final_scale = base_scale * texture_adjustment
        
        return np.clip(final_scale, 0.2, 1.5)
    
    def _detect_horizon(self, image: np.ndarray) -> Optional[int]:
        """Detecta línea de horizonte usando gradientes horizontales"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detectar bordes horizontales
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calcular gradiente horizontal promedio por fila
            horizontal_gradients = np.mean(np.abs(sobel_x), axis=1)
            
            # Buscar la fila con mayor gradiente horizontal (posible horizonte)
            # Ignorar bordes superiores e inferiores
            h = len(horizontal_gradients)
            search_start = int(h * 0.2)
            search_end = int(h * 0.8)
            
            if search_end > search_start:
                local_max = np.argmax(horizontal_gradients[search_start:search_end])
                horizon_y = search_start + local_max
                
                # Validar que sea un horizonte creíble
                if horizontal_gradients[horizon_y] > np.mean(horizontal_gradients) * 1.5:
                    return horizon_y
            
            return None
        except Exception:
            return None
    
    def _analyze_local_texture(self, image: np.ndarray, x: int, y: int, radius: int = 50) -> float:
        """Analiza textura local para estimar profundidad"""
        try:
            h, w = image.shape[:2]
            
            # Definir región de análisis
            x1 = max(0, x - radius)
            x2 = min(w, x + radius)
            y1 = max(0, y - radius)
            y2 = min(h, y + radius)
            
            # Extraer región
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return 0.5
            
            # Convertir a escala de grises
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Calcular varianza (textura)
            texture_variance = np.var(gray_region)
            
            # Normalizar (más textura = más cerca)
            # Valores típicos: 0-10000, normalizar a 0-1
            normalized_texture = min(texture_variance / 5000.0, 1.0)
            
            return normalized_texture
        except Exception:
            return 0.5
    
    def estimate_depth_midas(self, background_image: np.ndarray, person_position: Tuple[int, int]) -> float:
        """Estimación de profundidad usando MiDaS"""
        if self.midas_model is None:
            return self.estimate_depth_heuristic(background_image, person_position)
        
        try:
            # Preprocesar imagen
            input_tensor = self.midas_transform(background_image).to(self.device)
            
            # Predicción
            with torch.no_grad():
                prediction = self.midas_model(input_tensor)
                
                # Redimensionar a tamaño original
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=background_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convertir a numpy
            depth_map = prediction.cpu().numpy()
            
            # Obtener profundidad en la posición de la persona
            pos_x, pos_y = person_position
            h, w = depth_map.shape
            
            # Asegurar que las coordenadas estén dentro de los límites
            pos_x = max(0, min(w-1, pos_x))
            pos_y = max(0, min(h-1, pos_y))
            
            # Obtener profundidad promedio en una región pequeña
            region_size = 20
            x1 = max(0, pos_x - region_size//2)
            x2 = min(w, pos_x + region_size//2)
            y1 = max(0, pos_y - region_size//2)
            y2 = min(h, pos_y + region_size//2)
            
            local_depth = np.mean(depth_map[y1:y2, x1:x2])
            
            # Normalizar profundidad a escala (0.2 - 1.5)
            # MiDaS devuelve valores inversos (mayor = más cerca)
            max_depth = np.max(depth_map)
            min_depth = np.min(depth_map)
            
            if max_depth > min_depth:
                normalized_depth = (local_depth - min_depth) / (max_depth - min_depth)
                # Invertir: más profundidad = más pequeño
                scale = 1.5 - (normalized_depth * 1.3)  # Rango: 0.2 - 1.5
            else:
                scale = 1.0
            
            return np.clip(scale, 0.2, 1.5)
            
        except Exception as e:
            st.warning(f"Error en MiDaS: {e}")
            return self.estimate_depth_heuristic(background_image, person_position)
    
    def estimate_realistic_scale(self, background_image: np.ndarray, person_position: Tuple[int, int], method: str = "heuristic") -> float:
        """Función principal para estimar escala realista"""
        if method == "midas" and self.midas_model is not None:
            return self.estimate_depth_midas(background_image, person_position)
        else:
            return self.estimate_depth_heuristic(background_image, person_position)


class RealisticCompositor:
    """
    Compositor inteligente que usa estimación de profundidad para
    crear composiciones realistas con mejoras de bordes.
    """
    
    def __init__(self, device='cpu'):
        self.depth_estimator = DepthEstimator(device)
        self.processor = ImageProcessor()
    
    def compose_realistic(self, foreground_rgba: np.ndarray, background_rgb: np.ndarray, 
                         person_position: Tuple[int, int], 
                         depth_method: str = "heuristic",
                         manual_scale: float = 1.0,
                         blur_intensity: float = 0.0,
                         target_coverage: float = 0.15,
                         edge_enhancement: dict = None) -> np.ndarray:
        """
        Crea composición realista con escala inteligente y mejoras de bordes.
        
        Args:
            foreground_rgba: Imagen RGBA de la persona
            background_rgb: Imagen RGB del fondo
            person_position: (x, y) posición donde colocar la persona
            depth_method: "heuristic" o "midas"
            manual_scale: Factor de escala manual (multiplica la automática)
            blur_intensity: Intensidad de blur por profundidad (0-1)
            target_coverage: Cobertura objetivo de la persona (0.05-0.5)
            edge_enhancement: Configuración de mejoras de bordes
        """
        if edge_enhancement is None:
            edge_enhancement = {
                'edge_smoothing': 0,
                'grabcut_refine': False,
                'hair_enhancement': False,
                'color_bleeding': 0,
                'lighting_match': False
            }
        
        # 1. Calcular escala base inteligente
        base_scale = self._calculate_base_scale(
            foreground_rgba.shape[:2], background_rgb.shape[:2], target_coverage
        )
        
        # 2. Estimar escala por profundidad
        depth_scale = self.depth_estimator.estimate_realistic_scale(
            background_rgb, person_position, depth_method
        )
        
        # 3. Ajuste anatómico
        pos_y_pct = person_position[1] / background_rgb.shape[0]
        anatomical_scale = self._anatomical_scale_adjustment(
            foreground_rgba.shape[0], background_rgb.shape[0], pos_y_pct
        )
        
        # 4. Combinar todos los factores de escala
        combined_scale = base_scale * depth_scale * anatomical_scale * manual_scale
        final_scale = self._validate_scale(combined_scale)
        
        # 5. Aplicar mejoras de bordes antes del escalado
        if any(edge_enhancement.values()):
            foreground_enhanced = self._enhance_segmentation_edges(
                foreground_rgba, background_rgb, edge_enhancement
            )
        else:
            foreground_enhanced = foreground_rgba
        
        # 6. Redimensionar persona
        person_scaled = self._scale_person(foreground_enhanced, final_scale)
        
        # 7. Aplicar blur por profundidad
        if blur_intensity > 0:
            person_scaled = self._apply_depth_blur(person_scaled, depth_scale, blur_intensity)
        
        # 8. Componer en posición
        result = self._compose_at_position(person_scaled, background_rgb, person_position)
        
        return result, {
            'base_scale': base_scale,
            'depth_scale': depth_scale,
            'anatomical_scale': anatomical_scale,
            'final_scale': final_scale
        }
    
    def _scale_person(self, foreground_rgba: np.ndarray, scale: float) -> np.ndarray:
        """Redimensiona persona manteniendo calidad"""
        h, w = foreground_rgba.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h <= 0 or new_w <= 0:
            return foreground_rgba
        
        # Usar interpolación cúbica para mejor calidad
        scaled = cv2.resize(foreground_rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return scaled
    
    def _apply_depth_blur(self, person_rgba: np.ndarray, depth_scale: float, blur_intensity: float) -> np.ndarray:
        """Aplica blur basado en profundidad"""
        # Calcular intensidad de blur
        # Objetos lejanos (escala pequeña) = más blur
        blur_amount = (1.0 - depth_scale) * blur_intensity * 15
        
        if blur_amount < 1.0:
            return person_rgba
        
        # Aplicar blur gaussiano
        kernel_size = int(blur_amount * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        try:
            blurred = cv2.GaussianBlur(person_rgba, (kernel_size, kernel_size), blur_amount/3)
            return blurred
        except Exception:
            # En caso de error, retornar imagen original
            return person_rgba
    
    def _compose_at_position(self, person_rgba: np.ndarray, background_rgb: np.ndarray, 
                           position: Tuple[int, int]) -> np.ndarray:
        """Compone persona en posición específica del fondo"""
        pos_x, pos_y = position
        person_h, person_w = person_rgba.shape[:2]
        bg_h, bg_w = background_rgb.shape[:2]
        
        # Crear canvas del tamaño del fondo
        result = background_rgb.copy()
        
        # Calcular posición de colocación (centrado en posición)
        start_x = max(0, pos_x - person_w // 2)
        start_y = max(0, pos_y - person_h // 2)
        end_x = min(bg_w, start_x + person_w)
        end_y = min(bg_h, start_y + person_h)
        
        # Calcular recorte de persona si es necesario
        crop_start_x = max(0, -start_x)
        crop_start_y = max(0, -start_y)
        crop_end_x = crop_start_x + (end_x - start_x)
        crop_end_y = crop_start_y + (end_y - start_y)
        
        if crop_end_x > crop_start_x and crop_end_y > crop_start_y:
            # Extraer región de persona
            person_region = person_rgba[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            
            # Aplicar composición alpha
            result[start_y:end_y, start_x:end_x] = self._alpha_blend(
                person_region, result[start_y:end_y, start_x:end_x]
            )
        
        return result
    
    def _alpha_blend(self, foreground_rgba: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
        """Composición alpha blending"""
        if foreground_rgba.shape[:2] != background_rgb.shape[:2]:
            return background_rgb
        
        # Verificar si foreground tiene canal alpha
        if foreground_rgba.shape[2] == 4:
            # RGBA
            fg_rgb = foreground_rgba[:, :, :3].astype(np.float32) / 255.0
            alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0
        elif foreground_rgba.shape[2] == 3:
            # RGB - crear canal alpha basado en si el píxel no es negro
            fg_rgb = foreground_rgba.astype(np.float32) / 255.0
            # Crear alpha basado en si el píxel no es completamente negro
            alpha = np.any(fg_rgb > 0.01, axis=2).astype(np.float32)
        else:
            return background_rgb
        
        bg_rgb = background_rgb.astype(np.float32) / 255.0
        
        # Expandir alpha
        alpha = alpha[:, :, np.newaxis]
        
        # Composición
        result = fg_rgb * alpha + bg_rgb * (1 - alpha)
        
        return (result * 255).astype(np.uint8)
    
    def _calculate_base_scale(self, person_size: Tuple[int, int], 
                             background_size: Tuple[int, int], 
                             target_coverage: float) -> float:
        """
        Calcula escala base para lograr cobertura objetivo.
        
        Args:
            person_size: (alto, ancho) de la persona segmentada
            background_size: (alto, ancho) del fondo
            target_coverage: Cobertura objetivo (0.05-0.5)
        """
        person_h, person_w = person_size
        bg_h, bg_w = background_size
        
        # Área de la persona y del fondo
        person_area = person_h * person_w
        background_area = bg_h * bg_w
        
        # Cobertura actual
        current_coverage = person_area / background_area
        
        # Escala necesaria para alcanzar cobertura objetivo
        scale = np.sqrt(target_coverage / current_coverage)
        
        return scale
    
    def _anatomical_scale_adjustment(self, person_height: int, 
                                   background_height: int, 
                                   position_y_pct: float) -> float:
        """
        Ajuste anatómico basado en posición vertical.
        
        Args:
            person_height: Alto de la persona en píxeles
            background_height: Alto del fondo en píxeles
            position_y_pct: Posición vertical (0=arriba, 1=abajo)
        """
        # Factor de perspectiva: objetos más abajo parecen más grandes
        perspective_factor = 0.8 + (position_y_pct * 0.4)  # Rango: 0.8 - 1.2
        
        # Factor de proporción: personas muy grandes/pequeñas necesitan ajuste
        size_ratio = person_height / background_height
        
        if size_ratio > 0.8:  # Persona muy grande
            size_adjustment = 0.7
        elif size_ratio < 0.2:  # Persona muy pequeña
            size_adjustment = 1.3
        else:  # Tamaño normal
            size_adjustment = 1.0
        
        return perspective_factor * size_adjustment
    
    def _validate_scale(self, scale: float, min_scale: float = 0.1, 
                       max_scale: float = 3.0) -> float:
        """
        Valida y ajusta la escala final.
        
        Args:
            scale: Escala calculada
            min_scale: Escala mínima permitida
            max_scale: Escala máxima permitida
        """
        return np.clip(scale, min_scale, max_scale)
    
    def _enhance_segmentation_edges(self, foreground_rgba: np.ndarray, 
                                  background_rgb: np.ndarray, 
                                  edge_config: dict) -> np.ndarray:
        """
        Aplica mejoras de bordes a la segmentación.
        
        Args:
            foreground_rgba: Imagen RGBA de la persona
            background_rgb: Imagen RGB del fondo
            edge_config: Configuración de mejoras
        """
        enhanced = foreground_rgba.copy()
        
        # 1. Suavizado inteligente de bordes
        if edge_config.get('edge_smoothing', 0) > 0:
            enhanced = self._smooth_edges(enhanced, edge_config['edge_smoothing'])
        
        # 2. Refinamiento con GrabCut
        if edge_config.get('grabcut_refine', False):
            enhanced = self._grabcut_refinement(enhanced, background_rgb)
        
        # 3. Mejora específica de cabello
        if edge_config.get('hair_enhancement', False):
            enhanced = self._enhance_hair_edges(enhanced)
        
        # 4. Sangrado de color
        if edge_config.get('color_bleeding', 0) > 0:
            enhanced = self._apply_color_bleeding(enhanced, background_rgb, 
                                                edge_config['color_bleeding'])
        
        # 5. Ajuste de iluminación
        if edge_config.get('lighting_match', False):
            enhanced = self._match_edge_lighting(enhanced, background_rgb)
        
        return enhanced
    
    def _smooth_edges(self, foreground_rgba: np.ndarray, intensity: float) -> np.ndarray:
        """
        Suaviza los bordes de la segmentación.
        
        Args:
            foreground_rgba: Imagen RGBA
            intensity: Intensidad del suavizado (0-1)
        """
        if intensity <= 0:
            return foreground_rgba
        
        # Extraer canal alpha
        alpha = foreground_rgba[:, :, 3]
        
        # Aplicar filtro bilateral al canal alpha para suavizar bordes
        kernel_size = int(intensity * 15) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        try:
            smoothed_alpha = cv2.bilateralFilter(alpha, kernel_size, 
                                               intensity * 80, intensity * 80)
            
            # Crear resultado
            result = foreground_rgba.copy()
            result[:, :, 3] = smoothed_alpha
            
            return result
        except Exception:
            return foreground_rgba
    
    def _grabcut_refinement(self, foreground_rgba: np.ndarray, 
                           background_rgb: np.ndarray) -> np.ndarray:
        """
        Refina bordes usando GrabCut.
        
        Args:
            foreground_rgba: Imagen RGBA de la persona
            background_rgb: Imagen RGB del fondo
        """
        try:
            # Convertir a RGB para GrabCut
            rgb_image = foreground_rgba[:, :, :3]
            alpha_mask = foreground_rgba[:, :, 3]
            
            # Crear máscara inicial para GrabCut
            mask = np.zeros(alpha_mask.shape, dtype=np.uint8)
            mask[alpha_mask > 200] = cv2.GC_PR_FGD  # Foreground probable
            mask[alpha_mask > 240] = cv2.GC_FGD     # Foreground definitivo
            mask[alpha_mask < 50] = cv2.GC_BGD      # Background definitivo
            
            # Crear rectángulo que contiene el objeto
            coords = np.where(alpha_mask > 100)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Aplicar GrabCut
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(rgb_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                
                # Crear nueva máscara alpha
                new_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                
                # Combinar con máscara original
                combined_mask = np.maximum(alpha_mask * 0.3, new_mask * 0.7).astype(np.uint8)
                
                # Crear resultado
                result = foreground_rgba.copy()
                result[:, :, 3] = combined_mask
                
                return result
                
        except Exception:
            pass
        
        return foreground_rgba
    
    def _enhance_hair_edges(self, foreground_rgba: np.ndarray) -> np.ndarray:
        """
        Mejora específica para bordes de cabello.
        
        Args:
            foreground_rgba: Imagen RGBA de la persona
        """
        try:
            alpha = foreground_rgba[:, :, 3]
            
            # Detectar regiones de cabello (parte superior de la imagen)
            h, w = alpha.shape
            hair_region = alpha[:h//3, :]  # Tercio superior
            
            # Aplicar filtro específico para cabello
            hair_enhanced = cv2.morphologyEx(hair_region, cv2.MORPH_CLOSE, 
                                           np.ones((3, 3), np.uint8))
            
            # Suavizado adicional
            hair_enhanced = cv2.GaussianBlur(hair_enhanced, (3, 3), 0.5)
            
            # Crear resultado
            result = foreground_rgba.copy()
            result[:h//3, :, 3] = hair_enhanced
            
            return result
            
        except Exception:
            return foreground_rgba
    
    def _apply_color_bleeding(self, foreground_rgba: np.ndarray, 
                            background_rgb: np.ndarray, 
                            intensity: float) -> np.ndarray:
        """
        Aplica sangrado de color en los bordes.
        
        Args:
            foreground_rgba: Imagen RGBA de la persona
            background_rgb: Imagen RGB del fondo
            intensity: Intensidad del sangrado (0-1)
        """
        if intensity <= 0:
            return foreground_rgba
        
        try:
            # Redimensionar fondo si es necesario
            if background_rgb.shape[:2] != foreground_rgba.shape[:2]:
                background_rgb = cv2.resize(background_rgb, 
                                          (foreground_rgba.shape[1], foreground_rgba.shape[0]))
            
            # Crear máscara de bordes
            alpha = foreground_rgba[:, :, 3]
            edge_mask = cv2.Canny(alpha, 50, 150)
            
            # Dilatar máscara de bordes
            kernel = np.ones((5, 5), np.uint8)
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
            
            # Aplicar sangrado
            result = foreground_rgba.copy()
            for i in range(3):  # RGB channels
                fg_channel = result[:, :, i]
                bg_channel = background_rgb[:, :, i]
                
                # Mezclar colores en los bordes
                blend_mask = (edge_mask / 255.0) * intensity
                result[:, :, i] = fg_channel * (1 - blend_mask) + bg_channel * blend_mask
            
            return result
            
        except Exception:
            return foreground_rgba
    
    def _match_edge_lighting(self, foreground_rgba: np.ndarray, 
                           background_rgb: np.ndarray) -> np.ndarray:
        """
        Ajusta la iluminación en los bordes.
        
        Args:
            foreground_rgba: Imagen RGBA de la persona
            background_rgb: Imagen RGB del fondo
        """
        try:
            # Redimensionar fondo si es necesario
            if background_rgb.shape[:2] != foreground_rgba.shape[:2]:
                background_rgb = cv2.resize(background_rgb, 
                                          (foreground_rgba.shape[1], foreground_rgba.shape[0]))
            
            # Calcular iluminación promedio del fondo
            bg_brightness = np.mean(background_rgb)
            
            # Crear máscara de bordes
            alpha = foreground_rgba[:, :, 3]
            edge_mask = cv2.Canny(alpha, 50, 150)
            kernel = np.ones((7, 7), np.uint8)
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)
            
            # Ajustar iluminación en bordes
            result = foreground_rgba.copy()
            for i in range(3):  # RGB channels
                fg_channel = result[:, :, i].astype(np.float32)
                fg_brightness = np.mean(fg_channel[alpha > 100])
                
                # Factor de ajuste
                adjust_factor = bg_brightness / (fg_brightness + 1e-6)
                adjust_factor = np.clip(adjust_factor, 0.7, 1.3)
                
                # Aplicar ajuste solo en bordes
                edge_influence = (edge_mask / 255.0) * 0.3
                adjusted_channel = fg_channel * (1 - edge_influence + edge_influence * adjust_factor)
                
                result[:, :, i] = np.clip(adjusted_channel, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception:
            return foreground_rgba

# --- Funciones Principales ---

@st.cache_resource
def load_model_resnet50():
    """Carga el modelo ResNet50"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, "checkpoints", "best_segmentation.pth")
    
    try:
        model = UNetAutoencoder_ResNet50(pretrained=False, use_attention=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando modelo ResNet50: {e}")
        return None, None

@st.cache_resource
def load_model_resnet34():
    """Carga el modelo ResNet34"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, "checkpoints", "resnet34", "best_segmentation.pth")
    
    try:
        model = UNetAutoencoder_ResNet34(pretrained=False, use_attention=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando modelo ResNet34: {e}")
        return None, None

def preprocess_image(image, image_size=384):
    """
    Preprocesa la imagen para que sea compatible con el modelo,
    replicando EXACTAMENTE el preprocesamiento del entrenamiento.
    """
    # 1. Convertir PIL Image a numpy array
    image_np = np.array(image)

    # 2. Usar ImageProcessor para redimensionar con padding
    dummy_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    processor = ImageProcessor()
    image_processed, _, restore_metadata = processor.resize_with_padding(
        image_np, dummy_mask, image_size
    )

    # 3. Normalizar a [0, 1] y convertir a tensor
    image_processed_float = image_processed.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_processed_float.transpose(2, 0, 1)).unsqueeze(0)

    return image_tensor, restore_metadata

def postprocess_and_visualize(original_image, model_output, restore_metadata):
    """
    Procesa la salida del modelo y genera imágenes de cada paso.
    """
    output_cpu = model_output.cpu().detach()[0]
    processor = ImageProcessor()

    # --- 1. Salida RGB cruda del modelo ---
    raw_rgb_pil = transforms.ToPILImage()(output_cpu[:3, :, :])

    # --- 2. Máscara de opacidad (alpha) cruda del modelo ---
    raw_mask_pil = transforms.ToPILImage()(output_cpu[3, :, :])

    # --- 3. Máscara restaurada al tamaño original ---
    raw_mask_np = np.array(raw_mask_pil)
    dummy_processed_image = np.zeros_like(raw_mask_np)
    _, restored_mask_np = processor.restore_original_size(
        dummy_processed_image, raw_mask_np, restore_metadata
    )
    restored_mask_pil = Image.fromarray(restored_mask_np)

    # --- 4. Aplicación de la máscara a la imagen original ---
    mask_np = restored_mask_np / 255.0
    original_np = np.array(original_image)
    
    # Crear imagen RGBA con canal alpha
    segmented_rgba = np.zeros((original_np.shape[0], original_np.shape[1], 4), dtype=np.uint8)
    segmented_rgba[:, :, :3] = original_np  # RGB
    segmented_rgba[:, :, 3] = restored_mask_np  # Alpha
    
    final_result_pil = Image.fromarray(segmented_rgba, 'RGBA')

    return {
        "raw_rgb": raw_rgb_pil,
        "raw_mask": raw_mask_pil,
        "restored_mask": restored_mask_pil,
        "final_result": final_result_pil
    }

def process_with_model(model, device, original_image, model_name):
    """
    Procesa una imagen con el modelo especificado y mide el tiempo.
    """
    start_time = time.time()
    
    # Preprocesar la imagen
    input_tensor, restore_metadata = preprocess_image(original_image)
    input_tensor = input_tensor.to(device)
    
    # Realizar la inferencia
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocesar
    viz_dict = postprocess_and_visualize(original_image, output_tensor, restore_metadata)
    
    processing_time = time.time() - start_time
    
    return viz_dict, processing_time

def process_realistic_composition(segmented_image, background_image, pos_x_pct, pos_y_pct, 
                                depth_method, manual_scale, blur_intensity, 
                                target_coverage=0.15, edge_enhancement=None):
    """
    Procesa composición realista con los parámetros especificados.
    """
    start_time = time.time()
    
    # Convertir PIL a numpy si es necesario
    if isinstance(segmented_image, Image.Image):
        segmented_np = np.array(segmented_image)
    else:
        segmented_np = segmented_image
    
    if isinstance(background_image, Image.Image):
        background_np = np.array(background_image)
    else:
        background_np = background_image
    
    # Convertir porcentajes a coordenadas
    bg_h, bg_w = background_np.shape[:2]
    person_pos_x = int((pos_x_pct / 100.0) * bg_w)
    person_pos_y = int((pos_y_pct / 100.0) * bg_h)
    
    # Realizar composición realista
    composition, scale_info = compositor.compose_realistic(
        segmented_np, background_np, 
        (person_pos_x, person_pos_y),
        depth_method, manual_scale, blur_intensity,
        target_coverage, edge_enhancement
    )
    
    processing_time = time.time() - start_time
    
    return {
        "composition": composition,
        "scale_info": scale_info,
        "processing_time": processing_time,
        "person_position": (person_pos_x, person_pos_y)
    }

# --- Carga de Modelos ---
st.sidebar.header("⚙️ Configuración de Modelos")

# Cargar modelos
with st.sidebar:
    st.write("**Estado de los modelos:**")
    
    model_resnet50, device = load_model_resnet50()
    if model_resnet50:
        st.success("✅ ResNet50 cargado correctamente")
    else:
        st.error("❌ Error cargando ResNet50")
    
    model_resnet34, _ = load_model_resnet34()
    if model_resnet34:
        st.success("✅ ResNet34 cargado correctamente")
    else:
        st.error("❌ Error cargando ResNet34")

# Inicializar compositor realista
@st.cache_resource
def load_realistic_compositor():
    """Carga el compositor realista"""
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return RealisticCompositor(device_name)

compositor = load_realistic_compositor()

# --- Controles de Composición Realista ---
st.sidebar.header("🎯 Composición Realista")

with st.sidebar:
    st.write("**Método de Estimación de Profundidad:**")
    depth_method = st.selectbox(
        "Selecciona método:",
        ["heuristic", "midas"],
        index=0,
        help="Heurística: Rápida, funciona siempre. MiDaS: Más precisa, requiere más recursos."
    )
    
    if depth_method == "midas" and not MIDAS_AVAILABLE:
        st.warning("⚠️ MiDaS no disponible. Instala timm: `pip install timm`")
        depth_method = "heuristic"
    
    st.write("**Controles de Escala:**")
    manual_scale = st.slider(
        "Escala manual", 
        0.1, 2.0, 1.0, 0.1,
        help="Multiplica la escala automática"
    )
    
    st.write("**Posición en el Fondo:**")
    position_x = st.slider(
        "Posición Horizontal (%)", 
        0, 100, 50, 5,
        help="Posición horizontal en el fondo"
    )
    
    position_y = st.slider(
        "Posición Vertical (%)", 
        0, 100, 70, 5,
        help="Posición vertical en el fondo"
    )
    
    st.write("**Cobertura Objetivo:**")
    target_coverage = st.slider(
        "Cobertura de la persona (%)", 
        5, 50, 15, 5,
        help="Porcentaje del fondo que debe cubrir la persona"
    ) / 100.0
    
    st.write("**Efectos de Profundidad:**")
    blur_intensity = st.slider(
        "Intensidad de Blur por Profundidad", 
        0.0, 1.0, 0.0, 0.1,
        help="Objetos lejanos se ven más borrosos"
    )
    
    show_depth_info = st.checkbox(
        "Mostrar información de profundidad", 
        True,
        help="Muestra detalles del análisis de profundidad"
    )

# --- Controles de Mejoras de Bordes ---
st.sidebar.header("🎨 Mejoras de Bordes")

with st.sidebar:
    st.write("**Suavizado de Bordes:**")
    edge_smoothing = st.slider(
        "Intensidad de Suavizado", 
        0.0, 1.0, 0.0, 0.1,
        help="Suaviza los bordes de la segmentación"
    )
    
    st.write("**Refinamiento Avanzado:**")
    grabcut_refine = st.checkbox(
        "Refinamiento GrabCut", 
        False,
        help="Mejora bordes usando algoritmo GrabCut"
    )
    
    hair_enhancement = st.checkbox(
        "Mejora de Cabello", 
        False,
        help="Mejora específica para bordes de cabello"
    )
    
    st.write("**Integración con Fondo:**")
    color_bleeding = st.slider(
        "Sangrado de Color", 
        0.0, 1.0, 0.0, 0.1,
        help="Mezcla colores del fondo en los bordes"
    )
    
    lighting_match = st.checkbox(
        "Ajuste de Iluminación", 
        False,
        help="Ajusta la iluminación en los bordes"
    )
    
    # Crear configuración de edge enhancement
    edge_enhancement = {
        'edge_smoothing': edge_smoothing,
        'grabcut_refine': grabcut_refine,
        'hair_enhancement': hair_enhancement,
        'color_bleeding': color_bleeding,
        'lighting_match': lighting_match
    }

# Selector de modelos
model_options = []
if model_resnet50:
    model_options.append("ResNet50")
if model_resnet34:
    model_options.append("ResNet34")

if not model_options:
    st.error("❌ No se pudo cargar ningún modelo. Verifica que los archivos estén en las rutas correctas.")
    st.stop()

selected_models = st.sidebar.multiselect(
    "Selecciona los modelos a comparar:",
    options=model_options,
    default=model_options,
    help="Puedes seleccionar uno o ambos modelos para comparar"
)

if not selected_models:
    st.warning("⚠️ Por favor selecciona al menos un modelo para continuar.")
    st.stop()

# --- Interfaz de Usuario ---

# --- Subir Archivos ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Imagen Principal")
    uploaded_file = st.file_uploader(
        "Sube una imagen para segmentar:",
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen que contenga personas para comparar el rendimiento de los modelos."
    )

with col2:
    st.subheader("🖼️ Imagen de Fondo (Opcional)")
    background_file = st.file_uploader(
        "Sube una imagen de fondo:",
        type=["jpg", "jpeg", "png"],
        help="Para composición realista. Si no subes una, solo se mostrará la segmentación."
    )

if uploaded_file is not None:
    # Leer la imagen subida
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Leer imagen de fondo si existe
    background_image = None
    if background_file is not None:
        background_bytes = background_file.getvalue()
        background_image = Image.open(io.BytesIO(background_bytes)).convert("RGB")

    # Mostrar imágenes
    display_cols = st.columns(2 if background_image else 1)
    
    with display_cols[0]:
        st.image(original_image, caption="🖼️ Imagen Original", use_container_width=True)
    
    if background_image:
        with display_cols[1]:
            st.image(background_image, caption="🖼️ Imagen de Fondo", use_container_width=True)
    
    st.markdown("---")

    # Botón para iniciar el procesamiento
    button_text = "🚀 Procesar y Componer" if background_image else "🚀 Procesar Segmentación"
    if st.button(button_text, use_container_width=True, type="primary"):
        
        results = {}
        composition_results = {}
        
        # Procesar con cada modelo seleccionado
        for model_name in selected_models:
            with st.spinner(f'🧠 Procesando con {model_name}...'):
                if model_name == "ResNet50":
                    viz_dict, processing_time = process_with_model(model_resnet50, device, original_image, model_name)
                elif model_name == "ResNet34":
                    viz_dict, processing_time = process_with_model(model_resnet34, device, original_image, model_name)
                
                results[model_name] = {
                    "viz_dict": viz_dict,
                    "processing_time": processing_time
                }
                
                # Si hay imagen de fondo, realizar composición realista
                if background_image:
                    with st.spinner(f'🎨 Componiendo con {model_name}...'):
                        composition_result = process_realistic_composition(
                            viz_dict["final_result"], 
                            background_image,
                            position_x, position_y,
                            depth_method, manual_scale, blur_intensity,
                            target_coverage, edge_enhancement
                        )
                        composition_results[model_name] = composition_result
        
        # Mostrar resultados
        st.subheader("📊 Resultados de la Comparación")
        
        # Tabla de métricas
        if len(selected_models) > 1:
            st.write("**⏱️ Tiempo de procesamiento:**")
            metrics_data = []
            for model_name in selected_models:
                row = {
                    "Modelo": model_name,
                    "Segmentación (s)": f"{results[model_name]['processing_time']:.3f}",
                    "Velocidad Relativa": f"{results[selected_models[0]]['processing_time'] / results[model_name]['processing_time']:.2f}x"
                }
                
                # Añadir tiempo de composición si existe
                if model_name in composition_results:
                    row["Composición (s)"] = f"{composition_results[model_name]['processing_time']:.3f}"
                
                metrics_data.append(row)
            
            st.table(metrics_data)
            st.markdown("---")
        
        # Mostrar información de profundidad si está habilitada
        if background_image and show_depth_info:
            st.subheader("🎯 Información de Profundidad")
            
            depth_cols = st.columns(len(selected_models))
            for i, model_name in enumerate(selected_models):
                if model_name in composition_results:
                    comp_result = composition_results[model_name]
                    
                    with depth_cols[i]:
                        st.write(f"**{model_name}:**")
                        scale_info = comp_result['scale_info']
                        st.metric("Escala Profundidad", f"{scale_info['depth_scale']:.2f}")
                        st.metric("Escala Final", f"{scale_info['final_scale']:.2f}")
                        st.write(f"**Posición:** ({comp_result['person_position'][0]}, {comp_result['person_position'][1]})")
                        st.write(f"**Método:** {depth_method}")
                        if show_depth_info:
                            st.write(f"**Base:** {scale_info['base_scale']:.2f}")
                            st.write(f"**Anatómico:** {scale_info['anatomical_scale']:.2f}")
            
            st.markdown("---")
        
        # Mostrar resultados de cada modelo
        if len(selected_models) == 1:
            # Un solo modelo - mostrar en formato completo
            model_name = selected_models[0]
            result = results[model_name]
            
            st.subheader(f"🔍 Resultados de {model_name}")
            st.info(f"Tiempo de segmentación: {result['processing_time']:.3f} segundos")
            
            if background_image and model_name in composition_results:
                # Mostrar segmentación + composición
                comp_result = composition_results[model_name]
                
                # Primero mostrar el proceso de segmentación
                st.write("**🔪 Proceso de Segmentación:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result["viz_dict"]["raw_rgb"], caption="1. Salida RGB del Modelo", use_container_width=True)
                with col2:
                    st.image(result["viz_dict"]["raw_mask"], caption="2. Máscara de Opacidad", use_container_width=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.image(result["viz_dict"]["restored_mask"], caption="3. Máscara Restaurada", use_container_width=True)
                with col4:
                    st.image(result["viz_dict"]["final_result"], caption="4. Persona Segmentada", use_container_width=True)
                
                # Luego mostrar la composición realista
                st.write("**🎨 Composición Realista:**")
                st.info(f"Tiempo de composición: {comp_result['processing_time']:.3f} segundos")
                
                # Información detallada de escalado
                if show_depth_info:
                    scale_info = comp_result['scale_info']
                    st.write("**📏 Información de Escalado:**")
                    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                    with col_info1:
                        st.metric("Escala Base", f"{scale_info['base_scale']:.2f}")
                    with col_info2:
                        st.metric("Profundidad", f"{scale_info['depth_scale']:.2f}")
                    with col_info3:
                        st.metric("Anatómico", f"{scale_info['anatomical_scale']:.2f}")
                    with col_info4:
                        st.metric("Final", f"{scale_info['final_scale']:.2f}")
                
                col5, col6 = st.columns(2)
                with col5:
                    st.image(background_image, caption="5. Fondo Original", use_container_width=True)
                with col6:
                    st.image(comp_result["composition"], caption="6. Composición Final", use_container_width=True)
            else:
                # Solo segmentación
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result["viz_dict"]["raw_rgb"], caption="1. Salida RGB del Modelo", use_container_width=True)
                with col2:
                    st.image(result["viz_dict"]["raw_mask"], caption="2. Máscara de Opacidad", use_container_width=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.image(result["viz_dict"]["restored_mask"], caption="3. Máscara Restaurada", use_container_width=True)
                with col4:
                    st.image(result["viz_dict"]["final_result"], caption="4. Resultado Final", use_container_width=True)
        
        else:
            # Múltiples modelos - mostrar comparación lado a lado
            st.subheader("🔄 Comparación Lado a Lado")
            
            # Crear tabs para cada paso del proceso
            if background_image:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 RGB Crudo", "🎭 Máscara Alpha", "🔧 Máscara Restaurada", "✨ Segmentación", "🎯 Composición"])
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["🎨 RGB Crudo", "🎭 Máscara Alpha", "🔧 Máscara Restaurada", "✨ Resultado Final"])
            
            with tab1:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["raw_rgb"], 
                                caption=f"{model_name} - RGB", 
                                use_container_width=True)
            
            with tab2:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["raw_mask"], 
                                caption=f"{model_name} - Máscara", 
                                use_container_width=True)
            
            with tab3:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["restored_mask"], 
                                caption=f"{model_name} - Restaurada", 
                                use_container_width=True)
            
            with tab4:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["final_result"], 
                                caption=f"{model_name} - Segmentada", 
                                use_container_width=True)
            
            # Tab de composición si hay fondo
            if background_image:
                with tab5:
                    cols = st.columns(len(selected_models))
                    for i, model_name in enumerate(selected_models):
                        with cols[i]:
                            if model_name in composition_results:
                                comp_result = composition_results[model_name]
                                st.image(comp_result["composition"], 
                                        caption=f"{model_name} - Composición", 
                                        use_container_width=True)
                                scale_info = comp_result['scale_info']
                                st.caption(f"Escala Final: {scale_info['final_scale']:.2f}")
                                if show_depth_info:
                                    st.caption(f"Base: {scale_info['base_scale']:.2f} | Profundidad: {scale_info['depth_scale']:.2f} | Anatómico: {scale_info['anatomical_scale']:.2f}")
                            else:
                                st.write("Sin composición")

# --- Información adicional ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 Información de los Modelos")
    st.markdown("""
    **ResNet50:**
    - Más profundo (50 capas)
    - Mayor capacidad de representación
    - Más parámetros
    
    **ResNet34:**
    - Más ligero (34 capas)
    - Procesamiento más rápido
    - Menor uso de memoria
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Estimación de Profundidad")
    st.markdown("""
    **Método Heurístico:**
    - ⚡ Rápido (~50ms)
    - 🔋 Bajo uso de memoria
    - 📐 Basado en posición Y y gradientes
    
    **Método MiDaS:**
    - 🎯 Más preciso
    - ⏱️ Más lento (2-15s)
    - 🧠 Requiere más recursos
    - 🏆 Mejor para composiciones complejas
    """)
    
    st.markdown("---")
    st.markdown("### 🎨 Mejoras de Bordes")
    st.markdown("""
    **Suavizado:** Suaviza bordes abruptos
    **GrabCut:** Refina automáticamente bordes
    **Cabello:** Mejora específica para cabello
    **Sangrado:** Mezcla colores del fondo
    **Iluminación:** Ajusta luz en bordes
    """)
    
    st.markdown("### 💡 Consejos")
    st.markdown("""
    **Para mejores resultados:**
    - Sube imágenes con personas claras
    - Usa fondos con perspectiva definida
    - Ajusta la posición según el contexto
    - Prueba ambos métodos de profundidad
    - Usa mejoras de bordes para mayor realismo
    
    **Escalado Inteligente:**
    - **Base:** Cobertura objetivo (5-50%)
    - **Profundidad:** Análisis automático
    - **Anatómico:** Posición y proporción
    - **Final:** Combinación validada
    """)
    
    st.markdown("---")
    device_info = "🖥️ CPU" if not torch.cuda.is_available() else "🚀 GPU"
    midas_status = "✅ Disponible" if MIDAS_AVAILABLE else "❌ No disponible"
    st.markdown(f"""
    **Sistema:**
    - Dispositivo: {device_info}
    - MiDaS: {midas_status}
    """)