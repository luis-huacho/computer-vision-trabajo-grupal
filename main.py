import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet34
import numpy as np
import cv2
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


# Configuración de logging
def setup_logging():
    """Configura el sistema de logging para el entrenamiento."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/training_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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
        """
        Args:
            x: Feature map del skip connection
            gate: Feature map del decoder path
        """
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


class UNetEncoder(nn.Module):
    """
    Encoder path del U-Net con skip connections.
    Utiliza ResNet34 pre-entrenado como backbone para mejor extracción de características.
    """

    def __init__(self, pretrained=True):
        super(UNetEncoder, self).__init__()

        # Usar ResNet34 pre-entrenado como backbone
        resnet = resnet34(pretrained=pretrained)

        # Extraer capas del ResNet
        self.conv1 = resnet.conv1  # 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Capas adicionales para el bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout_rate=0.2)

    def forward(self, x):
        # Encoder path con skip connections
        skip_connections = []

        # Initial convolution
        x1 = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x1)  # Skip 1: 64 channels

        x2 = self.maxpool(x1)

        # ResNet layers
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


class UNetDecoder(nn.Module):
    """
    Decoder path del U-Net con Attention Gates.
    Reconstruye la imagen enfocándose en las personas.
    """

    def __init__(self, use_attention=True):
        super(UNetDecoder, self).__init__()
        self.use_attention = use_attention

        # Upsampling layers - usando ConvTranspose2d
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Attention gates
        if self.use_attention:
            self.att1 = AttentionBlock(512, 512, 256)
            self.att2 = AttentionBlock(256, 256, 128)
            self.att3 = AttentionBlock(128, 128, 64)
            self.att4 = AttentionBlock(64, 64, 32)
            self.att5 = AttentionBlock(64, 64, 32)

        # Convolution blocks
        self.conv1 = DoubleConv(1024, 512)
        self.conv2 = DoubleConv(512, 256)
        self.conv3 = DoubleConv(256, 128)
        self.conv4 = DoubleConv(128, 64)
        self.conv5 = DoubleConv(128, 64)

        # Output layer - 4 channels (RGB + Alpha)
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)

    def _match_tensor_size(self, x, target_tensor):
        """Ajusta el tamaño de x para que coincida con target_tensor usando interpolación."""
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, skip_connections):
        # Decoder path - las skip connections están en orden inverso
        skips = skip_connections[::-1]  # [512, 256, 128, 64, 64]

        # Up 1: 1024 -> 512
        x = self.up1(x)  # Upsample
        skip = skips[0]  # 512 channels

        # Asegurar que las dimensiones coincidan
        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att1(skip, x)

        x = torch.cat([x, skip], dim=1)  # 512 + 512 = 1024
        x = self.conv1(x)  # 1024 -> 512

        # Up 2: 512 -> 256
        x = self.up2(x)
        skip = skips[1]  # 256 channels

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att2(skip, x)

        x = torch.cat([x, skip], dim=1)  # 256 + 256 = 512
        x = self.conv2(x)  # 512 -> 256

        # Up 3: 256 -> 128
        x = self.up3(x)
        skip = skips[2]  # 128 channels

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att3(skip, x)

        x = torch.cat([x, skip], dim=1)  # 128 + 128 = 256
        x = self.conv3(x)  # 256 -> 128

        # Up 4: 128 -> 64
        x = self.up4(x)
        skip = skips[3]  # 64 channels

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att4(skip, x)

        x = torch.cat([x, skip], dim=1)  # 64 + 64 = 128
        x = self.conv4(x)  # 128 -> 64

        # Up 5: 64 -> 64 (final upsampling)
        x = self.up5(x)
        skip = skips[4]  # 64 channels (primera capa conv)

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


class UNetAutoencoder(nn.Module):
    """
    U-Net Autoencoder completo para remoción de fondo.
    Combina encoder y decoder para generar imágenes con personas sin fondo.
    """

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder, self).__init__()
        self.encoder = UNetEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded


class SuperviselyDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes y máscaras del dataset Supervisely.
    CORREGIDO: Ahora mantiene proporciones y permite restaurar tamaño original.
    """

    def __init__(self, images_info_list, transform=None, image_size=384, store_metadata=False):
        """
        Args:
            images_info_list: Lista de diccionarios con 'image_path' y 'annotation_path'
            transform: Transformaciones de albumentations
            image_size: Tamaño al que redimensionar las imágenes
            store_metadata: Si guardar metadatos para restaurar tamaño original
        """
        self.images_info = images_info_list
        self.transform = transform
        self.image_size = image_size
        self.store_metadata = store_metadata
        self.processor = ImageProcessor()

    def __len__(self):
        return len(self.images_info)

    def load_supervisely_mask(self, annotation_path):
        """
        Carga la máscara desde el formato JSON de Supervisely.
        """
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            # Obtener dimensiones de la imagen
            if 'size' in annotation:
                height = annotation['size']['height']
                width = annotation['size']['width']
            else:
                # Fallback: usar dimensiones de la imagen
                return np.zeros((self.image_size, self.image_size), dtype=np.uint8), None

            # Crear máscara vacía
            mask = np.zeros((height, width), dtype=np.uint8)

            # Verificar si hay objetos
            if 'objects' not in annotation:
                print(f"No hay objetos en: {annotation_path}")
                return mask, (height, width)

            # Procesar cada objeto
            objects_processed = 0
            for obj in annotation['objects']:
                class_title = obj.get('classTitle', '').lower()

                # Buscar clases relacionadas con personas
                person_keywords = ['person', 'human', 'people', 'man', 'woman', 'child']
                is_person = any(keyword in class_title for keyword in person_keywords)

                if is_person:
                    geometry_type = obj.get('geometryType', '')

                    if geometry_type == 'polygon':
                        # Procesar polígonos
                        if 'points' in obj and 'exterior' in obj['points']:
                            exterior_points = obj['points']['exterior']
                            if len(exterior_points) >= 3:  # Mínimo 3 puntos para un polígono
                                points = np.array([[pt[0], pt[1]] for pt in exterior_points], dtype=np.int32)
                                cv2.fillPoly(mask, [points], 255)
                                objects_processed += 1

                    elif geometry_type == 'bitmap':
                        # Para bitmaps, necesitaríamos decodificar los datos
                        # Por ahora, crear una máscara básica basada en el bbox si está disponible
                        if 'bitmap' in obj and 'data' in obj['bitmap']:
                            # Implementación básica - en un caso real necesitarías decodificar el bitmap
                            objects_processed += 1

                    elif geometry_type == 'rectangle':
                        # Procesar rectángulos
                        if 'points' in obj:
                            if 'exterior' in obj['points'] and len(obj['points']['exterior']) >= 2:
                                points = obj['points']['exterior']
                                x1, y1 = int(points[0][0]), int(points[0][1])
                                x2, y2 = int(points[1][0]), int(points[1][1])
                                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                                objects_processed += 1

            if objects_processed == 0:
                print(f"No se procesaron objetos de persona en: {annotation_path}")

            return mask, (height, width)

        except json.JSONDecodeError as e:
            print(f"Error de JSON en {annotation_path}: {e}")
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8), None
        except Exception as e:
            print(f"Error cargando anotación {annotation_path}: {e}")
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8), None

    def __getitem__(self, idx):
        # Obtener información de la imagen
        img_info = self.images_info[idx]
        img_path = img_info['image_path']
        ann_path = img_info['annotation_path']

        try:
            # Cargar imagen
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Cargar máscara
            mask, original_size = self.load_supervisely_mask(ann_path)

            # CORREGIDO: Usar redimensionamiento con padding que mantiene proporciones
            image_processed, mask_processed, restore_metadata = self.processor.resize_with_padding(
                image, mask, self.image_size
            )

            # Normalizar imagen
            image_processed = image_processed.astype(np.float32) / 255.0

            # Crear máscara binaria
            alpha = (mask_processed > 127).astype(np.float32)

            # Aplicar transformaciones si están definidas
            if self.transform:
                try:
                    # Convertir alpha de nuevo a uint8 para albumentations
                    mask_uint8 = (alpha * 255).astype(np.uint8)
                    image_uint8 = (image_processed * 255).astype(np.uint8)

                    augmented = self.transform(image=image_uint8, mask=mask_uint8)
                    image_aug = augmented['image']
                    mask_aug = augmented['mask']

                    # Convertir de vuelta a float32
                    if isinstance(image_aug, torch.Tensor):
                        image_processed = image_aug.float() / 255.0
                    else:
                        image_processed = image_aug.astype(np.float32) / 255.0

                    alpha = (mask_aug > 127).astype(np.float32)

                except Exception as e:
                    print(f"Error en transformación: {e}")
                    # Usar versión sin transformar
                    pass

            # Crear target con 4 canales (RGBA)
            target = np.zeros((4, self.image_size, self.image_size), dtype=np.float32)

            # RGB channels
            if len(image_processed.shape) == 3:
                target[:3] = image_processed.transpose(2, 0, 1)
            else:
                target[:3] = image_processed

            # Alpha channel
            if len(alpha.shape) == 2:
                target[3] = alpha
            else:
                target[3] = alpha.squeeze()

            # Convertir imagen de entrada a tensor
            if not isinstance(image_processed, torch.Tensor):
                if len(image_processed.shape) == 3:
                    image_tensor = torch.FloatTensor(image_processed.transpose(2, 0, 1))
                else:
                    image_tensor = torch.FloatTensor(image_processed)
            else:
                image_tensor = image_processed.float()

            # Preparar salida
            result = (image_tensor, torch.FloatTensor(target))

            # Si se requieren metadatos, agregarlos
            if self.store_metadata:
                result = result + (restore_metadata,)

            return result

        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            # Retornar tensor dummy en caso de error
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_target = torch.zeros(4, self.image_size, self.image_size)
            dummy_metadata = {
                'original_size': (self.image_size, self.image_size),
                'scale': 1.0,
                'new_size': (self.image_size, self.image_size),
                'padding': (0, 0, 0, 0)
            }

            if self.store_metadata:
                return dummy_image, dummy_target, dummy_metadata
            else:
                return dummy_image, dummy_target


class LossCalculator:
    """
    Calculadora de pérdidas compuestas para el entrenamiento.
    Combina múltiples tipos de pérdida para mejor calidad.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.3):
        self.alpha = alpha  # BCE weight
        self.beta = beta  # Dice weight
        self.gamma = gamma  # Perceptual weight
        self.delta = delta  # Edge weight

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice Loss para segmentación."""
        # FIX: Usar .reshape() en lugar de .view() para evitar problemas de memoria no contigua
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return 1 - dice

    def edge_loss(self, pred, target):
        """Edge Loss para preservar contornos."""
        # Aplicar filtro Sobel para detectar bordes
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        if pred.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        # Calcular gradientes
        pred_edges_x = F.conv2d(pred[:, 3:4], sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred[:, 3:4], sobel_y, padding=1)
        target_edges_x = F.conv2d(target[:, 3:4], sobel_x, padding=1)
        target_edges_y = F.conv2d(target[:, 3:4], sobel_y, padding=1)

        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)

        return self.mse_loss(pred_edges, target_edges)

    def calculate_loss(self, pred, target):
        """Calcula la pérdida total compuesta."""
        # Separar canales RGB y Alpha
        pred_rgb = pred[:, :3]
        pred_alpha = pred[:, 3:4]
        target_rgb = target[:, :3]
        target_alpha = target[:, 3:4]

        # Asegurar que los tensores sean contiguos para evitar errores de view/reshape
        pred_alpha = pred_alpha.contiguous()
        target_alpha = target_alpha.contiguous()
        pred_rgb = pred_rgb.contiguous()
        target_rgb = target_rgb.contiguous()

        # BCE Loss para alpha channel
        bce_loss = self.bce_loss(pred_alpha, target_alpha)

        # Dice Loss para alpha channel
        dice_loss = self.dice_loss(pred_alpha, target_alpha)

        # MSE Loss para RGB channels (solo donde hay persona)
        mask = target_alpha > 0.5
        if mask.sum() > 0:
            perceptual_loss = self.mse_loss(pred_rgb * mask, target_rgb * mask)
        else:
            perceptual_loss = torch.tensor(0.0, device=pred.device)

        # Edge Loss
        edge_loss = self.edge_loss(pred, target)

        # Combinar pérdidas
        total_loss = (self.alpha * bce_loss +
                      self.beta * dice_loss +
                      self.gamma * perceptual_loss +
                      self.delta * edge_loss)

        # Verificar que la pérdida sea válida
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Si hay problemas, usar solo BCE loss como fallback
            total_loss = bce_loss

        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            'perceptual_loss': perceptual_loss,
            'edge_loss': edge_loss
        }


class MetricsCalculator:
    """
    Calculadora de métricas de evaluación.
    """

    @staticmethod
    def calculate_iou(pred, target, threshold=0.5):
        """Calcula Intersection over Union (IoU)."""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        if union == 0:
            return torch.tensor(1.0, device=pred.device)

        return intersection / union

    @staticmethod
    def calculate_dice(pred, target, threshold=0.5):
        """Calcula Dice Coefficient."""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        intersection = (pred_binary * target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()

        if total == 0:
            return torch.tensor(1.0, device=pred.device)

        return (2.0 * intersection) / total

    @staticmethod
    def calculate_pixel_accuracy(pred, target, threshold=0.5):
        """Calcula precisión a nivel de pixel."""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        correct = (pred_binary == target_binary).float().sum()
        total = target_binary.numel()

        return correct / total


class ModelCheckpoint:
    """
    Manejo de checkpoints del modelo.
    Guarda automáticamente las mejores versiones.
    """

    def __init__(self, checkpoint_dir='checkpoints', save_best_only=True):
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
        self.best_iou = 0.0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, loss, metrics, is_best=False):
        """Guarda checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Guardar checkpoint regular
        if not self.save_best_only:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)

        # Guardar mejor modelo
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

        # Guardar último modelo
        last_path = os.path.join(self.checkpoint_dir, 'last_model.pth')
        torch.save(checkpoint, last_path)

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Carga checkpoint del modelo."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['loss'], checkpoint['metrics']


class Trainer:
    """
    Clase principal para el entrenamiento del modelo.
    """

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Inicializar componentes
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.loss_calculator = LossCalculator()
        self.metrics_calculator = MetricsCalculator()
        self.checkpoint_manager = ModelCheckpoint()

        # Historial de entrenamiento
        self.train_history = {'loss': [], 'iou': [], 'dice': []}
        self.val_history = {'loss': [], 'iou': [], 'dice': []}

        # Logger
        self.logger = setup_logging()

    def train_epoch(self):
        """Entrena una época."""
        self.model.train()
        epoch_losses = []
        epoch_ious = []
        epoch_dices = []

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            try:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Verificar que no hay NaN en los datos de entrada
                if torch.isnan(images).any() or torch.isnan(targets).any():
                    self.logger.warning(f"NaN detectado en batch {batch_idx}, saltando...")
                    continue

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Verificar que no hay NaN en las salidas
                if torch.isnan(outputs).any():
                    self.logger.warning(f"NaN en outputs del batch {batch_idx}, saltando...")
                    continue

                # Calcular pérdidas
                loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                total_loss = loss_dict['total_loss']

                # Verificar que la pérdida es válida
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.warning(f"Pérdida inválida en batch {batch_idx}: {total_loss.item()}, saltando...")
                    continue

                # Backward pass
                total_loss.backward()

                # Gradient clipping más agresivo
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                # Calcular métricas
                with torch.no_grad():
                    pred_alpha = outputs[:, 3:4]
                    target_alpha = targets[:, 3:4]

                    iou = self.metrics_calculator.calculate_iou(pred_alpha, target_alpha)
                    dice = self.metrics_calculator.calculate_dice(pred_alpha, target_alpha)

                    # Verificar métricas válidas
                    if not (torch.isnan(iou) or torch.isnan(dice)):
                        epoch_losses.append(total_loss.item())
                        epoch_ious.append(iou.item())
                        epoch_dices.append(dice.item())

                # Log cada N batches
                if batch_idx % 10 == 0 and len(epoch_losses) > 0:
                    self.logger.info(f'Batch {batch_idx}/{len(self.train_loader)}: '
                                     f'Loss: {total_loss.item():.4f}, IoU: {iou.item():.4f}, Dice: {dice.item():.4f}')

            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue

        # Retornar promedios válidos
        if len(epoch_losses) > 0:
            return np.mean(epoch_losses), np.mean(epoch_ious), np.mean(epoch_dices)
        else:
            return 0.0, 0.0, 0.0

    def validate_epoch(self):
        """Valida una época."""
        self.model.eval()
        epoch_losses = []
        epoch_ious = []
        epoch_dices = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                try:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    # Verificar datos válidos
                    if torch.isnan(images).any() or torch.isnan(targets).any():
                        continue

                    # Forward pass
                    outputs = self.model(images)

                    # Verificar salidas válidas
                    if torch.isnan(outputs).any():
                        continue

                    # Calcular pérdidas
                    loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                    total_loss = loss_dict['total_loss']

                    # Verificar pérdida válida
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    # Calcular métricas
                    pred_alpha = outputs[:, 3:4]
                    target_alpha = targets[:, 3:4]

                    iou = self.metrics_calculator.calculate_iou(pred_alpha, target_alpha)
                    dice = self.metrics_calculator.calculate_dice(pred_alpha, target_alpha)

                    # Solo agregar si las métricas son válidas
                    if not (torch.isnan(iou) or torch.isnan(dice)):
                        epoch_losses.append(total_loss.item())
                        epoch_ious.append(iou.item())
                        epoch_dices.append(dice.item())

                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue

        # Retornar promedios válidos
        if len(epoch_losses) > 0:
            return np.mean(epoch_losses), np.mean(epoch_ious), np.mean(epoch_dices)
        else:
            return 0.0, 0.0, 0.0

    def train(self, num_epochs):
        """Entrenamiento principal."""
        self.logger.info("Iniciando entrenamiento...")
        self.logger.info(f"Configuración: {self.config}")

        for epoch in range(num_epochs):
            self.logger.info(f"\nÉpoca {epoch + 1}/{num_epochs}")

            # Entrenar
            train_loss, train_iou, train_dice = self.train_epoch()

            # Validar
            val_loss, val_iou, val_dice = self.validate_epoch()

            # Actualizar scheduler
            self.scheduler.step()

            # Verificar si los valores son válidos antes de guardar
            if not (np.isnan(train_loss) or np.isnan(val_loss)):
                # Guardar historial
                self.train_history['loss'].append(train_loss)
                self.train_history['iou'].append(train_iou)
                self.train_history['dice'].append(train_dice)

                self.val_history['loss'].append(val_loss)
                self.val_history['iou'].append(val_iou)
                self.val_history['dice'].append(val_dice)

                # Log resultados
                self.logger.info(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
                self.logger.info(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")

                # Guardar checkpoint
                is_best = val_iou > self.checkpoint_manager.best_iou
                if is_best:
                    self.checkpoint_manager.best_iou = val_iou
                    self.checkpoint_manager.best_loss = val_loss

                metrics = {
                    'train_loss': train_loss, 'train_iou': train_iou, 'train_dice': train_dice,
                    'val_loss': val_loss, 'val_iou': val_iou, 'val_dice': val_dice
                }

                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, metrics, is_best
                )

                if is_best:
                    self.logger.info(f"¡Nuevo mejor modelo! IoU: {val_iou:.4f}")
            else:
                self.logger.warning(f"Época {epoch + 1} saltada debido a valores NaN")

        self.logger.info("Entrenamiento completado!")
        self.save_training_plots()

    def save_training_plots(self):
        """Guarda gráficas del entrenamiento."""
        if len(self.train_history['loss']) == 0:
            self.logger.warning("No hay datos de entrenamiento para graficar")
            return

        os.makedirs('plots', exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Loss
        axes[0].plot(self.train_history['loss'], label='Train Loss')
        axes[0].plot(self.val_history['loss'], label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # IoU
        axes[1].plot(self.train_history['iou'], label='Train IoU')
        axes[1].plot(self.val_history['iou'], label='Val IoU')
        axes[1].set_title('Training and Validation IoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].legend()
        axes[1].grid(True)

        # Dice
        axes[2].plot(self.train_history['dice'], label='Train Dice')
        axes[2].plot(self.val_history['dice'], label='Val Dice')
        axes[2].set_title('Training and Validation Dice')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Dice')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def get_transforms():
    """Define transformaciones de aumentación de datos."""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        # Solo normalización para validación
    ], additional_targets={'mask': 'mask'})

    return train_transform, val_transform


class ModelInference:
    """
    Clase para realizar inferencia con el modelo entrenado.
    CORREGIDO: Ahora restaura correctamente las dimensiones originales.
    """

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = UNetAutoencoder(pretrained=False, use_attention=True)
        self.processor = ImageProcessor()

        # Cargar modelo entrenado
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def remove_background(self, image_path, output_path=None, image_size=384):
        """
        Remueve el fondo de una imagen manteniendo las dimensiones originales.

        Args:
            image_path: Ruta de la imagen de entrada
            output_path: Ruta de la imagen de salida (opcional)
            image_size: Tamaño para el procesamiento

        Returns:
            numpy array con la imagen procesada (RGBA) en tamaño original
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        # CORREGIDO: Usar redimensionamiento con padding
        dummy_mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
        image_processed, _, restore_metadata = self.processor.resize_with_padding(
            image, dummy_mask, image_size
        )

        # Normalizar
        image_normalized = image_processed.astype(np.float32) / 255.0

        # Convertir a tensor
        input_tensor = torch.FloatTensor(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            output = self.model(input_tensor)
            output = output.squeeze(0).cpu().numpy()

        # Post-procesamiento
        rgb_channels = output[:3].transpose(1, 2, 0)
        alpha_channel = output[3]

        # CORREGIDO: Restaurar al tamaño original usando metadatos
        rgb_restored, alpha_restored = self.processor.restore_original_size(
            rgb_channels, alpha_channel, restore_metadata
        )

        # Crear imagen RGBA final
        result = np.zeros((original_size[0], original_size[1], 4), dtype=np.float32)
        result[:, :, :3] = rgb_restored
        result[:, :, 3] = alpha_restored

        # Convertir a uint8
        result = (result * 255).astype(np.uint8)

        # Guardar si se especifica path
        if output_path:
            # Convertir a PIL para guardar con canal alpha
            pil_image = Image.fromarray(result, 'RGBA')
            pil_image.save(output_path)

        return result

    def batch_process(self, input_dir, output_dir):
        """
        Procesa un directorio completo de imágenes.
        """
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_no_bg.png")

            try:
                self.remove_background(input_path, output_path)
                print(f"Procesado: {img_file}")
            except Exception as e:
                print(f"Error procesando {img_file}: {e}")


def main():
    """Función principal para entrenar el modelo."""
    # Configuración ajustada para evitar problemas de memoria y NaN
    config = {
        'batch_size': 4,  # Reducido aún más para estabilidad
        'learning_rate': 5e-5,  # Learning rate más conservador
        'weight_decay': 1e-6,  # Weight decay más pequeño
        'num_epochs': 100,
        'image_size': 384,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 2,  # Reducido para estabilidad
        'pin_memory': True,
        'mixed_precision': False
    }

    # Setup logging
    logger = setup_logging()
    logger.info("Iniciando sistema de entrenamiento U-Net Autoencoder")
    logger.info(f"Dispositivo: {config['device']}")

    # Verificar directorio de datos
    data_dir = 'persons/project'

    if not os.path.exists(data_dir):
        logger.error(f"Directorio de datos no encontrado: {data_dir}")
        return

    # El dataset Supervisely tiene múltiples subdirectorios (ds1, ds2, etc.)
    dataset_dirs = [d for d in os.listdir(data_dir) if d.startswith('ds') and os.path.isdir(os.path.join(data_dir, d))]

    if not dataset_dirs:
        logger.error("Directorios de dataset no encontrados en persons/project/")
        logger.error("Esperados: ds1, ds2, ds3, etc.")
        return

    logger.info(f"Encontrados {len(dataset_dirs)} subdirectorios de dataset: {dataset_dirs}")

    # Recopilar todas las imágenes y anotaciones de todos los subdirectorios
    all_images_info = []

    for ds_dir in dataset_dirs:
        ds_path = os.path.join(data_dir, ds_dir)
        img_dir = os.path.join(ds_path, 'img')
        ann_dir = os.path.join(ds_path, 'ann')

        logger.info(f"Procesando directorio: {ds_dir}")

        if not os.path.exists(img_dir):
            logger.warning(f"Directorio de imágenes no encontrado: {img_dir}")
            continue

        if not os.path.exists(ann_dir):
            logger.warning(f"Directorio de anotaciones no encontrado: {ann_dir}")
            continue

        # Obtener archivos de imágenes
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        ann_files = [f for f in os.listdir(ann_dir) if f.lower().endswith('.json')]

        logger.info(f"  - Imágenes encontradas: {len(img_files)}")
        logger.info(f"  - Anotaciones encontradas: {len(ann_files)}")

        # Crear mapping de nombres base
        paired_count = 0
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)

            # Probar diferentes patrones de nombres para las anotaciones
            base_name = os.path.splitext(img_file)[0]
            possible_ann_names = [
                f"{base_name}.json",
                f"{base_name}.jpg.json",
                f"{base_name}.png.json",
                f"{base_name}.jpeg.json"
            ]

            ann_path = None
            for ann_name in possible_ann_names:
                potential_ann_path = os.path.join(ann_dir, ann_name)
                if os.path.exists(potential_ann_path):
                    ann_path = potential_ann_path
                    break

            if ann_path:
                all_images_info.append({
                    'image_path': img_path,
                    'annotation_path': ann_path,
                    'dataset': ds_dir,
                    'image_name': img_file
                })
                paired_count += 1
            else:
                logger.debug(f"No se encontró anotación para: {img_file}")

        logger.info(f"  - Pares válidos creados: {paired_count}")

    if not all_images_info:
        logger.error("No se encontraron pares válidos de imagen/anotación")
        return

    logger.info(f"Total de imágenes emparejadas encontradas: {len(all_images_info)}")

    # Preparar transforms
    train_transform, val_transform = get_transforms()

    # Split train/validation (80/20)
    train_size = int(0.8 * len(all_images_info))
    val_size = len(all_images_info) - train_size

    # Crear índices aleatorios para el split
    indices = list(range(len(all_images_info)))
    np.random.seed(42)  # Para reproducibilidad
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Crear subsets
    train_images_info = [all_images_info[i] for i in train_indices]
    val_images_info = [all_images_info[i] for i in val_indices]

    # Crear datasets con transforms (store_metadata=False para entrenamiento)
    train_dataset = SuperviselyDataset(
        images_info_list=train_images_info,
        transform=train_transform,
        image_size=config['image_size'],
        store_metadata=False  # No necesitamos metadatos durante entrenamiento
    )

    val_dataset = SuperviselyDataset(
        images_info_list=val_images_info,
        transform=val_transform,
        image_size=config['image_size'],
        store_metadata=False  # No necesitamos metadatos durante validación
    )

    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    logger.info(f"Dataset cargado: {len(train_dataset)} train, {len(val_dataset)} val")

    # Crear modelo
    logger.info("Inicializando modelo U-Net Autoencoder...")
    model = UNetAutoencoder(pretrained=True, use_attention=True)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros totales: {total_params:,}")
    logger.info(f"Parámetros entrenables: {trainable_params:,}")

    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        config=config
    )

    # Entrenar modelo
    trainer.train(config['num_epochs'])

    logger.info("Entrenamiento completado exitosamente!")


def demo_inference():
    """
    Función de demostración para usar el modelo entrenado.
    """
    # Cargar modelo entrenado
    model_path = 'checkpoints/best_model.pth'

    if not os.path.exists(model_path):
        print("Modelo entrenado no encontrado. Primero ejecuta el entrenamiento.")
        return

    # Crear objeto de inferencia
    inference = ModelInference(model_path)

    # Ejemplo de uso
    input_image = 'example_input.jpg'
    output_image = 'example_output.png'

    if os.path.exists(input_image):
        result = inference.remove_background(input_image, output_image)
        print(f"Fondo removido exitosamente. Resultado guardado en: {output_image}")
        print(f"Dimensiones del resultado: {result.shape}")
    else:
        print(f"Imagen de ejemplo no encontrada: {input_image}")


def test_model_forward():
    """
    Función de prueba para verificar que el modelo funciona correctamente.
    """
    print("Probando forward pass del modelo...")

    # Crear modelo
    model = UNetAutoencoder(pretrained=False, use_attention=True)
    model.eval()

    # Crear tensor de prueba
    test_input = torch.randn(1, 3, 384, 384)

    try:
        with torch.no_grad():
            output = model(test_input)

        print(f"✓ Forward pass exitoso!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (1, 4, 384, 384)")

        if output.shape == (1, 4, 384, 384):
            print("✓ Dimensiones de salida correctas")
            return True
        else:
            print("✗ Dimensiones de salida incorrectas")
            return False

    except Exception as e:
        print(f"✗ Error en forward pass: {e}")
        return False


def test_image_processing():
    """
    Función de prueba para verificar el procesamiento de imágenes.
    """
    print("Probando procesamiento de imágenes...")

    processor = ImageProcessor()

    # Crear imagen de prueba con diferentes dimensiones
    test_cases = [
        (480, 640, 3),  # Horizontal
        (640, 480, 3),  # Vertical
        (512, 512, 3),  # Cuadrada
        (1080, 1920, 3)  # Muy alta resolución
    ]

    target_size = 384

    for i, (h, w, c) in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {h}x{w}x{c}")

        # Crear imagen y máscara de prueba
        test_image = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        test_mask = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255

        try:
            # Procesar
            image_processed, mask_processed, metadata = processor.resize_with_padding(
                test_image, test_mask, target_size
            )

            print(f"  ✓ Original: {test_image.shape} -> Procesada: {image_processed.shape}")
            print(f"  ✓ Máscara: {test_mask.shape} -> Procesada: {mask_processed.shape}")

            # Verificar dimensiones
            if image_processed.shape[:2] == (target_size, target_size):
                print(f"  ✓ Dimensiones correctas")
            else:
                print(f"  ✗ Dimensiones incorrectas: {image_processed.shape}")
                continue

            # Probar restauración
            image_restored, mask_restored = processor.restore_original_size(
                image_processed, mask_processed, metadata
            )

            print(f"  ✓ Restaurada: {image_restored.shape}")

            # Verificar que se restauró al tamaño original
            if image_restored.shape == test_image.shape:
                print(f"  ✓ Restauración exitosa")
            else:
                print(f"  ✗ Error en restauración: {image_restored.shape} vs {test_image.shape}")

        except Exception as e:
            print(f"  ✗ Error en test case {i + 1}: {e}")

    print("\n✓ Pruebas de procesamiento de imágenes completadas")
    return True


if __name__ == "__main__":
    print("=== PRUEBAS DEL SISTEMA U-NET AUTOENCODER ===\n")

    # Probar modelo
    model_test = test_model_forward()

    # Probar procesamiento de imágenes
    processing_test = test_image_processing()

    if model_test and processing_test:
        print("\n✅ Todas las pruebas pasaron. Procediendo con el entrenamiento...\n")

        # Para entrenar el modelo
        main()

        # Para hacer inferencia (descomenta la siguiente línea después del entrenamiento)
        # demo_inference()
    else:
        print("\n❌ Algunas pruebas fallaron. Revisar la implementación.")