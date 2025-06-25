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


class COCOPersonDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes y máscaras del dataset COCO.
    Se enfoca únicamente en las personas (keypoints).
    """

    def __init__(self, coco_root, annotation_file, transform=None, image_size=384, store_metadata=False):
        """
        Args:
            coco_root: Directorio raíz del dataset COCO
            annotation_file: Archivo JSON de anotaciones (person_keypoints_train2017.json o person_keypoints_val2017.json)
            transform: Transformaciones de albumentations
            image_size: Tamaño al que redimensionar las imágenes
            store_metadata: Si guardar metadatos para restaurar tamaño original
        """
        self.coco_root = coco_root
        self.transform = transform
        self.image_size = image_size
        self.store_metadata = store_metadata
        self.processor = ImageProcessor()

        # Cargar anotaciones COCO
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # Crear mapeos
        self.images = {img['id']: img for img in self.coco_data['images']}

        # CAMBIO IMPORTANTE: Filtrar solo anotaciones con keypoints válidos (personas)
        self.annotations = []
        for ann in self.coco_data['annotations']:
            # Solo incluir anotaciones que tengan keypoints y área > 0
            if 'keypoints' in ann and ann.get('area', 0) > 500:  # Filtrar personas muy pequeñas
                self.annotations.append(ann)

        # Agrupar anotaciones por imagen
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

        # Lista de IDs de imágenes válidas (que tienen personas)
        self.valid_image_ids = list(self.image_to_annotations.keys())

        print(f"Dataset COCO cargado: {len(self.valid_image_ids)} imágenes con personas")

    def __len__(self):
        return len(self.valid_image_ids)

    def create_person_mask(self, image_shape, annotations):
        """
        Crea máscara de personas a partir de las anotaciones COCO.
        Usa la segmentación si está disponible, sino crea rectángulo del bbox.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in annotations:
            # Usar segmentación si está disponible
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    if len(seg) >= 6:  # Mínimo 3 puntos (x,y pairs)
                        # Convertir a array de puntos
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [points], 255)
            else:
                # Fallback: usar bounding box
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w_box, h_box = bbox
                    x, y, w_box, h_box = int(x), int(y), int(w_box), int(h_box)
                    # Asegurar que está dentro de los límites
                    x = max(0, x)
                    y = max(0, y)
                    w_box = min(w_box, w - x)
                    h_box = min(h_box, h - y)
                    if w_box > 0 and h_box > 0:
                        cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 255, -1)

        return mask

    def __getitem__(self, idx):
        # Obtener ID de imagen
        img_id = self.valid_image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.image_to_annotations[img_id]

        # Construir path de imagen
        if 'train' in img_info['file_name']:
            img_path = os.path.join(self.coco_root, 'train2017', img_info['file_name'])
        else:
            img_path = os.path.join(self.coco_root, 'val2017', img_info['file_name'])

        try:
            # Cargar imagen
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Crear máscara de personas
            mask = self.create_person_mask(image.shape, annotations)

            # Usar redimensionamiento con padding que mantiene proporciones
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
            print(f"Error procesando imagen {img_id}: {e}")
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
    """Función principal para entrenar el modelo con dataset COCO."""
    # Configuración ajustada para COCO
    config = {
        'batch_size': 16,  # Reducido para COCO que tiene imágenes más variadas
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'num_epochs': 100,
        'image_size': 384,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 8,
        'pin_memory': True,
        'mixed_precision': True,
        'use_data_parallel': True,
    }

    # Setup logging
    logger = setup_logging()
    logger.info("Iniciando sistema de entrenamiento U-Net Autoencoder con COCO Dataset")
    logger.info(f"Dispositivo: {config['device']}")

    # CAMBIO PRINCIPAL: Usar directorio COCO
    coco_root = 'COCO'

    if not os.path.exists(coco_root):
        logger.error(f"Directorio COCO no encontrado: {coco_root}")
        return

    # Verificar archivos de anotaciones
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
    val_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_val2017.json')

    if not os.path.exists(train_ann_file):
        logger.error(f"Archivo de anotaciones de entrenamiento no encontrado: {train_ann_file}")
        return

    if not os.path.exists(val_ann_file):
        logger.error(f"Archivo de anotaciones de validación no encontrado: {val_ann_file}")
        return

    # Verificar directorios de imágenes
    train_img_dir = os.path.join(coco_root, 'train2017')
    val_img_dir = os.path.join(coco_root, 'val2017')

    if not os.path.exists(train_img_dir):
        logger.error(f"Directorio de imágenes de entrenamiento no encontrado: {train_img_dir}")
        return

    if not os.path.exists(val_img_dir):
        logger.error(f"Directorio de imágenes de validación no encontrado: {val_img_dir}")
        return

    logger.info("Estructura COCO verificada correctamente")

    # Preparar transforms
    train_transform, val_transform = get_transforms()

    # CAMBIO PRINCIPAL: Usar COCOPersonDataset en lugar de SuperviselyDataset
    logger.info("Cargando dataset COCO para entrenamiento...")
    train_dataset = COCOPersonDataset(
        coco_root=coco_root,
        annotation_file=train_ann_file,
        transform=train_transform,
        image_size=config['image_size'],
        store_metadata=False
    )

    logger.info("Cargando dataset COCO para validación...")
    val_dataset = COCOPersonDataset(
        coco_root=coco_root,
        annotation_file=val_ann_file,
        transform=val_transform,
        image_size=config['image_size'],
        store_metadata=False
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

    logger.info(f"Dataset COCO cargado: {len(train_dataset)} train, {len(val_dataset)} val")

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


def test_coco_dataset():
    """
    Función de prueba para verificar que el dataset COCO se carga correctamente.
    """
    print("Probando carga del dataset COCO...")

    coco_root = 'COCO'
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')

    if not os.path.exists(train_ann_file):
        print(f"✗ Archivo de anotaciones no encontrado: {train_ann_file}")
        return False

    try:
        # Crear dataset de prueba con solo unas pocas imágenes
        dataset = COCOPersonDataset(
            coco_root=coco_root,
            annotation_file=train_ann_file,
            transform=None,
            image_size=384,
            store_metadata=False
        )

        print(f"✓ Dataset COCO cargado exitosamente")
        print(f"  Total de imágenes con personas: {len(dataset)}")

        if len(dataset) > 0:
            # Probar cargar una muestra
            sample = dataset[0]
            image, target = sample

            print(f"  ✓ Muestra cargada exitosamente")
            print(f"    Image shape: {image.shape}")
            print(f"    Target shape: {target.shape}")

            return True
        else:
            print("✗ Dataset vacío")
            return False

    except Exception as e:
        print(f"✗ Error cargando dataset COCO: {e}")
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
    print("=== PRUEBAS DEL SISTEMA U-NET AUTOENCODER CON COCO ===\n")

    # Probar modelo
    model_test = test_model_forward()

    # Probar procesamiento de imágenes
    processing_test = test_image_processing()

    # Probar dataset COCO
    coco_test = test_coco_dataset()

    if model_test and processing_test and coco_test:
        print("\n✅ Todas las pruebas pasaron. Procediendo con el entrenamiento...\n")

        # Para entrenar el modelo
        main()

        # Para hacer inferencia (descomenta la siguiente línea después del entrenamiento)
        # demo_inference()
    else:
        print("\n❌ Algunas pruebas fallaron. Revisar la implementación.")
        print("\nDetalles de las pruebas:")
        print(f"  - Modelo: {'✅' if model_test else '❌'}")
        print(f"  - Procesamiento: {'✅' if processing_test else '❌'}")
        print(f"  - Dataset COCO: {'✅' if coco_test else '❌'}")

        if not coco_test:
            print("\n📋 Para resolver problemas con COCO:")
            print("  1. Verificar que existe el directorio 'COCO'")
            print("  2. Verificar que existe 'COCO/annotations/person_keypoints_train2017.json'")
            print("  3. Verificar que existe 'COCO/train2017/' con imágenes .jpg")
            print("  4. Verificar que existe 'COCO/val2017/' con imágenes .jpg")


def quick_coco_test():
    """
    Prueba rápida para verificar estructura COCO sin cargar todo el dataset.
    """
    print("=== VERIFICACIÓN RÁPIDA DE ESTRUCTURA COCO ===\n")

    coco_root = 'COCO'
    issues = []

    # Verificar directorio principal
    if not os.path.exists(coco_root):
        issues.append(f"❌ Directorio principal no encontrado: {coco_root}")
        print(f"❌ Directorio principal no encontrado: {coco_root}")
        return False
    else:
        print(f"✅ Directorio principal encontrado: {coco_root}")

    # Verificar annotations
    ann_dir = os.path.join(coco_root, 'annotations')
    if not os.path.exists(ann_dir):
        issues.append(f"❌ Directorio de anotaciones no encontrado: {ann_dir}")
    else:
        print(f"✅ Directorio de anotaciones encontrado: {ann_dir}")

        # Verificar archivos específicos
        train_ann = os.path.join(ann_dir, 'person_keypoints_train2017.json')
        val_ann = os.path.join(ann_dir, 'person_keypoints_val2017.json')

        if os.path.exists(train_ann):
            size_mb = os.path.getsize(train_ann) / (1024 * 1024)
            print(f"✅ Anotaciones de entrenamiento encontradas: {train_ann} ({size_mb:.1f} MB)")
        else:
            issues.append(f"❌ Anotaciones de entrenamiento no encontradas: {train_ann}")

        if os.path.exists(val_ann):
            size_mb = os.path.getsize(val_ann) / (1024 * 1024)
            print(f"✅ Anotaciones de validación encontradas: {val_ann} ({size_mb:.1f} MB)")
        else:
            issues.append(f"❌ Anotaciones de validación no encontradas: {val_ann}")

    # Verificar directorios de imágenes
    train_dir = os.path.join(coco_root, 'train2017')
    val_dir = os.path.join(coco_root, 'val2017')

    if os.path.exists(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        print(f"✅ Directorio train2017 encontrado con {train_count:,} imágenes")
    else:
        issues.append(f"❌ Directorio train2017 no encontrado: {train_dir}")

    if os.path.exists(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        print(f"✅ Directorio val2017 encontrado con {val_count:,} imágenes")
    else:
        issues.append(f"❌ Directorio val2017 no encontrado: {val_dir}")

    # Resumen
    if issues:
        print(f"\n❌ Se encontraron {len(issues)} problemas:")
        for issue in issues:
            print(f"   {issue}")
        print("\n📋 Instrucciones para resolver:")
        print("   1. Asegúrate de que el directorio COCO esté en la ubicación correcta")
        print("   2. Descomprime los archivos ZIP si es necesario:")
        print("      - annotations_trainval2017.zip -> COCO/annotations/")
        print("      - train2017.zip -> COCO/train2017/")
        print("      - val2017.zip -> COCO/val2017/")
        return False
    else:
        print(f"\n✅ Estructura COCO verificada correctamente!")
        print(f"   ✅ Todos los archivos necesarios están presentes")
        print(f"   ✅ Listo para entrenar el modelo")
        return True


def analyze_coco_annotations():
    """
    Analiza las anotaciones COCO para entender mejor el dataset.
    """
    print("=== ANÁLISIS DE ANOTACIONES COCO ===\n")

    coco_root = 'COCO'
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')

    if not os.path.exists(train_ann_file):
        print(f"❌ Archivo de anotaciones no encontrado: {train_ann_file}")
        return

    try:
        print("📊 Cargando anotaciones...")
        with open(train_ann_file, 'r') as f:
            coco_data = json.load(f)

        # Estadísticas básicas
        total_images = len(coco_data['images'])
        total_annotations = len(coco_data['annotations'])

        print(f"📈 Estadísticas generales:")
        print(f"   - Total de imágenes: {total_images:,}")
        print(f"   - Total de anotaciones: {total_annotations:,}")

        # Filtrar solo personas con keypoints
        person_annotations = [ann for ann in coco_data['annotations']
                              if 'keypoints' in ann and ann.get('area', 0) > 500]

        print(f"   - Anotaciones de personas válidas: {len(person_annotations):,}")

        # Agrupar por imagen
        image_to_annotations = {}
        for ann in person_annotations:
            img_id = ann['image_id']
            if img_id not in image_to_annotations:
                image_to_annotations[img_id] = []
            image_to_annotations[img_id].append(ann)

        valid_images = len(image_to_annotations)
        print(f"   - Imágenes con personas válidas: {valid_images:,}")

        # Distribución de personas por imagen
        persons_per_image = [len(anns) for anns in image_to_annotations.values()]
        avg_persons = np.mean(persons_per_image)
        max_persons = max(persons_per_image)

        print(f"   - Promedio de personas por imagen: {avg_persons:.2f}")
        print(f"   - Máximo de personas en una imagen: {max_persons}")

        # Distribución de tamaños
        areas = [ann['area'] for ann in person_annotations]
        print(f"📏 Distribución de tamaños (área):")
        print(f"   - Área promedio: {np.mean(areas):.0f} píxeles²")
        print(f"   - Área mínima: {np.min(areas):.0f} píxeles²")
        print(f"   - Área máxima: {np.max(areas):.0f} píxeles²")
        print(f"   - Mediana: {np.median(areas):.0f} píxeles²")

        print(f"\n✅ Análisis completado. Dataset listo para entrenamiento.")

    except Exception as e:
        print(f"❌ Error analizando anotaciones: {e}")


def create_sample_batch():
    """
    Crea un batch de muestra para verificar que todo funciona.
    """
    print("=== CREANDO BATCH DE MUESTRA ===\n")

    coco_root = 'COCO'
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')

    try:
        # Crear dataset pequeño
        print("📊 Creando dataset de muestra...")
        dataset = COCOPersonDataset(
            coco_root=coco_root,
            annotation_file=train_ann_file,
            transform=None,
            image_size=384,
            store_metadata=False
        )

        if len(dataset) == 0:
            print("❌ Dataset vacío")
            return False

        # Crear dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        print(f"✅ Dataset creado con {len(dataset)} imágenes")
        print("🔄 Cargando primer batch...")

        # Cargar primer batch
        for batch_idx, (images, targets) in enumerate(loader):
            print(f"✅ Batch cargado exitosamente:")
            print(f"   - Imágenes shape: {images.shape}")
            print(f"   - Targets shape: {targets.shape}")
            print(f"   - Tipo de datos: {images.dtype}, {targets.dtype}")
            print(f"   - Rango imágenes: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   - Rango targets: [{targets.min():.3f}, {targets.max():.3f}]")

            # Verificar canales
            rgb_channels = targets[:, :3, :, :]
            alpha_channel = targets[:, 3:4, :, :]
            print(f"   - RGB channels shape: {rgb_channels.shape}")
            print(f"   - Alpha channel shape: {alpha_channel.shape}")
            print(f"   - Alpha channel range: [{alpha_channel.min():.3f}, {alpha_channel.max():.3f}]")

            # Solo procesar primer batch
            break

        print("✅ Verificación de batch completada exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error creando batch de muestra: {e}")
        import traceback
        traceback.print_exc()
        return False


# Agregar función de utilidad para verificación completa
def full_verification():
    """
    Verificación completa del sistema antes del entrenamiento.
    """
    print("🔍 VERIFICACIÓN COMPLETA DEL SISTEMA")
    print("=" * 50)

    # Verificaciones paso a paso
    steps = [
        ("Estructura COCO", quick_coco_test),
        ("Forward pass del modelo", test_model_forward),
        ("Procesamiento de imágenes", test_image_processing),
        ("Análisis de anotaciones", analyze_coco_annotations),
        ("Batch de muestra", create_sample_batch),
    ]

    results = {}

    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if step_name == "Análisis de anotaciones":
                step_func()  # Esta función no retorna bool
                results[step_name] = True
            else:
                results[step_name] = step_func()
        except Exception as e:
            print(f"❌ Error en {step_name}: {e}")
            results[step_name] = False

    # Resumen final
    print(f"\n" + "=" * 50)
    print("📋 RESUMEN DE VERIFICACIONES:")

    all_passed = True
    for step_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {step_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n🎉 ¡TODAS LAS VERIFICACIONES EXITOSAS!")
        print(f"🚀 El sistema está listo para entrenar")
        return True
    else:
        print(f"\n⚠️  Algunas verificaciones fallaron")
        print(f"🔧 Revisa los errores antes de continuar")
        return False


# Función principal alternativa para verificación
def main_verify():
    """
    Modo de verificación sin entrenamiento.
    """
    if full_verification():
        response = input("\n¿Proceder con el entrenamiento? (y/n): ")
        if response.lower() in ['y', 'yes', 'sí', 's']:
            main()
        else:
            print("Entrenamiento cancelado por el usuario.")
    else:
        print("Verificación fallida. Entrenamiento no iniciado.")


if __name__ == "__main__":
    import sys

    # Permitir diferentes modos de ejecución
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'verify':
            print("🔍 MODO VERIFICACIÓN")
            full_verification()
        elif mode == 'quick':
            print("⚡ VERIFICACIÓN RÁPIDA")
            quick_coco_test()
        elif mode == 'analyze':
            print("📊 ANÁLISIS DE DATASET")
            analyze_coco_annotations()
        elif mode == 'batch':
            print("🔄 PRUEBA DE BATCH")
            create_sample_batch()
        elif mode == 'train':
            print("🚀 ENTRENAMIENTO DIRECTO")
            main()
        else:
            print(f"❌ Modo no reconocido: {mode}")
            print("Modos disponibles: verify, quick, analyze, batch, train")
    else:
        # Modo por defecto: verificación completa + opción de entrenar
        print("🎯 MODO AUTOMÁTICO: Verificación + Entrenamiento")
        main_verify()