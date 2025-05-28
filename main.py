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
        
        # Asegurar mismas dimensiones
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
        x = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x)  # Skip 1: 64 channels
        
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        skip_connections.append(x)  # Skip 2: 64 channels
        
        x = self.layer2(x)
        skip_connections.append(x)  # Skip 3: 128 channels
        
        x = self.layer3(x)
        skip_connections.append(x)  # Skip 4: 256 channels
        
        x = self.layer4(x)
        skip_connections.append(x)  # Skip 5: 512 channels
        
        # Bottleneck
        x = self.bottleneck(x)
        
        return x, skip_connections

class UNetDecoder(nn.Module):
    """
    Decoder path del U-Net con Attention Gates.
    Reconstruye la imagen enfocándose en las personas.
    """
    def __init__(self, use_attention=True):
        super(UNetDecoder, self).__init__()
        self.use_attention = use_attention
        
        # Upsampling layers
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
        
    def forward(self, x, skip_connections):
        # Decoder path
        skips = skip_connections[::-1]  # Reverse order
        
        # Up 1
        x = self.up1(x)
        if self.use_attention:
            skip = self.att1(skips[0], x)
        else:
            skip = skips[0]
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        
        # Up 2
        x = self.up2(x)
        if self.use_attention:
            skip = self.att2(skips[1], x)
        else:
            skip = skips[1]
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        
        # Up 3
        x = self.up3(x)
        if self.use_attention:
            skip = self.att3(skips[2], x)
        else:
            skip = skips[2]
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)
        
        # Up 4
        x = self.up4(x)
        if self.use_attention:
            skip = self.att4(skips[3], x)
        else:
            skip = skips[3]
        x = torch.cat([x, skip], dim=1)
        x = self.conv4(x)
        
        # Up 5
        x = self.up5(x)
        if self.use_attention:
            skip = self.att5(skips[4], x)
        else:
            skip = skips[4]
        x = torch.cat([x, skip], dim=1)
        x = self.conv5(x)
        
        # Final output
        x = self.final_conv(x)
        
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
    """
    def __init__(self, images_dir, annotations_dir, transform=None, image_size=256):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_size = image_size
        
        # Obtener lista de imágenes
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def load_supervisely_mask(self, annotation_path):
        """
        Carga la máscara desde el formato JSON de Supervisely.
        """
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            # Crear máscara vacía
            height = annotation['size']['height']
            width = annotation['size']['width']
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Procesar objetos
            for obj in annotation['objects']:
                if obj['classTitle'] in ['person', 'person_poly', 'person_bmp']:
                    # Extraer puntos de la geometría
                    if 'points' in obj['geometryType']:
                        points = np.array(obj['points']['exterior'], dtype=np.int32)
                        cv2.fillPoly(mask, [points], 255)
                    elif obj['geometryType'] == 'bitmap':
                        # Procesar bitmap si está disponible
                        pass
            
            return mask
        except Exception as e:
            # Retornar máscara vacía si hay error
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cargar máscara
        annotation_name = img_name.replace('.jpg', '.json').replace('.png', '.json')
        annotation_path = os.path.join(self.annotations_dir, annotation_name)
        
        if os.path.exists(annotation_path):
            mask = self.load_supervisely_mask(annotation_path)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Redimensionar
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Crear imagen objetivo (persona con fondo transparente)
        target = image.copy().astype(np.float32) / 255.0
        alpha = (mask > 0).astype(np.float32)
        
        # Aplicar transformaciones
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Normalizar
        image = image.astype(np.float32) / 255.0
        alpha = alpha.reshape(1, self.image_size, self.image_size)
        
        # Crear target con 4 canales (RGBA)
        target_rgba = np.concatenate([target.transpose(2, 0, 1), alpha], axis=0)
        
        return torch.FloatTensor(image.transpose(2, 0, 1)), torch.FloatTensor(target_rgba)

class LossCalculator:
    """
    Calculadora de pérdidas compuestas para el entrenamiento.
    Combina múltiples tipos de pérdida para mejor calidad.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.3):
        self.alpha = alpha  # BCE weight
        self.beta = beta    # Dice weight
        self.gamma = gamma  # Perceptual weight
        self.delta = delta  # Edge weight
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice Loss para segmentación."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
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
        
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-6)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)
        
        return self.mse_loss(pred_edges, target_edges)
    
    def calculate_loss(self, pred, target):
        """Calcula la pérdida total compuesta."""
        # Separar canales RGB y Alpha
        pred_rgb = pred[:, :3]
        pred_alpha = pred[:, 3:4]
        target_rgb = target[:, :3]
        target_alpha = target[:, 3:4]
        
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
            return torch.tensor(1.0)
        
        return intersection / union
    
    @staticmethod
    def calculate_dice(pred, target, threshold=0.5):
        """Calcula Dice Coefficient."""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()
        
        if total == 0:
            return torch.tensor(1.0)
        
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
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calcular pérdidas
            loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calcular métricas
            with torch.no_grad():
                pred_alpha = outputs[:, 3:4]
                target_alpha = targets[:, 3:4]
                
                iou = self.metrics_calculator.calculate_iou(pred_alpha, target_alpha)
                dice = self.metrics_calculator.calculate_dice(pred_alpha, target_alpha)
                
                epoch_losses.append(total_loss.item())
                epoch_ious.append(iou.item())
                epoch_dices.append(dice.item())
            
            # Log cada N batches
            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}/{len(self.train_loader)}: '
                               f'Loss: {total_loss.item():.4f}, IoU: {iou.item():.4f}, Dice: {dice.item():.4f}')
        
        return np.mean(epoch_losses), np.mean(epoch_ious), np.mean(epoch_dices)
    
    def validate_epoch(self):
        """Valida una época."""
        self.model.eval()
        epoch_losses = []
        epoch_ious = []
        epoch_dices = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calcular pérdidas
                loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                total_loss = loss_dict['total_loss']
                
                # Calcular métricas
                pred_alpha = outputs[:, 3:4]
                target_alpha = targets[:, 3:4]
                
                iou = self.metrics_calculator.calculate_iou(pred_alpha, target_alpha)
                dice = self.metrics_calculator.calculate_dice(pred_alpha, target_alpha)
                
                epoch_losses.append(total_loss.item())
                epoch_ious.append(iou.item())
                epoch_dices.append(dice.item())
        
        return np.mean(epoch_losses), np.mean(epoch_ious), np.mean(epoch_dices)
    
    def train(self, num_epochs):
        """Entrenamiento principal."""
        self.logger.info("Iniciando entrenamiento...")
        self.logger.info(f"Configuración: {self.config}")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nÉpoca {epoch+1}/{num_epochs}")
            
            # Entrenar
            train_loss, train_iou, train_dice = self.train_epoch()
            
            # Validar
            val_loss, val_iou, val_dice = self.validate_epoch()
            
            # Actualizar scheduler
            self.scheduler.step()
            
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
        
        self.logger.info("Entrenamiento completado!")
        self.save_training_plots()
    
    def save_training_plots(self):
        """Guarda gráficas del entrenamiento."""
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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def main():
    """Función principal para entrenar el modelo."""
    # Configuración
    config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'image_size': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'pin_memory': True
    }
    
    # Setup logging
    logger = setup_logging()
    logger.info("Iniciando sistema de entrenamiento U-Net Autoencoder")
    logger.info(f"Dispositivo: {config['device']}")
    
    # Verificar directorio de datos
    data_dir = 'data/supervisely_persons'
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        logger.error("Directorio de datos no encontrado. Por favor descarga el dataset Supervisely Persons.")
        logger.error(f"Esperado: {images_dir} y {annotations_dir}")
        return
    
    # Preparar transforms
    train_transform, val_transform = get_transforms()
    
    # Crear datasets
    logger.info("Cargando datasets...")
    full_dataset = SuperviselyDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transform=None,
        image_size=config['image_size']
    )
    
    # Split train/validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Aplicar transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
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

class ModelInference:
    """
    Clase para realizar inferencia con el modelo entrenado.
    """
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = UNetAutoencoder(pretrained=False, use_attention=True)
        
        # Cargar modelo entrenado
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Transform para inferencia
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def remove_background(self, image_path, output_path=None):
        """
        Remueve el fondo de una imagen.
        
        Args:
            image_path: Ruta de la imagen de entrada
            output_path: Ruta de la imagen de salida (opcional)
        
        Returns:
            numpy array con la imagen procesada (RGBA)
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Redimensionar para el modelo
        image_resized = cv2.resize(image, (256, 256))
        
        # Aplicar transformaciones
        transformed = self.transform(image=image_resized)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inferencia
        with torch.no_grad():
            output = self.model(input_tensor)
            output = output.squeeze(0).cpu().numpy()
        
        # Post-procesamiento
        rgb_channels = output[:3].transpose(1, 2, 0)
        alpha_channel = output[3]
        
        # Redimensionar al tamaño original
        rgb_resized = cv2.resize(rgb_channels, (original_size[1], original_size[0]))
        alpha_resized = cv2.resize(alpha_channel, (original_size[1], original_size[0]))
        
        # Crear imagen RGBA
        result = np.zeros((original_size[0], original_size[1], 4), dtype=np.float32)
        result[:, :, :3] = rgb_resized
        result[:, :, 3] = alpha_resized
        
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
    else:
        print(f"Imagen de ejemplo no encontrada: {input_image}")

if __name__ == "__main__":
    # Para entrenar el modelo
    main()
    
    # Para hacer inferencia (descomenta la siguiente línea después del entrenamiento)
    # demo_inference()