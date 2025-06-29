import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16
import numpy as np
import cv2
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


warnings.filterwarnings('ignore')


class UNetHarmonizer(nn.Module):
    """
    U-Net especializado en harmonización de iluminación y color.
    Toma una imagen compuesta (persona + fondo) y la armoniza para verse natural.
    """

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetHarmonizer, self).__init__()
        self.use_attention = use_attention

        # Import de clases necesarias desde main
        from models import AttentionBlock, DoubleConv

        # Encoder usando VGG16 pre-entrenado para mejor extracción de características de color
        if pretrained:
            try:
                vgg = vgg16(pretrained=True).features
                self.encoder_layers = []

                # Extraer layers específicos de VGG16
                self.conv1 = nn.Sequential(*list(vgg.children())[:4])  # 64 channels
                self.conv2 = nn.Sequential(*list(vgg.children())[4:9])  # 128 channels
                self.conv3 = nn.Sequential(*list(vgg.children())[9:16])  # 256 channels
                self.conv4 = nn.Sequential(*list(vgg.children())[16:23])  # 512 channels
                self.conv5 = nn.Sequential(*list(vgg.children())[23:30])  # 512 channels
            except:
                # Fallback si no se puede cargar VGG pre-entrenado
                pretrained = False

        if not pretrained:
            # Encoder básico si no se usa pre-entrenado
            self.conv1 = self._make_layer(3, 64)
            self.conv2 = self._make_layer(64, 128)
            self.conv3 = self._make_layer(128, 256)
            self.conv4 = self._make_layer(256, 512)
            self.conv5 = self._make_layer(512, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout_rate=0.2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Attention gates para harmonización
        if self.use_attention:
            self.att1 = AttentionBlock(512, 512, 256)
            self.att2 = AttentionBlock(256, 256, 128)
            self.att3 = AttentionBlock(128, 128, 64)
            self.att4 = AttentionBlock(64, 64, 32)
            self.att5 = AttentionBlock(32, 32, 16)

        # Convoluciones del decoder
        self.dec_conv1 = DoubleConv(1024, 512)
        self.dec_conv2 = DoubleConv(512, 256)
        self.dec_conv3 = DoubleConv(256, 128)
        self.dec_conv4 = DoubleConv(128, 64)
        self.dec_conv5 = DoubleConv(64, 32)

        # Capa final - 3 canales RGB armonizados
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def _make_layer(self, in_channels, out_channels):
        """Crea una capa de convolución básica."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _match_tensor_size(self, x, target_tensor):
        """Ajusta el tamaño de x para que coincida con target_tensor."""
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # Encoder path con skip connections
        skip_connections = []

        # Encoder
        x1 = self.conv1(x)  # 64 channels
        skip_connections.append(x1)
        x1_pool = self.pool(x1)

        x2 = self.conv2(x1_pool)  # 128 channels
        skip_connections.append(x2)
        x2_pool = self.pool(x2)

        x3 = self.conv3(x2_pool)  # 256 channels
        skip_connections.append(x3)
        x3_pool = self.pool(x3)

        x4 = self.conv4(x3_pool)  # 512 channels
        skip_connections.append(x4)
        x4_pool = self.pool(x4)

        x5 = self.conv5(x4_pool)  # 512 channels
        skip_connections.append(x5)
        x5_pool = self.pool(x5)

        # Bottleneck
        bottleneck = self.bottleneck(x5_pool)

        # Decoder path
        skips = skip_connections[::-1]  # Invertir orden

        # Up 1
        up1 = self.up1(bottleneck)
        skip = self._match_tensor_size(skips[0], up1)
        if self.use_attention:
            skip = self.att1(skip, up1)
        up1 = torch.cat([up1, skip], dim=1)
        up1 = self.dec_conv1(up1)

        # Up 2
        up2 = self.up2(up1)
        skip = self._match_tensor_size(skips[1], up2)
        if self.use_attention:
            skip = self.att2(skip, up2)
        up2 = torch.cat([up2, skip], dim=1)
        up2 = self.dec_conv2(up2)

        # Up 3
        up3 = self.up3(up2)
        skip = self._match_tensor_size(skips[2], up3)
        if self.use_attention:
            skip = self.att3(skip, up3)
        up3 = torch.cat([up3, skip], dim=1)
        up3 = self.dec_conv3(up3)

        # Up 4
        up4 = self.up4(up3)
        skip = self._match_tensor_size(skips[3], up4)
        if self.use_attention:
            skip = self.att4(skip, up4)
        up4 = torch.cat([up4, skip], dim=1)
        up4 = self.dec_conv4(up4)

        # Up 5 (final)
        up5 = self.up5(up4)
        skip = self._match_tensor_size(skips[4], up5)
        if self.use_attention:
            skip = self.att5(skip, up5)
        up5 = torch.cat([up5, skip], dim=1)
        up5 = self.dec_conv5(up5)

        # Output final
        output = self.final_conv(up5)

        # Aplicar tanh para rango [-1, 1] y luego sigmoid para [0, 1]
        output = torch.tanh(output)
        output = torch.sigmoid(output)

        return output


class HarmonizationDataset(Dataset):
    """
    Dataset para entrenamiento de harmonización.
    Genera pares sintéticos de imágenes compuestas vs armonizadas.
    """

    def __init__(self, foreground_dir, background_dir, transform=None, image_size=384):
        """
        Args:
            foreground_dir: Directorio con imágenes RGBA de personas
            background_dir: Directorio con imágenes RGB de fondos
            transform: Transformaciones de albumentations
            image_size: Tamaño de las imágenes
        """
        self.foreground_dir = foreground_dir
        self.background_dir = background_dir
        self.transform = transform
        self.image_size = image_size

        # Import ImageProcessor desde main
        from utils import ImageProcessor
        self.processor = ImageProcessor()

        # Obtener listas de archivos
        self.foreground_files = [f for f in os.listdir(foreground_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.background_files = [f for f in os.listdir(background_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(self.foreground_files) == 0 or len(self.background_files) == 0:
            raise ValueError("Directorios de foreground o background vacíos")

    def __len__(self):
        return len(self.foreground_files) * 2  # Múltiples combinaciones por foreground

    def __getitem__(self, idx):
        # Seleccionar foreground
        fg_idx = idx % len(self.foreground_files)
        foreground_path = os.path.join(self.foreground_dir, self.foreground_files[fg_idx])

        # Seleccionar background aleatorio
        bg_idx = np.random.randint(0, len(self.background_files))
        background_path = os.path.join(self.background_dir, self.background_files[bg_idx])

        try:
            # Cargar imágenes
            foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
            background = cv2.imread(background_path, cv2.IMREAD_COLOR)

            if foreground is None or background is None:
                # Fallback a imagen dummy
                return self._get_dummy_sample()

            # Convertir BGR a RGB
            if foreground.shape[2] == 4:  # RGBA
                foreground = cv2.cvtColor(foreground, cv2.COLOR_BGRA2RGBA)
            else:  # RGB
                foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
                # Añadir canal alpha (asumiendo que toda la imagen es foreground)
                alpha = np.ones((foreground.shape[0], foreground.shape[1], 1), dtype=np.uint8) * 255
                foreground = np.concatenate([foreground, alpha], axis=2)

            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

            # Redimensionar
            foreground = cv2.resize(foreground, (self.image_size, self.image_size))
            background = cv2.resize(background, (self.image_size, self.image_size))

            # Crear composición
            composite = self.processor.composite_foreground_background(foreground, background)

            # El target es la imagen original del foreground (sin fondo)
            # Para simplificar, usamos la composición como target también
            # En un dataset real, tendrías la imagen armonizada real
            target = composite.copy()

            # Normalizar
            composite = composite.astype(np.float32) / 255.0
            target = target.astype(np.float32) / 255.0

            # Aplicar transformaciones
            if self.transform:
                try:
                    composite_uint8 = (composite * 255).astype(np.uint8)
                    target_uint8 = (target * 255).astype(np.uint8)

                    augmented = self.transform(image=composite_uint8, mask=target_uint8)
                    composite = augmented['image'].astype(np.float32) / 255.0
                    target = augmented['mask'].astype(np.float32) / 255.0
                except Exception as e:
                    print(f"Error en transformación: {e}")
                    pass

            # Convertir a tensores
            composite_tensor = torch.FloatTensor(composite.transpose(2, 0, 1))
            target_tensor = torch.FloatTensor(target.transpose(2, 0, 1))

            return composite_tensor, target_tensor

        except Exception as e:
            print(f"Error procesando {foreground_path}: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Retorna una muestra dummy en caso de error."""
        dummy_image = torch.zeros(3, self.image_size, self.image_size)
        dummy_target = torch.zeros(3, self.image_size, self.image_size)
        return dummy_image, dummy_target


class HarmonizationLossCalculator:
    """
    Calculadora de pérdidas específica para harmonización.
    Combina pérdidas perceptuales, de color y de consistencia.
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, delta=0.2):
        self.alpha = alpha  # MSE weight
        self.beta = beta  # Perceptual weight
        self.gamma = gamma  # Color consistency weight
        self.delta = delta  # Style weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # VGG para pérdida perceptual
        try:
            vgg = vgg16(pretrained=True).features
            self.vgg_layers = vgg[:16]  # Hasta conv3_3
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
        except:
            self.vgg_layers = None

    def perceptual_loss(self, pred, target):
        """Calcula pérdida perceptual usando características VGG."""
        if self.vgg_layers is None:
            return torch.tensor(0.0, device=pred.device)

        # Mover VGG a mismo dispositivo
        if pred.is_cuda and not next(self.vgg_layers.parameters()).is_cuda:
            self.vgg_layers = self.vgg_layers.cuda()

        # Extraer características
        pred_features = self.vgg_layers(pred)
        target_features = self.vgg_layers(target)

        return self.mse_loss(pred_features, target_features)

    def color_consistency_loss(self, pred, target):
        """Calcula pérdida de consistencia de color en diferentes espacios."""
        # Pérdida en espacio LAB
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)

        # Separar luminancia y cromaticidad
        pred_l, pred_ab = pred_lab[:, :1], pred_lab[:, 1:]
        target_l, target_ab = target_lab[:, :1], target_lab[:, 1:]

        # Pérdida más fuerte en cromaticidad
        l_loss = self.l1_loss(pred_l, target_l)
        ab_loss = self.l1_loss(pred_ab, target_ab)

        return l_loss + 2.0 * ab_loss

    def rgb_to_lab(self, rgb):
        """Conversión aproximada RGB a LAB usando operaciones tensoriales."""
        # Normalizar RGB a [0, 1]
        rgb = torch.clamp(rgb, 0, 1)

        # Conversión simplificada RGB -> LAB
        # Esta es una aproximación, no la conversión exacta
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # Luminancia aproximada
        l = 0.299 * r + 0.587 * g + 0.114 * b

        # Componentes a y b aproximados
        a = 0.5 * (r - g) + 0.5
        b_comp = 0.5 * (0.5 * (r + g) - b) + 0.5

        return torch.cat([l, a, b_comp], dim=1)

    def style_loss(self, pred, target):
        """Calcula pérdida de estilo usando matriz de Gram."""

        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram / (c * h * w)

        pred_gram = gram_matrix(pred)
        target_gram = gram_matrix(target)

        return self.mse_loss(pred_gram, target_gram)

    def calculate_loss(self, pred, target):
        """Calcula la pérdida total compuesta para harmonización."""
        # Pérdida MSE básica
        mse_loss = self.mse_loss(pred, target)

        # Pérdida perceptual
        perceptual_loss = self.perceptual_loss(pred, target)

        # Pérdida de consistencia de color
        color_loss = self.color_consistency_loss(pred, target)

        # Pérdida de estilo
        style_loss = self.style_loss(pred, target)

        # Combinar pérdidas
        total_loss = (self.alpha * mse_loss +
                      self.beta * perceptual_loss +
                      self.gamma * color_loss +
                      self.delta * style_loss)

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'perceptual_loss': perceptual_loss,
            'color_loss': color_loss,
            'style_loss': style_loss
        }


class HarmonizationTrainer:
    """
    Clase para entrenar el modelo de harmonización.
    """

    def __init__(self, model, train_loader, val_loader, device, config, rank=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.rank = rank

        # Inicializar componentes
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.loss_calculator = HarmonizationLossCalculator()

        # Import ModelCheckpoint desde main
        from utils import ModelCheckpoint
        self.checkpoint_manager = ModelCheckpoint()

        # Historial de entrenamiento
        self.train_history = {'loss': [], 'mse': [], 'perceptual': []}
        self.val_history = {'loss': [], 'mse': [], 'perceptual': []}

        # Logger
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configura el logger para harmonización."""
        logger = logging.getLogger('harmonization')
        if self.rank == 0:
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'logs/harmonization_{timestamp}.log'
            logger.setLevel(logging.INFO)

            if not logger.handlers:
                handler = logging.FileHandler(log_filename)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        else:
            logger.setLevel(logging.CRITICAL) # Solo el rank 0 logea

        return logger

    def train_epoch(self):
        """Entrena una época."""
        self.model.train()
        self.train_loader.sampler.set_epoch(self.epoch) # Sincronizar shuffling
        epoch_losses = []
        epoch_mse = []
        epoch_perceptual = []

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            try:
                images = images.to(self.device)
                targets = targets.to(self.device)

                if torch.isnan(images).any() or torch.isnan(targets).any():
                    if self.rank == 0: self.logger.warning(f"NaN detectado en batch {batch_idx}, saltando...")
                    continue

                self.optimizer.zero_grad()
                outputs = self.model(images)

                if torch.isnan(outputs).any():
                    if self.rank == 0: self.logger.warning(f"NaN en outputs del batch {batch_idx}, saltando...")
                    continue

                loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                total_loss = loss_dict['total_loss']

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if self.rank == 0: self.logger.warning(f"Pérdida inválida en batch {batch_idx}: {total_loss.item()}, saltando...")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_losses.append(total_loss.item())
                epoch_mse.append(loss_dict['mse_loss'].item())
                epoch_perceptual.append(loss_dict['perceptual_loss'].item())

                if self.rank == 0 and batch_idx % 20 == 0:
                    self.logger.info(f'Batch {batch_idx}/{len(self.train_loader)}: '
                                     f'Loss: {total_loss.item():.4f}, MSE: {loss_dict["mse_loss"].item():.4f}')

            except Exception as e:
                if self.rank == 0: self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if len(epoch_losses) > 0:
            return np.mean(epoch_losses), np.mean(epoch_mse), np.mean(epoch_perceptual)
        else:
            return 0.0, 0.0, 0.0

    def validate_epoch(self):
        """Valida una época."""
        self.model.eval()
        epoch_losses = []
        epoch_mse = []
        epoch_perceptual = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                try:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    if torch.isnan(images).any() or torch.isnan(targets).any():
                        continue

                    outputs = self.model(images)

                    if torch.isnan(outputs).any():
                        continue

                    loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                    total_loss = loss_dict['total_loss']

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    epoch_losses.append(total_loss.item())
                    epoch_mse.append(loss_dict['mse_loss'].item())
                    epoch_perceptual.append(loss_dict['perceptual_loss'].item())

                except Exception as e:
                    if self.rank == 0: self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue

        if len(epoch_losses) > 0:
            return np.mean(epoch_losses), np.mean(epoch_mse), np.mean(epoch_perceptual)
        else:
            return 0.0, 0.0, 0.0

    def train(self, num_epochs):
        """Entrenamiento principal."""
        if self.rank == 0:
            self.logger.info("Iniciando entrenamiento de harmonización distribuido...")
            self.logger.info(f"Configuración: {self.config}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            if self.rank == 0: self.logger.info(f"\nÉpoca {epoch + 1}/{num_epochs}")

            train_loss, train_mse, train_perceptual = self.train_epoch()
            val_loss, val_mse, val_perceptual = self.validate_epoch()
            self.scheduler.step()

            if not (np.isnan(train_loss) or np.isnan(val_loss)):
                if self.rank == 0:
                    self.train_history['loss'].append(train_loss)
                    self.train_history['mse'].append(train_mse)
                    self.train_history['perceptual'].append(train_perceptual)

                    self.val_history['loss'].append(val_loss)
                    self.val_history['mse'].append(val_mse)
                    self.val_history['perceptual'].append(val_perceptual)

                    self.logger.info(f"Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}")
                    self.logger.info(f"Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}")

                    is_best = val_loss < self.checkpoint_manager.best_loss
                    if is_best:
                        self.checkpoint_manager.best_loss = val_loss

                    metrics = {
                        'train_loss': train_loss, 'train_mse': train_mse,
                        'val_loss': val_loss, 'val_mse': val_mse
                    }
                    
                    # Al guardar, se guarda el state_dict del modelo subyacente
                    self.checkpoint_manager.save_checkpoint(
                        self.model.module, self.optimizer, epoch, val_loss, metrics, is_best, 'harmonizer'
                    )

                    if is_best:
                        self.logger.info(f"¡Nuevo mejor modelo de harmonización! Loss: {val_loss:.4f}")
            else:
                if self.rank == 0: self.logger.warning(f"Época {epoch + 1} saltada debido a valores NaN")

        if self.rank == 0:
            self.logger.info("Entrenamiento de harmonización completado!")
            self.save_training_plots()

    def save_training_plots(self):
        """Guarda gráficas del entrenamiento."""
        if len(self.train_history['loss']) == 0:
            self.logger.warning("No hay datos de entrenamiento para graficar")
            return

        os.makedirs('plots', exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.train_history['loss'], label='Train Loss')
        axes[0].plot(self.val_history['loss'], label='Val Loss')
        axes[0].set_title('Harmonization Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(self.train_history['mse'], label='Train MSE')
        axes[1].plot(self.val_history['mse'], label='Val MSE')
        axes[1].set_title('MSE Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        axes[1].grid(True)
        axes[2].plot(self.train_history['perceptual'], label='Train Perceptual')
        axes[2].plot(self.val_history['perceptual'], label='Val Perceptual')
        axes[2].set_title('Perceptual Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Perceptual Loss')
        axes[2].legend()
        axes[2].grid(True)
        plt.tight_layout()
        plt.savefig('plots/harmonization_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def get_harmonization_transforms():
    """Define transformaciones específicas para harmonización."""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
    ], additional_targets={'mask': 'mask'})

    return train_transform, val_transform


def create_sample_harmonization_dataset():
    """
    Crea un dataset de muestra para harmonización usando imágenes COCO.
    """
    print("=== CREANDO DATASET DE MUESTRA PARA HARMONIZACIÓN ===\n")
    foreground_dir = 'dataset/foregrounds'
    background_dir = 'dataset/backgrounds'
    coco_root = 'COCO'
    os.makedirs(foreground_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)
    try:
        existing_fg = len([f for f in os.listdir(foreground_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        existing_bg = len([f for f in os.listdir(background_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"📊 Estado actual:")
        print(f"   - Foregrounds existentes: {existing_fg}")
        print(f"   - Backgrounds existentes: {existing_bg}")
        if existing_fg >= 10 and existing_bg >= 10:
            print("✅ Ya existen suficientes imágenes para harmonización")
            return True
        if os.path.exists(os.path.join(coco_root, 'val2017')):
            print("🎯 Generando imágenes de muestra desde COCO...")
            coco_images = [f for f in os.listdir(os.path.join(coco_root, 'val2017'))
                           if f.endswith('.jpg')][:20]
            created_bg = 0
            for img_file in coco_images:
                if created_bg >= 10:
                    break
                src_path = os.path.join(coco_root, 'val2017', img_file)
                dst_path = os.path.join(background_dir, f'bg_{created_bg:03d}.jpg')
                try:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    created_bg += 1
                except Exception as e:
                    print(f"Error copiando {img_file}: {e}")
            print(f"   ✅ Creados {created_bg} backgrounds desde COCO")
        if existing_fg < 10:
            print("🎨 Generando foregrounds sintéticos...")
            for i in range(10 - existing_fg):
                fg_image = np.random.randint(50, 200, (384, 384, 3), dtype=np.uint8)
                center = (192, 192)
                radius = 100
                y, x = np.ogrid[:384, :384]
                mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
                alpha = np.zeros((384, 384), dtype=np.uint8)
                alpha[mask] = 255
                rgba_image = np.dstack([fg_image, alpha])
                fg_path = os.path.join(foreground_dir, f'fg_synthetic_{i:03d}.png')
                cv2.imwrite(fg_path, rgba_image)
            print(f"   ✅ Creados {10 - existing_fg} foregrounds sintéticos")
        if existing_bg < 10:
            print("🖼️ Generando backgrounds sintéticos...")
            for i in range(10 - existing_bg):
                bg_image = np.zeros((384, 384, 3), dtype=np.uint8)
                color1 = np.random.randint(0, 255, 3)
                color2 = np.random.randint(0, 255, 3)
                for j in range(384):
                    ratio = j / 384.0
                    bg_image[j, :] = color1 * (1 - ratio) + color2 * ratio
                noise = np.random.randint(-20, 20, (384, 384, 3))
                bg_image = np.clip(bg_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                bg_path = os.path.join(background_dir, f'bg_synthetic_{i:03d}.jpg')
                cv2.imwrite(bg_path, cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR))
            print(f"   ✅ Creados {10 - existing_bg} backgrounds sintéticos")
        final_fg = len([f for f in os.listdir(foreground_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        final_bg = len([f for f in os.listdir(background_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"\n📈 Resultado final:")
        print(f"   - Total foregrounds: {final_fg}")
        print(f"   - Total backgrounds: {final_bg}")
        if final_fg >= 5 and final_bg >= 5:
            print("✅ Dataset de harmonización listo")
            return True
        else:
            print("❌ No se pudo crear dataset suficiente")
            return False
    except Exception as e:
        print(f"❌ Error creando dataset de harmonización: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_harmonization_dataset():
    """
    Función de prueba para verificar el dataset de harmonización.
    """
    print("Probando dataset de harmonización...")
    temp_fg_dir = 'temp_foregrounds'
    temp_bg_dir = 'temp_backgrounds'
    os.makedirs(temp_fg_dir, exist_ok=True)
    os.makedirs(temp_bg_dir, exist_ok=True)
    try:
        fg_image = np.random.randint(0, 256, (384, 384, 4), dtype=np.uint8)
        cv2.imwrite(os.path.join(temp_fg_dir, 'test_fg.png'), fg_image)
        bg_image = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(temp_bg_dir, 'test_bg.jpg'), bg_image)
        dataset = HarmonizationDataset(
            foreground_dir=temp_fg_dir,
            background_dir=temp_bg_dir,
            transform=None,
            image_size=384
        )
        print(f"✓ Dataset de harmonización creado exitosamente")
        print(f"  Total de combinaciones: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            composite, target = sample
            print(f"  ✓ Muestra de harmonización cargada exitosamente")
            print(f"    Composite shape: {composite.shape}")
            print(f"    Target shape: {target.shape}")
            os.remove(os.path.join(temp_fg_dir, 'test_fg.png'))
            os.remove(os.path.join(temp_bg_dir, 'test_bg.jpg'))
            os.rmdir(temp_fg_dir)
            os.rmdir(temp_bg_dir)
            return True
        else:
            print("✗ Dataset de harmonización vacío")
            return False
    except Exception as e:
        print(f"✗ Error creando dataset de harmonización: {e}")
        try:
            os.remove(os.path.join(temp_fg_dir, 'test_fg.png'))
            os.remove(os.path.join(temp_bg_dir, 'test_bg.jpg'))
            os.rmdir(temp_fg_dir)
            os.rmdir(temp_bg_dir)
        except:
            pass
        return False


def test_harmonizer_forward():
    """
    Función de prueba para verificar el modelo de harmonización.
    """
    print("Probando forward pass del modelo de harmonización...")
    harmonizer = UNetHarmonizer(pretrained=False, use_attention=True)
    harmonizer.eval()
    test_input = torch.randn(1, 3, 384, 384)
    try:
        with torch.no_grad():
            harmonized_output = harmonizer(test_input)
        print(f"✓ Forward pass de harmonización exitoso!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {harmonized_output.shape}")
        print(f"  Expected output shape: (1, 3, 384, 384)")
        if harmonized_output.shape == (1, 3, 384, 384):
            print("✓ Dimensiones de salida de harmonización correctas")
            return True
        else:
            print("✗ Dimensiones de salida de harmonización incorrectas")
            return False
    except Exception as e:
        print(f"✗ Error en forward pass de harmonización: {e}")
        return False


def train_harmonization_model(config=None):
    """
    Función principal para entrenar el modelo de harmonización.
    """
    # --- INICIO: Configuración para DDP ---
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device_id = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device_id)
    # --- FIN: Configuración para DDP ---

    if config is None:
        config = {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'weight_decay': 1e-6,
            'num_epochs': 50,
            'image_size': 384,
            'num_workers': 4,
            'pin_memory': True,
        }
    
    config['device'] = f'cuda:{device_id}'

    logger = logging.getLogger('harmonization_main')
    if rank == 0:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.info("Iniciando entrenamiento de modelo de harmonización")
        logger.info(f"Dispositivo: {config['device']}, World Size: {world_size}")

    foreground_dir = 'dataset/foregrounds'
    background_dir = 'dataset/backgrounds'
    os.makedirs(foreground_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    if len(os.listdir(foreground_dir)) == 0 and rank == 0:
        logger.warning(f"Directorio de foregrounds vacío: {foreground_dir}")
        return False
    if len(os.listdir(background_dir)) == 0 and rank == 0:
        logger.warning(f"Directorio de backgrounds vacío: {background_dir}")
        return False

    train_transform, val_transform = get_harmonization_transforms()

    if rank == 0: logger.info("Creando dataset de harmonización...")
    dataset = HarmonizationDataset(
        foreground_dir=foreground_dir,
        background_dir=background_dir,
        transform=train_transform,
        image_size=config['image_size']
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # --- INICIO: Cambios para DDP DataLoader ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # shuffle es False porque el Sampler se encarga
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,
        sampler=val_sampler
    )
    # --- FIN: Cambios para DDP DataLoader ---

    if rank == 0: logger.info(f"Dataset de harmonización cargado: {len(train_dataset)} train, {len(val_dataset)} val")

    if rank == 0: logger.info("Inicializando modelo U-Net Harmonizer...")
    harmonizer = UNetHarmonizer(pretrained=True, use_attention=True).to(config['device'])
    
    # --- INICIO: Envolver modelo con DDP ---
    harmonizer = DDP(harmonizer, device_ids=[device_id])
    # --- FIN: Envolver modelo con DDP ---

    if rank == 0:
        total_params = sum(p.numel() for p in harmonizer.parameters())
        trainable_params = sum(p.numel() for p in harmonizer.parameters() if p.requires_grad)
        logger.info(f"Parámetros totales del harmonizer: {total_params:,}")
        logger.info(f"Parámetros entrenables del harmonizer: {trainable_params:,}")

    harmonization_trainer = HarmonizationTrainer(
        model=harmonizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        config=config,
        rank=rank
    )

    harmonization_trainer.train(config['num_epochs'])

    if rank == 0: logger.info("Entrenamiento de harmonización completado exitosamente!")
    
    dist.destroy_process_group()
    return True


# Clase para inferencia de harmonización (será importada desde main.py)
class HarmonizationInference:
    """
    Clase específica para inferencia de harmonización.
    """

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = UNetHarmonizer(pretrained=False, use_attention=True)

        # Cargar modelo entrenado
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Cargar el state_dict del modelo subyacente si fue guardado desde DDP
            state_dict = checkpoint['model_state_dict']
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

            self.model.to(device)
            self.model.eval()
            print(f"Modelo de harmonización cargado: {model_path}")
        else:
            print(f"Modelo de harmonización no encontrado: {model_path}")
            self.model = None

    def harmonize_composition(self, composite_rgb, output_path=None):
        """
        Armoniza una composición RGB.

        Args:
            composite_rgb: Imagen RGB compuesta (H, W, 3)
            output_path: Ruta para guardar resultado (opcional)

        Returns:
            Imagen RGB armonizada (H, W, 3)
        """
        if self.model is None:
            print("Modelo de harmonización no disponible.")
            return composite_rgb

        original_size = composite_rgb.shape[:2]

        # Redimensionar para el modelo
        composite_resized = cv2.resize(composite_rgb, (384, 384))

        # Normalizar y convertir a tensor
        composite_normalized = composite_resized.astype(np.float32) / 255.0
        input_tensor = torch.FloatTensor(composite_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # Harmonización
        with torch.no_grad():
            harmonized = self.model(input_tensor)
            harmonized = harmonized.squeeze(0).cpu().numpy()

        # Post-procesamiento
        harmonized = harmonized.transpose(1, 2, 0)
        harmonized = (harmonized * 255).astype(np.uint8)

        # Restaurar tamaño original
        harmonized = cv2.resize(harmonized, (original_size[1], original_size[0]))

        # Guardar si se especifica path
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(harmonized, cv2.COLOR_RGB2BGR))

        return harmonized


if __name__ == "__main__":
    # --- INICIO: Llamada a la función de entrenamiento ---
    # Comprobamos si el script se está ejecutando con torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        train_harmonization_model()
    else:
        # --- FIN: Llamada a la función de entrenamiento ---
        # Pruebas del módulo de harmonización
        print("=== PRUEBAS DEL MÓDULO DE HARMONIZACIÓN ===\n")

        tests = [
            ("Forward pass del harmonizer", test_harmonizer_forward),
            ("Dataset de harmonización", test_harmonization_dataset),
        ]

        results = {}

        for test_name, test_func in tests:
            print(f"📋 {test_name}...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ Error en {test_name}: {e}")
                results[test_name] = False

        # Resumen
        print(f"\n" + "=" * 50)
        print("📋 RESUMEN DE PRUEBAS DE HARMONIZACIÓN:")

        all_passed = True
        for test_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n🎉 ¡TODAS LAS PRUEBAS DE HARMONIZACIÓN EXITOSAS!")
            print(f"🎨 El módulo de harmonización está listo para usar")
        else:
            print(f"\n⚠️  Algunas pruebas fallaron")
            print(f"🔧 Revisa los errores antes de continuar")