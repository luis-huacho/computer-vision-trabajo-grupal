import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class Trainer:
    """
    Clase principal para el entrenamiento del modelo de segmentación.
    """

    def __init__(self, model, train_loader, val_loader, device, config, rank=0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.rank = rank

        # Inicializar componentes
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        # Import desde utils
        try:
            from utils import LossCalculator, MetricsCalculator, ModelCheckpoint
            self.loss_calculator = LossCalculator()
            self.metrics_calculator = MetricsCalculator()
            self.checkpoint_manager = ModelCheckpoint()
        except ImportError:
            print("⚠️  Componentes de utils no disponibles")
            self.loss_calculator = None
            self.metrics_calculator = None
            self.checkpoint_manager = None

        # Historial de entrenamiento
        self.train_history = {'loss': [], 'iou': [], 'dice': []}
        self.val_history = {'loss': [], 'iou': [], 'dice': []}

        # Logger
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configura el logger para el entrenamiento."""
        logger = logging.getLogger('segmentation')
        if self.rank == 0:
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'logs/segmentation_{timestamp}.log'
            logger.setLevel(logging.INFO)

            # Evitar duplicar handlers
            if not logger.handlers:
                handler = logging.FileHandler(log_filename)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

                # También log a consola
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        else:
            logger.setLevel(logging.CRITICAL)  # Solo el rank 0 logea

        return logger

    def train_epoch(self):
        """Entrena una época."""
        self.model.train()
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)  # Sincronizar shuffling
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
                if self.loss_calculator:
                    loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                    total_loss = loss_dict['total_loss']
                else:
                    # Fallback simple
                    total_loss = torch.nn.functional.mse_loss(outputs, targets)

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

                    if self.metrics_calculator:
                        iou = self.metrics_calculator.calculate_iou(pred_alpha, target_alpha)
                        dice = self.metrics_calculator.calculate_dice(pred_alpha, target_alpha)
                    else:
                        # Fallback simple
                        iou = torch.tensor(0.5)
                        dice = torch.tensor(0.5)

                    # Verificar métricas válidas
                    if not (torch.isnan(iou) or torch.isnan(dice)):
                        epoch_losses.append(total_loss.item())
                        epoch_ious.append(iou.item())
                        epoch_dices.append(dice.item())

                # Log cada N batches (solo rank 0)
                if self.rank == 0 and batch_idx % 10 == 0 and len(epoch_losses) > 0:
                    self.logger.info(f'Batch {batch_idx}/{len(self.train_loader)}: '
                                     f'Loss: {total_loss.item():.4f}, IoU: {iou.item():.4f}, Dice: {dice.item():.4f}')

            except Exception as e:
                if self.rank == 0:
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
                    if self.loss_calculator:
                        loss_dict = self.loss_calculator.calculate_loss(outputs, targets)
                        total_loss = loss_dict['total_loss']
                    else:
                        total_loss = torch.nn.functional.mse_loss(outputs, targets)

                    # Verificar pérdida válida
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    # Calcular métricas
                    pred_alpha = outputs[:, 3:4]
                    target_alpha = targets[:, 3:4]

                    if self.metrics_calculator:
                        iou = self.metrics_calculator.calculate_iou(pred_alpha, target_alpha)
                        dice = self.metrics_calculator.calculate_dice(pred_alpha, target_alpha)
                    else:
                        iou = torch.tensor(0.5)
                        dice = torch.tensor(0.5)

                    # Solo agregar si las métricas son válidas
                    if not (torch.isnan(iou) or torch.isnan(dice)):
                        epoch_losses.append(total_loss.item())
                        epoch_ious.append(iou.item())
                        epoch_dices.append(dice.item())

                except Exception as e:
                    if self.rank == 0:
                        self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue

        # Retornar promedios válidos
        if len(epoch_losses) > 0:
            return np.mean(epoch_losses), np.mean(epoch_ious), np.mean(epoch_dices)
        else:
            return 0.0, 0.0, 0.0

    def train(self, num_epochs):
        """Entrenamiento principal."""
        if self.rank == 0:
            self.logger.info("Iniciando entrenamiento distribuido...")
            self.logger.info(f"Configuración: {self.config}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            if self.rank == 0:
                self.logger.info(f"\nÉpoca {epoch + 1}/{num_epochs}")

            # Entrenar
            train_loss, train_iou, train_dice = self.train_epoch()

            # Validar
            val_loss, val_iou, val_dice = self.validate_epoch()

            # Actualizar scheduler
            self.scheduler.step()

            # Verificar si los valores son válidos antes de guardar
            if not (np.isnan(train_loss) or np.isnan(val_loss)):
                if self.rank == 0:
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

                    # Guardar checkpoint (solo rank 0)
                    if self.checkpoint_manager:
                        is_best = val_iou > self.checkpoint_manager.best_iou
                        if is_best:
                            self.checkpoint_manager.best_iou = val_iou
                            self.checkpoint_manager.best_loss = val_loss

                        metrics = {
                            'train_loss': train_loss, 'train_iou': train_iou, 'train_dice': train_dice,
                            'val_loss': val_loss, 'val_iou': val_iou, 'val_dice': val_dice
                        }

                        # Guardar el state_dict del modelo subyacente si es DDP
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        self.checkpoint_manager.save_checkpoint(
                            model_to_save, self.optimizer, epoch, val_loss, metrics, is_best, 'segmentation'
                        )

                        if is_best:
                            self.logger.info(f"¡Nuevo mejor modelo! IoU: {val_iou:.4f}")
            else:
                if self.rank == 0:
                    self.logger.warning(f"Época {epoch + 1} saltada debido a valores NaN")

        if self.rank == 0:
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
        plt.savefig('plots/segmentation_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def train_segmentation(config=None):
    """
    Función principal para entrenar el modelo de segmentación.
    Soporta entrenamiento distribuido con DDP.
    """
    # --- INICIO: Configuración para DDP ---
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_id = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(device_id)
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        device_id = 0
        is_distributed = False
    # --- FIN: Configuración para DDP ---

    # Configuración por defecto si no se proporciona
    if config is None:
        config = {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'num_epochs': 100,
            'image_size': 384,
            'device': f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu',
            'num_workers': 8,
            'pin_memory': True,
        }
    else:
        config['device'] = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

    # Setup logging (solo rank 0)
    logger = logging.getLogger('segmentation_main')
    if rank == 0:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - [RANK %(rank)s] - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        # Imprimir configuración de forma legible
        logger.info("="*50)
        logger.info("Iniciando entrenamiento de modelo de segmentación")
        logger.info(f"Dispositivo: {config['device']}, World Size: {world_size}")
        logger.info("Configuración de entrenamiento:")
        for key, value in config.items():
            logger.info(f"  - {key}: {value}")
        logger.info("="*50)

    else:
        logger.setLevel(logging.CRITICAL)

    # Verificar dataset COCO
    coco_root = 'COCO'
    if not os.path.exists(coco_root):
        logger.error(f"Directorio COCO no encontrado: {coco_root}")
        return False

    # Verificar archivos de anotaciones
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
    val_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_val2017.json')

    if not os.path.exists(train_ann_file):
        logger.error(f"Archivo de anotaciones de entrenamiento no encontrado: {train_ann_file}")
        return False

    if not os.path.exists(val_ann_file):
        logger.error(f"Archivo de anotaciones de validación no encontrado: {val_ann_file}")
        return False

    logger.info("Estructura COCO verificada correctamente")

    # Preparar transforms
    try:
        from datasets import get_transforms
        train_transform, val_transform = get_transforms()
    except ImportError:
        logger.warning("Módulo datasets no disponible, usando transforms básicos")
        train_transform = None
        val_transform = None

    # Crear datasets COCO
    try:
        from datasets import COCOPersonDataset

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
    except ImportError:
        logger.error("Módulo datasets no disponible")
        return False

    # --- INICIO: Verificación de dataset vacío ---
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        if rank == 0:
            logger.error("Uno o ambos datasets están vacíos. Abortando entrenamiento.")
            logger.error(f"Tamaños: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # Sincronizar procesos antes de salir para evitar que otros continúen
        if is_distributed:
            dist.barrier()
        
        return False
    # --- FIN: Verificación de dataset vacío ---

    # --- INICIO: Crear data loaders con soporte DDP ---
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        shuffle = False  # DistributedSampler maneja el shuffle
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
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
    # --- FIN: Crear data loaders con soporte DDP ---

    if rank == 0:
        logger.info(f"Dataset COCO cargado: {len(train_dataset)} train, {len(val_dataset)} val")
        logger.info(f"DataLoaders listos: {len(train_loader)} batches de entrenamiento, {len(val_loader)} batches de validación")

    # Crear modelo
    try:
        from models import UNetAutoencoder

        if rank == 0:
            logger.info("Inicializando modelo U-Net Autoencoder...")
        model = UNetAutoencoder(pretrained=True, use_attention=True).to(config['device'])
    except ImportError:
        if rank == 0:
            logger.error("Módulo models no disponible")
        return False

    # --- INICIO: Envolver modelo con DDP ---
    if is_distributed:
        model = DDP(model, device_ids=[device_id])
    # --- FIN: Envolver modelo con DDP ---

    # Contar parámetros (solo rank 0)
    if rank == 0:
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
        config=config,
        rank=rank
    )

    # Entrenar modelo
    trainer.train(config['num_epochs'])

    if rank == 0:
        logger.info("Entrenamiento completado exitosamente!")
    
    # --- INICIO: Cleanup DDP ---
    if is_distributed:
        dist.destroy_process_group()
    # --- FIN: Cleanup DDP ---
    
    return True


if __name__ == "__main__":
    # Comprobamos si el script se está ejecutando con torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Ejecutar entrenamiento distribuido
        train_segmentation()
    else:
        # Prueba del módulo
        print("=== PRUEBA DEL MÓDULO DE ENTRENAMIENTO ===")

        # Configuración de prueba
        test_config = {
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'num_epochs': 2,
            'image_size': 256,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 0,
            'pin_memory': False,
        }

        print("Iniciando entrenamiento de prueba...")
        success = train_segmentation(test_config)

        if success:
            print("✅ Prueba de entrenamiento completada exitosamente!")
        else:
            print("❌ Error en la prueba de entrenamiento")