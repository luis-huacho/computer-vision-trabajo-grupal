#!/usr/bin/env python3
"""
Trainer - Entrenamiento de Segmentación con Soporte YAML y Multi-GPU
Soporta configuración desde YAML, múltiples arquitecturas y DDP.

Autores: Luis Huacho y Dominick Alvarez - Maestría en Informática, PUCP
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CARGA DE CONFIGURACIÓN
# ============================================================================

def load_config_from_env():
    """
    Carga configuración desde variable de entorno TRAIN_CONFIG_PATH.
    Si no está disponible, usa settings.py por defecto.

    Returns:
        Diccionario con la configuración
    """
    config_path = os.environ.get('TRAIN_CONFIG_PATH')

    if config_path and os.path.exists(config_path):
        print(f"📋 Cargando configuración desde: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        print("📋 Usando configuración desde settings.py")
        try:
            from settings import get_segmentation_config
            return get_segmentation_config()
        except ImportError:
            raise RuntimeError("No se pudo cargar configuración ni desde YAML ni desde settings.py")


def convert_yaml_config_to_trainer_format(yaml_config):
    """
    Convierte configuración YAML al formato esperado por Trainer.

    Args:
        yaml_config: Config desde YAML

    Returns:
        Config en formato trainer
    """
    training = yaml_config.get('training', {})
    model = yaml_config.get('model', {})
    dataset = yaml_config.get('dataset', {})

    trainer_config = {
        'batch_size': training.get('batch_size', 16),
        'learning_rate': training.get('learning_rate', 1e-4),
        'weight_decay': training.get('weight_decay', 1e-6),
        'num_epochs': training.get('num_epochs', 100),
        'image_size': model.get('image_size', 384),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': training.get('num_workers', 8),
        'pin_memory': training.get('pin_memory', True),
        'mixed_precision': training.get('mixed_precision', True),
        'gradient_clip_max_norm': training.get('gradient_clip_max_norm', 0.5),
        # Configuración de modelo
        'architecture': model.get('architecture', 'resnet50'),
        'use_pretrained': model.get('use_pretrained', True),
        'use_attention': model.get('use_attention', True),
        # Configuración de dataset
        'dataset_type': dataset.get('type', 'coco'),
        'dataset_root': dataset.get('root', 'COCO'),
        'min_person_area': dataset.get('min_person_area', 500),
        'min_keypoints': dataset.get('min_keypoints', 3),
        'train_val_split': dataset.get('train_val_split', 0.8),
        # Sampling
        'sampling': dataset.get('sampling', {'enabled': False}),
        # Loss weights
        'loss_weights': training.get('loss_weights', {}),
        # Scheduler
        'scheduler': training.get('scheduler', {'type': 'cosine_annealing', 'T_0': 10, 'T_mult': 2}),
        # Checkpoints
        'checkpoint_dir': yaml_config.get('checkpoints', {}).get('checkpoint_dir', 'checkpoints'),
        # Experiment info
        'experiment_name': yaml_config.get('experiment', {}).get('name', 'U-Net Training'),
    }

    return trainer_config


# ============================================================================
# TRAINER CLASS (usando la estructura de trainer-r34.py)
# ============================================================================

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
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Scheduler según configuración
        scheduler_config = config.get('scheduler', {'type': 'cosine_annealing'})
        if scheduler_config['type'] == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 2)
            )
        elif scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None

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
        self.train_history = {
            'loss': [], 'iou': [], 'dice': [],
            'precision': [], 'recall': [], 'f1_score': []
        }
        self.val_history = {
            'loss': [], 'iou': [], 'dice': [],
            'precision': [], 'recall': [], 'f1_score': []
        }

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
        epoch_precisions = []
        epoch_recalls = []
        epoch_f1s = []

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            try:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Verificar que no hay NaN
                if torch.isnan(images).any() or torch.isnan(targets).any():
                    self.logger.warning(f"NaN detectado en batch {batch_idx}, saltando...")
                    continue

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Calcular pérdidas
                if self.loss_calculator:
                    loss = self.loss_calculator.calculate_total_loss(outputs, targets)
                else:
                    # Fallback a BCE simple
                    loss = torch.nn.functional.binary_cross_entropy(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_max_norm']
                    )

                self.optimizer.step()

                # Calcular métricas
                if self.metrics_calculator:
                    metrics = self.metrics_calculator.calculate_metrics(outputs, targets)
                    epoch_losses.append(loss.item())
                    epoch_ious.append(metrics['iou'])
                    epoch_dices.append(metrics['dice'])
                    epoch_precisions.append(metrics['precision'])
                    epoch_recalls.append(metrics['recall'])
                    epoch_f1s.append(metrics['f1_score'])

                # Log progreso
                if self.rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                          f"Loss: {loss.item():.4f}")

            except Exception as e:
                self.logger.error(f"Error en batch {batch_idx}: {e}")
                continue

        # Promediar métricas de la época
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_iou = np.mean(epoch_ious) if epoch_ious else 0
        avg_dice = np.mean(epoch_dices) if epoch_dices else 0
        avg_precision = np.mean(epoch_precisions) if epoch_precisions else 0
        avg_recall = np.mean(epoch_recalls) if epoch_recalls else 0
        avg_f1 = np.mean(epoch_f1s) if epoch_f1s else 0

        self.train_history['loss'].append(avg_loss)
        self.train_history['iou'].append(avg_iou)
        self.train_history['dice'].append(avg_dice)
        self.train_history['precision'].append(avg_precision)
        self.train_history['recall'].append(avg_recall)
        self.train_history['f1_score'].append(avg_f1)

        if self.rank == 0:
            self.logger.info(
                f"Epoch {self.epoch} - Train Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, "
                f"Dice: {avg_dice:.4f}, F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
            )

    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        val_losses = []
        val_ious = []
        val_dices = []
        val_precisions = []
        val_recalls = []
        val_f1s = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)

                if self.loss_calculator:
                    loss = self.loss_calculator.calculate_total_loss(outputs, targets)
                    val_losses.append(loss.item())

                if self.metrics_calculator:
                    metrics = self.metrics_calculator.calculate_metrics(outputs, targets)
                    val_ious.append(metrics['iou'])
                    val_dices.append(metrics['dice'])
                    val_precisions.append(metrics['precision'])
                    val_recalls.append(metrics['recall'])
                    val_f1s.append(metrics['f1_score'])

        avg_loss = np.mean(val_losses) if val_losses else 0
        avg_iou = np.mean(val_ious) if val_ious else 0
        avg_dice = np.mean(val_dices) if val_dices else 0
        avg_precision = np.mean(val_precisions) if val_precisions else 0
        avg_recall = np.mean(val_recalls) if val_recalls else 0
        avg_f1 = np.mean(val_f1s) if val_f1s else 0

        self.val_history['loss'].append(avg_loss)
        self.val_history['iou'].append(avg_iou)
        self.val_history['dice'].append(avg_dice)
        self.val_history['precision'].append(avg_precision)
        self.val_history['recall'].append(avg_recall)
        self.val_history['f1_score'].append(avg_f1)

        if self.rank == 0:
            self.logger.info(
                f"Epoch {self.epoch} - Val Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, "
                f"Dice: {avg_dice:.4f}, F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
            )

        return avg_loss, avg_iou, avg_dice

    def train(self, num_epochs):
        """Loop principal de entrenamiento."""
        best_val_iou = 0.0

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch

            # Entrenar
            self.train_epoch()

            # Validar
            val_loss, val_iou, val_dice = self.validate()

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Guardar mejor modelo
            if self.rank == 0 and val_iou > best_val_iou:
                best_val_iou = val_iou
                if self.checkpoint_manager:
                    checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_path = os.path.join(checkpoint_dir, 'best_segmentation.pth')

                    # Unwrap DDP model if necessary
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_iou': val_iou,
                        'config': self.config
                    }, save_path)
                    self.logger.info(f"✅ Mejor modelo guardado en {save_path} (IoU: {val_iou:.4f})")

        if self.rank == 0:
            self.logger.info("🎉 Entrenamiento completado!")

            # Medir y loggear métricas de eficiencia
            try:
                from utils import EfficiencyMetrics
                self.logger.info("\n📊 Métricas de Eficiencia del Modelo:")

                # Determinar input size correcto según el modelo
                # ResNet-50/34 esperan 3 canales de entrada (RGB)
                input_size = (1, 3, self.config.get('image_size', 384), self.config.get('image_size', 384))

                # Obtener modelo sin DDP wrapper para mediciones
                model_to_measure = self.model.module if hasattr(self.model, 'module') else self.model

                efficiency = EfficiencyMetrics.get_all_efficiency_metrics(
                    model_to_measure,
                    input_size=input_size,
                    device=self.device,
                    num_runs=50  # Reducido para no demorar mucho
                )

                self.logger.info(f"  • Parámetros totales: {efficiency['total_params_M']:.2f}M ({efficiency['total_params']:,})")
                self.logger.info(f"  • Parámetros entrenables: {efficiency['trainable_params_M']:.2f}M ({efficiency['trainable_params']:,})")
                self.logger.info(f"  • FPS (Frames por segundo): {efficiency['fps']:.2f}")
                self.logger.info(f"  • Tiempo promedio por imagen: {efficiency['avg_time_ms']:.2f} ± {efficiency['std_time_ms']:.2f} ms")
                self.logger.info(f"  • Memoria GPU pico: {efficiency['peak_memory_mb']:.2f} MB")
                self.logger.info(f"  • Memoria GPU asignada: {efficiency['allocated_memory_mb']:.2f} MB")
            except Exception as e:
                self.logger.warning(f"⚠️  No se pudieron calcular métricas de eficiencia: {e}")


# ============================================================================
# SAMPLING DE DATASET
# ============================================================================

def apply_sampling(dataset, sampling_config, logger=None):
    """
    Aplica muestreo al dataset según configuración.

    Args:
        dataset: Dataset completo
        sampling_config: Configuración de sampling
        logger: Logger opcional

    Returns:
        Dataset (subset si sampling está habilitado, completo si no)
    """
    if not sampling_config.get('enabled', False):
        return dataset

    mode = sampling_config.get('mode', 'full')

    if mode == 'full':
        return dataset

    total_size = len(dataset)

    if mode == 'subset':
        subset_size = sampling_config.get('subset_size')
        if not subset_size:
            if logger:
                logger.warning("Sampling mode 'subset' requiere 'subset_size', usando dataset completo")
            return dataset

        subset_size = min(subset_size, total_size)
        strategy = sampling_config.get('strategy', 'random')

        if strategy == 'random':
            seed = sampling_config.get('random_seed', 42)
            np.random.seed(seed)
            indices = np.random.choice(total_size, subset_size, replace=False)
        elif strategy == 'first':
            indices = list(range(subset_size))
        else:
            indices = list(range(subset_size))

        if logger:
            logger.info(f"📊 Aplicando subset: {subset_size} / {total_size} imágenes ({subset_size/total_size*100:.1f}%)")

        return Subset(dataset, indices)

    elif mode == 'percentage':
        percentage = sampling_config.get('percentage')
        if not percentage:
            if logger:
                logger.warning("Sampling mode 'percentage' requiere 'percentage', usando dataset completo")
            return dataset

        subset_size = int(total_size * percentage)
        strategy = sampling_config.get('strategy', 'random')

        if strategy == 'random':
            seed = sampling_config.get('random_seed', 42)
            np.random.seed(seed)
            indices = np.random.choice(total_size, subset_size, replace=False)
        elif strategy == 'first':
            indices = list(range(subset_size))
        else:
            indices = list(range(subset_size))

        if logger:
            logger.info(f"📊 Aplicando percentage: {subset_size} / {total_size} imágenes ({percentage*100}%)")

        return Subset(dataset, indices)

    return dataset


# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def train_segmentation(config=None):
    """
    Función principal de entrenamiento con soporte DDP.

    Args:
        config: Configuración (dict). Si es None, se carga desde env o settings.py

    Returns:
        True si el entrenamiento fue exitoso
    """
    # Setup de DDP
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device_id = local_rank
    else:
        device_id = 0 if torch.cuda.is_available() else None

    # Setup logger
    logger = logging.getLogger('main')
    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.CRITICAL)

    # Cargar configuración
    if config is None:
        config = load_config_from_env()

    # Si config viene de YAML, convertir a formato trainer
    if 'training' in config:  # Es un config YAML
        config = convert_yaml_config_to_trainer_format(config)

    if rank == 0:
        logger.info(f"🎯 Experimento: {config.get('experiment_name', 'U-Net Training')}")
        logger.info(f"🏗️  Arquitectura: {config.get('architecture', 'resnet50')}")
        logger.info(f"📊 Batch size: {config['batch_size']}, Epochs: {config['num_epochs']}")

    # Cargar datasets
    try:
        dataset_type = config.get('dataset_type', 'coco')

        if dataset_type == 'coco':
            from datasets import COCOPersonDataset, get_transforms

            train_dataset = COCOPersonDataset(
                root=config.get('dataset_root', 'datasets/COCO'),
                split='train',
                transforms=get_transforms(train=True, size=config['image_size'])
            )

            val_dataset = COCOPersonDataset(
                root=config.get('dataset_root', 'datasets/COCO'),
                split='val',
                transforms=get_transforms(train=False, size=config['image_size'])
            )

        elif dataset_type == 'aisegment':
            from datasets import AISegmentDataset, get_transforms

            if rank == 0:
                logger.info(f"📊 Cargando AISegment Matting Human Dataset...")

            train_dataset = AISegmentDataset(
                root=config.get('dataset_root', 'datasets/AISegment'),
                split='train',
                transforms=get_transforms(train=True, size=config['image_size']),
                auto_download=config.get('auto_download', True),
                kaggle_dataset_id=config.get('kaggle_dataset_id', 'laurentmih/aisegmentcom-matting-human-datasets'),
                train_val_split=config.get('train_val_split', 0.8),
                random_seed=config.get('random_seed', 42)
            )

            # Usar el path real del train_dataset (después de descarga automática)
            actual_dataset_root = train_dataset.root

            val_dataset = AISegmentDataset(
                root=actual_dataset_root,  # Usar el path real después de descarga
                split='val',
                transforms=get_transforms(train=False, size=config['image_size']),
                auto_download=False,  # Ya descargado en train
                kaggle_dataset_id=config.get('kaggle_dataset_id', 'laurentmih/aisegmentcom-matting-human-datasets'),
                train_val_split=config.get('train_val_split', 0.8),
                random_seed=config.get('random_seed', 42)
            )

        else:
            raise ValueError(f"Dataset type '{dataset_type}' no soportado. Opciones: coco, aisegment")

    except ImportError as e:
        if rank == 0:
            logger.error(f"Error cargando dataset: {e}")
        return False

    # Aplicar sampling si está configurado
    sampling_config = config.get('sampling', {'enabled': False})
    if sampling_config.get('enabled', False):
        train_dataset = apply_sampling(train_dataset, sampling_config, logger if rank == 0 else None)
        val_dataset = apply_sampling(val_dataset, sampling_config, logger if rank == 0 else None)

    # Crear samplers para DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    shuffle = False if is_distributed else True

    # Crear data loaders
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

    if rank == 0:
        logger.info(f"📊 Dataset cargado: {len(train_dataset)} train, {len(val_dataset)} val")

    # Crear modelo según arquitectura
    architecture = config.get('architecture', 'resnet50')

    try:
        if architecture == 'resnet50':
            from models import UNetAutoencoder
            model = UNetAutoencoder(
                pretrained=config.get('use_pretrained', True),
                use_attention=config.get('use_attention', True)
            )
        elif architecture == 'resnet34':
            from models import UNetAutoencoder_R34
            model = UNetAutoencoder_R34(
                pretrained=config.get('use_pretrained', True),
                use_attention=config.get('use_attention', True)
            )
        else:
            raise ValueError(f"Arquitectura '{architecture}' no soportada")

        model = model.to(config['device'])

        if rank == 0:
            logger.info(f"🧠 Modelo {architecture} inicializado")

    except ImportError as e:
        if rank == 0:
            logger.error(f"Error cargando modelo: {e}")
        return False

    # Envolver con DDP
    if is_distributed:
        model = DDP(model, device_ids=[device_id])

    # Contar parámetros
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"📊 Parámetros totales: {total_params:,}")
        logger.info(f"📊 Parámetros entrenables: {trainable_params:,}")

    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        config=config,
        rank=rank
    )

    # Entrenar
    trainer.train(config['num_epochs'])

    if rank == 0:
        logger.info("✅ Entrenamiento completado!")

    # Cleanup DDP
    if is_distributed:
        dist.destroy_process_group()

    return True


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Verificar si se ejecuta con torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Ejecutar entrenamiento distribuido
        train_segmentation()
    else:
        # Prueba del módulo
        print("=== PRUEBA DEL MÓDULO DE ENTRENAMIENTO ===")
        print("⚠️  Para entrenamiento distribuido usa: torchrun --nproc_per_node=N trainer.py")

        # Configuración de prueba
        test_config = {
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'num_epochs': 2,
            'image_size': 384,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 0,
            'pin_memory': False,
            'architecture': 'resnet34',  # Más ligero para pruebas
            'use_pretrained': True,
            'use_attention': True,
            'dataset_type': 'coco',
            'dataset_root': 'COCO',
            'sampling': {'enabled': True, 'mode': 'subset', 'subset_size': 50},
            'checkpoint_dir': 'checkpoints/test',
            'experiment_name': 'Test Training'
        }

        print("Iniciando entrenamiento de prueba...")
        success = train_segmentation(test_config)

        if success:
            print("✅ Prueba de entrenamiento completada exitosamente!")
        else:
            print("❌ Error en la prueba de entrenamiento")
