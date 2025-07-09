import torch
import os
from datetime import datetime


# ============================================================================
# CONFIGURACIONES PRINCIPALES DEL EXPERIMENTO
# ============================================================================

class ExperimentConfig:
    """
    Configuración centralizada para todos los experimentos.
    """

    # Información del experimento
    EXPERIMENT_NAME = "U-Net Background Removal with Harmonization"
    VERSION = "2.0"
    AUTHORS = ["Luis Huacho", "Dominick Alvarez"]
    INSTITUTION = "Maestría en Informática, PUCP"

    # Configuración de dispositivos
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CUDA_AVAILABLE = torch.cuda.is_available()

    # Directorios principales
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'COCO')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

    # Directorios de harmonización
    HARMONIZATION_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    FOREGROUND_DIR = os.path.join(HARMONIZATION_DATA_DIR, 'foregrounds')
    BACKGROUND_DIR = os.path.join(HARMONIZATION_DATA_DIR, 'backgrounds')

    # Crear directorios si no existen
    @classmethod
    def create_directories(cls):
        """Crea todos los directorios necesarios."""
        dirs_to_create = [
            cls.CHECKPOINT_DIR,
            cls.LOGS_DIR,
            cls.PLOTS_DIR,
            cls.HARMONIZATION_DATA_DIR,
            cls.FOREGROUND_DIR,
            cls.BACKGROUND_DIR
        ]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

        print(f"✅ Directorios del experimento creados/verificados")


# ============================================================================
# CONFIGURACIONES DE ENTRENAMIENTO - SEGMENTACIÓN
# ============================================================================

class SegmentationConfig:
    """
    Configuración específica para el entrenamiento de segmentación.
    """

    # Parámetros del modelo
    IMAGE_SIZE = 384
    USE_PRETRAINED = True
    USE_ATTENTION = True

    # Parámetros de entrenamiento
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    NUM_EPOCHS = 100

    # Configuración del DataLoader
    NUM_WORKERS = 8
    PIN_MEMORY = True
    DROP_LAST = True

    # Configuración del scheduler
    SCHEDULER_TYPE = 'cosine_annealing'
    T_0 = 10  # Para CosineAnnealingWarmRestarts
    T_MULT = 2

    # Configuración de pérdidas
    LOSS_WEIGHTS = {
        'alpha': 1.0,  # BCE weight
        'beta': 1.0,  # Dice weight
        'gamma': 0.5,  # Perceptual weight
        'delta': 0.3  # Edge weight
    }

    # Configuración de optimización
    GRADIENT_CLIP_MAX_NORM = 0.5
    MIXED_PRECISION = True

    # Paths de modelos
    BEST_MODEL_PATH = os.path.join(ExperimentConfig.CHECKPOINT_DIR, 'best_segmentation.pth')
    LAST_MODEL_PATH = os.path.join(ExperimentConfig.CHECKPOINT_DIR, 'last_segmentation.pth')

    @classmethod
    def get_config_dict(cls):
        """Retorna configuración como diccionario."""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'num_epochs': cls.NUM_EPOCHS,
            'image_size': cls.IMAGE_SIZE,
            'device': ExperimentConfig.DEVICE,
            'num_workers': cls.NUM_WORKERS,
            'pin_memory': cls.PIN_MEMORY,
            'mixed_precision': cls.MIXED_PRECISION,
            'use_data_parallel': cls.USE_DATA_PARALLEL if hasattr(cls, 'USE_DATA_PARALLEL') else True,
        }


# ============================================================================
# CONFIGURACIONES DE ENTRENAMIENTO - HARMONIZACIÓN
# ============================================================================

class HarmonizationConfig:
    """
    Configuración específica para el entrenamiento de harmonización.
    """

    # Parámetros del modelo
    IMAGE_SIZE = 384
    USE_PRETRAINED = True
    USE_ATTENTION = True

    # Parámetros de entrenamiento (más conservadores)
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-6
    NUM_EPOCHS = 50

    # Configuración del DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DROP_LAST = True

    # Configuración del scheduler
    SCHEDULER_TYPE = 'cosine_annealing'
    T_0 = 10
    T_MULT = 2

    # Configuración de pérdidas para harmonización
    LOSS_WEIGHTS = {
        'alpha': 1.0,  # MSE weight
        'beta': 0.5,  # Perceptual weight
        'gamma': 0.3,  # Color consistency weight
        'delta': 0.2  # Style weight
    }

    # Configuración de optimización
    GRADIENT_CLIP_MAX_NORM = 0.5

    # Paths de modelos
    BEST_MODEL_PATH = os.path.join(ExperimentConfig.CHECKPOINT_DIR, 'best_harmonizer.pth')
    LAST_MODEL_PATH = os.path.join(ExperimentConfig.CHECKPOINT_DIR, 'last_harmonizer.pth')

    # Configuración específica de transformaciones
    AUGMENTATION_PROB = {
        'horizontal_flip': 0.5,
        'brightness_contrast': 0.7,
        'hue_saturation': 0.5,
        'color_jitter': 0.5,
        'gamma': 0.3,
        'clahe': 0.3
    }

    @classmethod
    def get_config_dict(cls):
        """Retorna configuración como diccionario."""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'num_epochs': cls.NUM_EPOCHS,
            'image_size': cls.IMAGE_SIZE,
            'device': ExperimentConfig.DEVICE,
            'num_workers': cls.NUM_WORKERS,
            'pin_memory': cls.PIN_MEMORY,
        }


# ============================================================================
# CONFIGURACIONES DE DATASET - COCO
# ============================================================================

class COCOConfig:
    """
    Configuración específica para el dataset COCO.
    """

    # Paths del dataset COCO
    COCO_ROOT = ExperimentConfig.DATA_DIR
    TRAIN_ANNOTATIONS = os.path.join(COCO_ROOT, 'annotations', 'person_keypoints_train2017.json')
    VAL_ANNOTATIONS = os.path.join(COCO_ROOT, 'annotations', 'person_keypoints_val2017.json')
    TRAIN_IMAGES = os.path.join(COCO_ROOT, 'train2017')
    VAL_IMAGES = os.path.join(COCO_ROOT, 'val2017')

    # Filtros de dataset
    MIN_PERSON_AREA = 500  # Área mínima para considerar una persona
    MIN_KEYPOINTS = 3  # Mínimo número de keypoints visibles

    # Split de datos
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val

    # Configuración de augmentación
    AUGMENTATION_CONFIG = {
        'horizontal_flip': 0.5,
        'random_rotate90': 0.3,
        'shift_scale_rotate': {
            'shift_limit': 0.1,
            'scale_limit': 0.2,
            'rotate_limit': 15,
            'p': 0.5
        },
        'brightness_contrast': {
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'p': 0.5
        },
        'hue_saturation_value': {
            'hue_shift_limit': 10,
            'sat_shift_limit': 20,
            'val_shift_limit': 10,
            'p': 0.3
        },
        'gaussian_blur': {
            'blur_limit': 3,
            'p': 0.2
        }
    }


# ============================================================================
# CONFIGURACIONES DE INFERENCIA
# ============================================================================

class InferenceConfig:
    """
    Configuración para inferencia y demos.
    """

    # Paths de modelos para inferencia
    SEGMENTATION_MODEL_PATH = SegmentationConfig.BEST_MODEL_PATH
    HARMONIZATION_MODEL_PATH = HarmonizationConfig.BEST_MODEL_PATH

    # Configuración de inferencia
    DEFAULT_IMAGE_SIZE = 384
    OUTPUT_FORMAT = 'PNG'  # Para imágenes con transparencia
    BATCH_PROCESSING_SIZE = 4

    # Paths de ejemplo
    EXAMPLE_INPUT = 'example_input.jpg'
    EXAMPLE_OUTPUT = 'example_output.png'
    EXAMPLE_BACKGROUND = 'example_background.jpg'
    EXAMPLE_HARMONIZED = 'example_harmonized.jpg'

    # Directorios de procesamiento en lote
    BATCH_INPUT_DIR = 'input_images'
    BATCH_OUTPUT_DIR = 'output_images'


# ============================================================================
# CONFIGURACIONES DE LOGGING Y MONITORING
# ============================================================================

class LoggingConfig:
    """
    Configuración para logging y monitoreo.
    """

    # Configuración de logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

    # Archivos de log
    MAIN_LOG_FILE = 'training_main.log'
    SEGMENTATION_LOG_FILE = 'training_segmentation.log'
    HARMONIZATION_LOG_FILE = 'training_harmonization.log'

    # Configuración de plots
    PLOT_DPI = 300
    PLOT_FORMAT = 'png'
    SAVE_PLOTS = True

    # Frecuencia de logging
    LOG_EVERY_N_BATCHES = 10
    SAVE_CHECKPOINT_EVERY_N_EPOCHS = 5

    # Métricas a trackear
    METRICS_TO_TRACK = [
        'loss', 'iou', 'dice', 'pixel_accuracy',
        'learning_rate', 'epoch_time'
    ]


# ============================================================================
# CONFIGURACIONES DE VALIDACIÓN Y TESTING
# ============================================================================

class ValidationConfig:
    """
    Configuración para validación y testing.
    """

    # Configuración de validación
    VALIDATION_FREQUENCY = 1  # Cada N épocas
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Métricas objetivo
    TARGET_METRICS = {
        'segmentation': {
            'iou': 0.85,
            'dice': 0.90,
            'pixel_accuracy': 0.95
        },
        'harmonization': {
            'mse': 0.01,
            'perceptual_loss': 0.05
        }
    }

    # Configuración de testing
    TEST_BATCH_SIZE = 1
    TEST_IMAGE_SIZES = [256, 384, 512]

    # Análisis de calidad
    QUALITY_ANALYSIS = {
        'coverage_threshold': 15.0,  # % mínimo de cobertura de persona
        'contrast_threshold': 60.0,  # Contraste mínimo de máscara
        'edge_threshold': 50.0,  # Definición mínima de bordes
        'resolution_threshold': 70.0  # Score mínimo de resolución
    }


# ============================================================================
# CONFIGURACIONES GLOBALES Y UTILIDADES
# ============================================================================

class GlobalConfig:
    """
    Configuración global del sistema.
    """

    # Información del sistema
    RANDOM_SEED = 42
    TORCH_BACKENDS_CUDNN_DETERMINISTIC = True
    TORCH_BACKENDS_CUDNN_BENCHMARK = False

    # Configuración de warnings
    SUPPRESS_WARNINGS = True

    # Paths de archivos importantes
    README_FILE = 'README.md'
    REQUIREMENTS_FILE = 'requirements.txt'

    # Configuración de colores para terminal
    COLORS = {
        'RED': '\033[0;31m',
        'GREEN': '\033[0;32m',
        'YELLOW': '\033[1;33m',
        'BLUE': '\033[0;34m',
        'CYAN': '\033[0;36m',
        'NC': '\033[0m'  # No Color
    }

    @classmethod
    def setup_reproducibility(cls):
        """Configura semillas para reproducibilidad."""
        import random
        import numpy as np

        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)

        torch.backends.cudnn.deterministic = cls.TORCH_BACKENDS_CUDNN_DETERMINISTIC
        torch.backends.cudnn.benchmark = cls.TORCH_BACKENDS_CUDNN_BENCHMARK

        print(f"✅ Reproducibilidad configurada (seed: {cls.RANDOM_SEED})")

    @classmethod
    def print_system_info(cls):
        """Imprime información del sistema."""
        print(f"🔍 INFORMACIÓN DEL SISTEMA:")
        print(f"   - Experimento: {ExperimentConfig.EXPERIMENT_NAME} v{ExperimentConfig.VERSION}")
        print(f"   - Autores: {', '.join(ExperimentConfig.AUTHORS)}")
        print(f"   - Institución: {ExperimentConfig.INSTITUTION}")
        print(f"   - Dispositivo: {ExperimentConfig.DEVICE}")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - CUDA disponible: {ExperimentConfig.CUDA_AVAILABLE}")

        if ExperimentConfig.CUDA_AVAILABLE:
            print(f"   - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   - Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        print(f"   - Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# FUNCIONES DE CONFIGURACIÓN
# ============================================================================

def get_segmentation_config():
    """Retorna configuración completa para segmentación."""
    return SegmentationConfig.get_config_dict()


def get_harmonization_config():
    """Retorna configuración completa para harmonización."""
    return HarmonizationConfig.get_config_dict()


def initialize_experiment():
    """Inicializa el experimento completo."""
    print("🚀 INICIALIZANDO EXPERIMENTO")
    print("=" * 50)

    # Configurar reproducibilidad
    GlobalConfig.setup_reproducibility()

    # Crear directorios
    ExperimentConfig.create_directories()

    # Mostrar información del sistema
    GlobalConfig.print_system_info()

    # Suprimir warnings si está configurado
    if GlobalConfig.SUPPRESS_WARNINGS:
        import warnings
        warnings.filterwarnings('ignore')
        print("⚠️  Warnings suprimidos")

    print("=" * 50)
    print("✅ Experimento inicializado correctamente\n")


def get_all_configs():
    """Retorna todas las configuraciones en un diccionario."""
    return {
        'experiment': ExperimentConfig,
        'segmentation': SegmentationConfig,
        'harmonization': HarmonizationConfig,
        'coco': COCOConfig,
        'inference': InferenceConfig,
        'logging': LoggingConfig,
        'validation': ValidationConfig,
        'global': GlobalConfig
    }


def print_config_summary():
    """Imprime resumen de todas las configuraciones."""
    print("📋 RESUMEN DE CONFIGURACIONES")
    print("=" * 50)

    print(f"🎯 Segmentación:")
    print(f"   - Epochs: {SegmentationConfig.NUM_EPOCHS}")
    print(f"   - Batch size: {SegmentationConfig.BATCH_SIZE}")
    print(f"   - Learning rate: {SegmentationConfig.LEARNING_RATE}")
    print(f"   - Image size: {SegmentationConfig.IMAGE_SIZE}")

    print(f"\n🎨 Harmonización:")
    print(f"   - Epochs: {HarmonizationConfig.NUM_EPOCHS}")
    print(f"   - Batch size: {HarmonizationConfig.BATCH_SIZE}")
    print(f"   - Learning rate: {HarmonizationConfig.LEARNING_RATE}")
    print(f"   - Image size: {HarmonizationConfig.IMAGE_SIZE}")

    print(f"\n📊 Dataset COCO:")
    print(f"   - Train/Val split: {COCOConfig.TRAIN_VAL_SPLIT}")
    print(f"   - Min person area: {COCOConfig.MIN_PERSON_AREA}")
    print(f"   - Data dir: {COCOConfig.COCO_ROOT}")

    print(f"\n💾 Paths:")
    print(f"   - Checkpoints: {ExperimentConfig.CHECKPOINT_DIR}")
    print(f"   - Logs: {ExperimentConfig.LOGS_DIR}")
    print(f"   - Plots: {ExperimentConfig.PLOTS_DIR}")

    print("=" * 50)


# ============================================================================
# CONFIGURACIONES ESPECÍFICAS POR MODO DE EJECUCIÓN
# ============================================================================

class ModeConfigs:
    """
    Configuraciones específicas para diferentes modos de ejecución.
    """

    QUICK_TEST = {
        'segmentation': {
            'num_epochs': 2,
            'batch_size': 2,
            'num_workers': 0
        },
        'harmonization': {
            'num_epochs': 2,
            'batch_size': 2,
            'num_workers': 0
        }
    }

    PRODUCTION = {
        'segmentation': SegmentationConfig.get_config_dict(),
        'harmonization': HarmonizationConfig.get_config_dict()
    }

    DEBUG = {
        'segmentation': {
            **SegmentationConfig.get_config_dict(),
            'num_epochs': 5,
            'batch_size': 4
        },
        'harmonization': {
            **HarmonizationConfig.get_config_dict(),
            'num_epochs': 3,
            'batch_size': 4
        }
    }


def get_config_for_mode(mode='production'):
    """
    Retorna configuración específica para un modo de ejecución.

    Args:
        mode: 'production', 'debug', o 'quick_test'
    """
    mode_configs = {
        'production': ModeConfigs.PRODUCTION,
        'debug': ModeConfigs.DEBUG,
        'quick_test': ModeConfigs.QUICK_TEST
    }

    if mode not in mode_configs:
        print(f"⚠️  Modo '{mode}' no reconocido, usando 'production'")
        mode = 'production'

    return mode_configs[mode]


if __name__ == "__main__":
    # Ejemplo de uso
    initialize_experiment()
    print_config_summary()

    # Obtener configuraciones
    seg_config = get_segmentation_config()
    harm_config = get_harmonization_config()

    print(f"\n🔧 Configuración de segmentación: {seg_config}")
    print(f"🔧 Configuración de harmonización: {harm_config}")