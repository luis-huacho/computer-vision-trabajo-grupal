#!/usr/bin/env python3
"""
Config Loader - Gestión de Configuraciones YAML
Carga y valida configuraciones desde archivos YAML para experimentos de deep learning.

Autores: Luis Huacho y Dominick Alvarez - Maestría en Informática, PUCP
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy


# ============================================================================
# FUNCIONES DE CARGA DE YAML
# ============================================================================

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Carga un archivo YAML y retorna su contenido como diccionario.

    Args:
        file_path: Path al archivo YAML

    Returns:
        Diccionario con la configuración

    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay errores de sintaxis YAML
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Archivo YAML vacío: {file_path}")

        print(f"✅ Configuración cargada desde: {file_path}")
        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parseando YAML en {file_path}: {e}")


def find_config_file(config_name: str, configs_dir: str = "configs") -> Path:
    """
    Busca un archivo de configuración en el directorio de configs.

    Args:
        config_name: Nombre del archivo (con o sin .yaml)
        configs_dir: Directorio donde buscar

    Returns:
        Path completo al archivo

    Raises:
        FileNotFoundError: Si no se encuentra el archivo
    """
    configs_path = Path(configs_dir)

    # Si no existe el directorio de configs
    if not configs_path.exists():
        raise FileNotFoundError(f"Directorio de configuraciones no encontrado: {configs_path}")

    # Intentar con y sin extensión .yaml
    possible_names = [
        config_name,
        f"{config_name}.yaml",
        f"{config_name}.yml"
    ]

    for name in possible_names:
        file_path = configs_path / name
        if file_path.exists():
            return file_path

    # Listar archivos disponibles
    available_configs = list(configs_path.glob("*.yaml")) + list(configs_path.glob("*.yml"))
    available_names = [f.stem for f in available_configs]

    raise FileNotFoundError(
        f"Configuración '{config_name}' no encontrada en {configs_path}\n"
        f"Configuraciones disponibles: {', '.join(available_names)}"
    )


# ============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================================

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valida que la configuración tenga la estructura correcta.

    Args:
        config: Diccionario de configuración

    Returns:
        True si es válida

    Raises:
        ValueError: Si falta alguna sección requerida o tiene valores inválidos
    """
    # Secciones requeridas
    required_sections = ['experiment', 'model', 'dataset', 'training']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Sección requerida '{section}' no encontrada en configuración")

    # Validar experiment
    if 'name' not in config['experiment']:
        raise ValueError("Falta 'experiment.name' en configuración")

    # Validar model
    model = config['model']
    if 'architecture' not in model:
        raise ValueError("Falta 'model.architecture' en configuración")

    valid_architectures = ['resnet50', 'resnet34']
    if model['architecture'] not in valid_architectures:
        raise ValueError(
            f"Arquitectura '{model['architecture']}' no válida. "
            f"Opciones: {', '.join(valid_architectures)}"
        )

    if 'image_size' not in model:
        raise ValueError("Falta 'model.image_size' en configuración")

    # Validar dataset
    dataset = config['dataset']
    if 'type' not in dataset:
        raise ValueError("Falta 'dataset.type' en configuración")

    valid_dataset_types = ['coco', 'aisegment', 'supervisely']
    if dataset['type'] not in valid_dataset_types:
        raise ValueError(
            f"Dataset type '{dataset['type']}' no válido. "
            f"Opciones: {', '.join(valid_dataset_types)}"
        )

    # Validar sampling si está habilitado
    if 'sampling' in dataset and dataset['sampling'].get('enabled', False):
        sampling = dataset['sampling']
        mode = sampling.get('mode', 'full')

        if mode == 'subset' and not sampling.get('subset_size'):
            raise ValueError("Sampling mode 'subset' requiere 'subset_size'")

        if mode == 'percentage':
            percentage = sampling.get('percentage')
            if percentage is None:
                raise ValueError("Sampling mode 'percentage' requiere 'percentage'")
            if not (0 < percentage <= 1):
                raise ValueError(f"Percentage debe estar entre 0 y 1, recibido: {percentage}")

    # Validar training
    training = config['training']
    required_training = ['num_epochs', 'batch_size', 'learning_rate']
    for param in required_training:
        if param not in training:
            raise ValueError(f"Falta 'training.{param}' en configuración")

    # Validar valores positivos
    if training['num_epochs'] <= 0:
        raise ValueError(f"num_epochs debe ser > 0, recibido: {training['num_epochs']}")
    if training['batch_size'] <= 0:
        raise ValueError(f"batch_size debe ser > 0, recibido: {training['batch_size']}")
    if training['learning_rate'] <= 0:
        raise ValueError(f"learning_rate debe ser > 0, recibido: {training['learning_rate']}")

    print("✅ Configuración validada correctamente")
    return True


# ============================================================================
# FUSIÓN CON VALORES POR DEFECTO
# ============================================================================

def merge_with_defaults(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona configuración YAML con valores por defecto de settings.py.

    Args:
        yaml_config: Configuración desde YAML

    Returns:
        Configuración completa con valores por defecto aplicados
    """
    try:
        from settings import (
            ExperimentConfig, SegmentationConfig, HarmonizationConfig,
            COCOConfig, ValidationConfig, LoggingConfig
        )
    except ImportError:
        print("⚠️  settings.py no disponible, usando solo configuración YAML")
        return yaml_config

    # Crear copia para no modificar original
    config = copy.deepcopy(yaml_config)

    # Aplicar valores por defecto de settings.py donde no estén especificados

    # Model defaults
    model = config.get('model', {})
    model.setdefault('use_pretrained', SegmentationConfig.USE_PRETRAINED)
    model.setdefault('use_attention', SegmentationConfig.USE_ATTENTION)
    model.setdefault('image_size', SegmentationConfig.IMAGE_SIZE)

    # Training defaults
    training = config.get('training', {})
    training.setdefault('num_workers', SegmentationConfig.NUM_WORKERS)
    training.setdefault('pin_memory', SegmentationConfig.PIN_MEMORY)
    training.setdefault('drop_last', SegmentationConfig.DROP_LAST)
    training.setdefault('gradient_clip_max_norm', SegmentationConfig.GRADIENT_CLIP_MAX_NORM)
    training.setdefault('mixed_precision', SegmentationConfig.MIXED_PRECISION)
    training.setdefault('weight_decay', SegmentationConfig.WEIGHT_DECAY)

    # Loss weights defaults
    loss_weights = training.get('loss_weights', {})
    default_weights = SegmentationConfig.LOSS_WEIGHTS
    for key, value in default_weights.items():
        loss_weights.setdefault(key, value)
    training['loss_weights'] = loss_weights

    # Dataset defaults
    dataset = config.get('dataset', {})
    dataset.setdefault('min_person_area', COCOConfig.MIN_PERSON_AREA)
    dataset.setdefault('min_keypoints', COCOConfig.MIN_KEYPOINTS)
    dataset.setdefault('train_val_split', COCOConfig.TRAIN_VAL_SPLIT)

    # Validation defaults
    validation = config.get('validation', {})
    validation.setdefault('frequency', ValidationConfig.VALIDATION_FREQUENCY)

    early_stopping = validation.get('early_stopping', {})
    early_stopping.setdefault('enabled', True)
    early_stopping.setdefault('patience', ValidationConfig.EARLY_STOPPING_PATIENCE)
    early_stopping.setdefault('min_delta', ValidationConfig.EARLY_STOPPING_MIN_DELTA)
    validation['early_stopping'] = early_stopping

    # Checkpoints defaults
    checkpoints = config.get('checkpoints', {})
    checkpoints.setdefault('save_best', True)
    checkpoints.setdefault('save_last', True)
    checkpoints.setdefault('save_every_n_epochs', LoggingConfig.SAVE_CHECKPOINT_EVERY_N_EPOCHS)
    checkpoints.setdefault('checkpoint_dir', ExperimentConfig.CHECKPOINT_DIR)

    # Logging defaults
    logging = config.get('logging', {})
    logging.setdefault('log_every_n_batches', LoggingConfig.LOG_EVERY_N_BATCHES)
    logging.setdefault('save_plots', LoggingConfig.SAVE_PLOTS)
    logging.setdefault('plot_dpi', LoggingConfig.PLOT_DPI)

    # Actualizar config
    config['model'] = model
    config['training'] = training
    config['dataset'] = dataset
    config['validation'] = validation
    config['checkpoints'] = checkpoints
    config['logging'] = logging

    print("✅ Configuración fusionada con valores por defecto")
    return config


# ============================================================================
# APLICACIÓN DE CONFIGURACIÓN
# ============================================================================

def apply_config_to_settings(config: Dict[str, Any]) -> None:
    """
    Aplica configuración YAML a las clases de settings.py (modifica en lugar).
    NOTA: Solo modifica atributos que sean seguros cambiar dinámicamente.

    Args:
        config: Configuración completa
    """
    try:
        from settings import SegmentationConfig, COCOConfig
    except ImportError:
        print("⚠️  settings.py no disponible, saltando aplicación de configuración")
        return

    # Aplicar configuración de entrenamiento
    training = config.get('training', {})
    if 'num_epochs' in training:
        SegmentationConfig.NUM_EPOCHS = training['num_epochs']
    if 'batch_size' in training:
        SegmentationConfig.BATCH_SIZE = training['batch_size']
    if 'learning_rate' in training:
        SegmentationConfig.LEARNING_RATE = training['learning_rate']
    if 'num_workers' in training:
        SegmentationConfig.NUM_WORKERS = training['num_workers']

    # Aplicar loss weights
    if 'loss_weights' in training:
        SegmentationConfig.LOSS_WEIGHTS = training['loss_weights']

    # Aplicar configuración de modelo
    model = config.get('model', {})
    if 'image_size' in model:
        SegmentationConfig.IMAGE_SIZE = model['image_size']

    # Aplicar configuración de dataset
    dataset = config.get('dataset', {})
    if 'min_person_area' in dataset:
        COCOConfig.MIN_PERSON_AREA = dataset['min_person_area']
    if 'min_keypoints' in dataset:
        COCOConfig.MIN_KEYPOINTS = dataset['min_keypoints']

    print("✅ Configuración aplicada a settings.py")


# ============================================================================
# FUNCIÓN PRINCIPAL DE CARGA
# ============================================================================

def load_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    validate: bool = True,
    apply_to_settings: bool = False
) -> Dict[str, Any]:
    """
    Función principal para cargar configuración desde YAML.

    Args:
        config_path: Path completo al archivo YAML (prioritario)
        config_name: Nombre del archivo en configs/ (si no se especifica path)
        validate: Si True, valida la configuración
        apply_to_settings: Si True, aplica valores a settings.py

    Returns:
        Diccionario con la configuración completa

    Raises:
        FileNotFoundError: Si no se encuentra el archivo
        ValueError: Si la configuración es inválida
    """
    # Determinar path del archivo
    if config_path:
        file_path = Path(config_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {config_path}")
    elif config_name:
        file_path = find_config_file(config_name)
    else:
        # Por defecto, buscar default.yaml
        file_path = find_config_file("default")

    # Cargar YAML
    config = load_yaml(file_path)

    # Validar
    if validate:
        validate_config(config)

    # Fusionar con defaults
    config = merge_with_defaults(config)

    # Aplicar a settings si se solicita
    if apply_to_settings:
        apply_config_to_settings(config)

    return config


# ============================================================================
# UTILIDADES
# ============================================================================

def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Imprime un resumen legible de la configuración.

    Args:
        config: Diccionario de configuración
    """
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE CONFIGURACIÓN")
    print("=" * 60)

    # Experimento
    exp = config.get('experiment', {})
    print(f"\n🎯 Experimento: {exp.get('name', 'N/A')}")
    print(f"   Descripción: {exp.get('description', 'N/A')}")
    print(f"   Modo: {exp.get('mode', 'N/A')}")

    # Modelo
    model = config.get('model', {})
    print(f"\n🧠 Modelo:")
    print(f"   Arquitectura: {model.get('architecture', 'N/A')}")
    print(f"   Image size: {model.get('image_size', 'N/A')}")
    print(f"   Pretrained: {model.get('use_pretrained', 'N/A')}")
    print(f"   Attention: {model.get('use_attention', 'N/A')}")

    # Dataset
    dataset = config.get('dataset', {})
    print(f"\n📊 Dataset:")
    print(f"   Tipo: {dataset.get('type', 'N/A')}")
    print(f"   Root: {dataset.get('root', 'N/A')}")

    sampling = dataset.get('sampling', {})
    if sampling.get('enabled', False):
        print(f"   Sampling: {sampling.get('mode', 'N/A')}")
        if sampling.get('mode') == 'subset':
            print(f"   Subset size: {sampling.get('subset_size', 'N/A')}")
        elif sampling.get('mode') == 'percentage':
            print(f"   Percentage: {sampling.get('percentage', 'N/A') * 100}%")
    else:
        print(f"   Sampling: Completo (sin muestreo)")

    # Training
    training = config.get('training', {})
    print(f"\n🏋️  Entrenamiento:")
    print(f"   Épocas: {training.get('num_epochs', 'N/A')}")
    print(f"   Batch size: {training.get('batch_size', 'N/A')}")
    print(f"   Learning rate: {training.get('learning_rate', 'N/A')}")
    print(f"   Workers: {training.get('num_workers', 'N/A')}")
    print(f"   Mixed precision: {training.get('mixed_precision', 'N/A')}")

    # Loss weights
    loss_weights = training.get('loss_weights', {})
    print(f"\n⚖️  Loss weights:")
    print(f"   Alpha (BCE): {loss_weights.get('alpha', 'N/A')}")
    print(f"   Beta (Dice): {loss_weights.get('beta', 'N/A')}")
    print(f"   Gamma (Perceptual): {loss_weights.get('gamma', 'N/A')}")
    print(f"   Delta (Edge): {loss_weights.get('delta', 'N/A')}")

    # Checkpoints
    checkpoints = config.get('checkpoints', {})
    print(f"\n💾 Checkpoints:")
    print(f"   Directorio: {checkpoints.get('checkpoint_dir', 'N/A')}")

    print("\n" + "=" * 60 + "\n")


def list_available_configs(configs_dir: str = "configs") -> list:
    """
    Lista todas las configuraciones disponibles en el directorio.

    Args:
        configs_dir: Directorio de configuraciones

    Returns:
        Lista de nombres de configuraciones disponibles
    """
    configs_path = Path(configs_dir)

    if not configs_path.exists():
        print(f"⚠️  Directorio {configs_dir} no encontrado")
        return []

    yaml_files = list(configs_path.glob("*.yaml")) + list(configs_path.glob("*.yml"))
    config_names = [f.stem for f in yaml_files]

    return sorted(config_names)


def print_available_configs(configs_dir: str = "configs") -> None:
    """
    Imprime lista de configuraciones disponibles.

    Args:
        configs_dir: Directorio de configuraciones
    """
    configs = list_available_configs(configs_dir)

    if not configs:
        print("⚠️  No se encontraron configuraciones")
        return

    print("📁 CONFIGURACIONES DISPONIBLES:")
    print("-" * 40)
    for config_name in configs:
        print(f"   - {config_name}")
    print("-" * 40)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 TESTING CONFIG LOADER\n")

    # Listar configuraciones disponibles
    print_available_configs()
    print()

    # Probar carga de default.yaml
    try:
        config = load_config(config_name="default", validate=True)
        print_config_summary(config)
        print("✅ Test exitoso!")
    except Exception as e:
        print(f"❌ Error en test: {e}")
