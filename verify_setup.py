#!/usr/bin/env python3
"""
Script de Verificación del Sistema
Verifica que todos los componentes del sistema estén correctamente configurados.

Autores: Luis Huacho y Dominick Alvarez - PUCP
"""

import sys
import os
from pathlib import Path
import importlib


def print_section(title):
    """Imprime una sección con formato."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def check_dependencies():
    """Verifica que las dependencias principales estén instaladas."""
    print_section("1. VERIFICANDO DEPENDENCIAS")

    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'albumentations': 'Albumentations',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'kagglehub': 'KaggleHub'
    }

    results = {}
    for module, name in dependencies.items():
        try:
            importlib.import_module(module)
            print(f"✅ {name:20s} - Instalado")
            results[name] = True
        except ImportError:
            print(f"❌ {name:20s} - NO instalado")
            results[name] = False

    return all(results.values())


def check_project_structure():
    """Verifica la estructura de directorios del proyecto."""
    print_section("2. VERIFICANDO ESTRUCTURA DE DIRECTORIOS")

    required_dirs = {
        'configs': 'Configuraciones YAML',
        'datasets': 'Directorio base de datasets',
        'checkpoints': 'Modelos guardados',
        'logs': 'Archivos de log',
        'plots': 'Gráficos de entrenamiento',
        'docs': 'Documentación'
    }

    results = {}
    for dir_name, description in required_dirs.items():
        exists = Path(dir_name).exists()
        status = "✅" if exists else "⚠️ "
        print(f"{status} {dir_name:15s} - {description}")
        results[dir_name] = exists

    return all(results.values())


def check_config_files():
    """Verifica que los archivos de configuración existan y sean válidos."""
    print_section("3. VERIFICANDO ARCHIVOS DE CONFIGURACIÓN")

    config_files = [
        'default.yaml',
        'resnet50_full.yaml',
        'resnet34_quick.yaml',
        'resnet50_10percent.yaml',
        'debug.yaml',
        'aisegment_full.yaml',
        'aisegment_10percent.yaml',
        'aisegment_quick.yaml'
    ]

    results = {}
    for config_file in config_files:
        config_path = Path('configs') / config_file
        exists = config_path.exists()

        if exists:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Validar estructura básica
                required_keys = ['experiment', 'model', 'dataset', 'training']
                valid = all(key in config for key in required_keys)

                if valid:
                    print(f"✅ {config_file:30s} - Válido")
                    results[config_file] = True
                else:
                    print(f"❌ {config_file:30s} - Estructura inválida")
                    results[config_file] = False
            except Exception as e:
                print(f"❌ {config_file:30s} - Error: {e}")
                results[config_file] = False
        else:
            print(f"❌ {config_file:30s} - No encontrado")
            results[config_file] = False

    return all(results.values())


def check_datasets():
    """Verifica la disponibilidad de datasets."""
    print_section("4. VERIFICANDO DATASETS")

    datasets = {
        'datasets/COCO': 'COCO Dataset (principal)',
        'datasets/AISegment': 'AISegment Matting Human Dataset',
        'persons/project': 'Supervisely Persons (legacy)'
    }

    results = {}
    for dataset_path, description in datasets.items():
        path = Path(dataset_path)
        exists = path.exists()

        if exists:
            # Contar contenido si existe
            if dataset_path == 'datasets/COCO':
                has_content = (
                    (path / 'annotations').exists() and
                    (path / 'train2017').exists() and
                    (path / 'val2017').exists()
                )
                status = "✅" if has_content else "⚠️  (vacío)"
            elif dataset_path == 'datasets/AISegment':
                has_content = (
                    (path / 'clip_img').exists() and
                    (path / 'matting').exists()
                )
                status = "✅" if has_content else "⚠️  (vacío)"
            else:
                status = "✅"

            print(f"{status} {description:40s} - {dataset_path}")
            results[dataset_path] = has_content if 'dataset' in dataset_path.lower() else True
        else:
            print(f"❌ {description:40s} - No encontrado")
            results[dataset_path] = False

    return any(results.values())  # Al menos un dataset debe existir


def check_project_modules():
    """Verifica que los módulos del proyecto sean importables."""
    print_section("5. VERIFICANDO MÓDULOS DEL PROYECTO")

    modules = {
        'settings': 'Configuración del sistema',
        'config_loader': 'Cargador de configs YAML',
        'datasets': 'Datasets (COCO, AISegment)',
        'models': 'Modelos (U-Net)',
        'utils': 'Utilidades y métricas',
        'trainer': 'Entrenador con DDP',
        'inference': 'Inferencia',
        'harmonization': 'Armonización'
    }

    results = {}
    for module, description in modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module:20s} - {description}")
            results[module] = True
        except Exception as e:
            print(f"❌ {module:20s} - Error: {str(e)[:50]}")
            results[module] = False

    return all(results.values())


def check_kaggle_api():
    """Verifica configuración de Kaggle API."""
    print_section("6. VERIFICANDO KAGGLE API (para AISegment)")

    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if kaggle_json.exists():
        try:
            import json
            with open(kaggle_json, 'r') as f:
                credentials = json.load(f)

            if 'username' in credentials and 'key' in credentials:
                print(f"✅ API Key de Kaggle configurada")
                print(f"   Usuario: {credentials.get('username')}")
                print(f"   Path: {kaggle_json}")
                return True
            else:
                print(f"⚠️  API Key existe pero estructura inválida")
                return False
        except Exception as e:
            print(f"❌ Error leyendo API Key: {e}")
            return False
    else:
        print(f"⚠️  API Key de Kaggle no encontrada")
        print(f"   Path esperado: {kaggle_json}")
        print(f"   Nota: Requerido para descarga automática de AISegment")
        print(f"   Ver: docs/AISegment_Setup.md")
        return False


def test_config_loader():
    """Test básico del config_loader."""
    print_section("7. TESTING CONFIG LOADER")

    try:
        from config_loader import load_config, validate_config

        # Intentar cargar default.yaml
        config = load_config(config_name="default", validate=True)
        print(f"✅ Config loader funcional")
        print(f"   Config default cargado exitosamente")
        print(f"   Experimento: {config.get('experiment', {}).get('name', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Error en config loader: {e}")
        return False


def print_summary(results):
    """Imprime resumen final."""
    print_section("RESUMEN DE VERIFICACIÓN")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\n📊 Resultados:")
    print(f"   ✅ Exitosos: {passed}/{total}")
    print(f"   ❌ Fallidos:  {failed}/{total}")

    if all(results.values()):
        print(f"\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print(f"\nPuedes comenzar a entrenar con:")
        print(f"   python main.py train --config debug")
        print(f"   python main.py train --config aisegment_quick")
        return 0
    elif passed > total / 2:
        print(f"\n⚠️  Sistema funcional con advertencias")
        print(f"\nRevisa los componentes marcados con ❌")
        print(f"El sistema debería funcionar para entrenamientos básicos")
        return 1
    else:
        print(f"\n❌ Sistema requiere configuración adicional")
        print(f"\nRevisa los componentes fallidos antes de entrenar")
        return 2


def main():
    """Función principal."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  VERIFICACIÓN DEL SISTEMA - U-Net Background Removal                 ║
║  Proyecto: Computer Vision - PUCP                                    ║
║  Autores: Luis Huacho y Dominick Alvarez                            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    results = {
        'Dependencias': check_dependencies(),
        'Estructura': check_project_structure(),
        'Configs': check_config_files(),
        'Datasets': check_datasets(),
        'Módulos': check_project_modules(),
        'Kaggle API': check_kaggle_api(),
        'Config Loader': test_config_loader()
    }

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
