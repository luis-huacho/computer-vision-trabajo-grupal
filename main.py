#!/usr/bin/env python3
"""
U-Net Background Removal with Harmonization - Main Entry Point

Sistema modular para remoción de fondo y harmonización de imágenes usando U-Net.
Desarrollado por Luis Huacho y Dominick Alvarez - Maestría en Informática, PUCP.

Uso:
    python main.py                    # Menú interactivo
    python main.py segmentation       # Entrenar segmentación
    python main.py harmonization      # Entrenar harmonización
    python main.py demo               # Demo de inferencia
    python main.py setup              # Configurar datasets
    python main.py config             # Mostrar configuraciones
    python main.py verify             # Verificar sistema
    python main.py status             # Estado de módulos
    python main.py help               # Mostrar ayuda
"""

import sys
import os


# ============================================================================
# VERIFICACIÓN DE MÓDULOS
# ============================================================================

def check_modules():
    """Verifica qué módulos están disponibles y retorna su estado."""
    modules = {}

    # Lista de módulos del sistema
    module_list = [
        'settings', 'utils', 'models', 'datasets',
        'trainer', 'inference', 'harmonization'
    ]

    for module_name in module_list:
        try:
            __import__(module_name)
            modules[module_name] = True
        except ImportError:
            modules[module_name] = False

    return modules


def print_system_status():
    """Muestra el estado completo del sistema."""
    modules = check_modules()

    print("🎭 SISTEMA U-NET CON HARMONIZACIÓN")
    print("=" * 50)
    print("📦 Estado de módulos:")

    status_icons = {True: "✅", False: "❌"}

    for module, available in modules.items():
        icon = status_icons[available]
        print(f"   {icon} {module}.py")

    # Determinar funcionalidades disponibles
    can_train_seg = all([modules['models'], modules['datasets'], modules['trainer']])
    can_train_harm = modules['harmonization']
    can_demo = modules['inference']
    can_verify = modules['utils']
    can_config = modules['settings']

    print(f"\n🎯 Funcionalidades disponibles:")
    print(f"   {status_icons[can_train_seg]} Entrenamiento de segmentación")
    print(f"   {status_icons[can_train_harm]} Entrenamiento de harmonización")
    print(f"   {status_icons[can_demo]} Inferencia y demos")
    print(f"   {status_icons[can_verify]} Verificación del sistema")
    print(f"   {status_icons[can_config]} Configuraciones centralizadas")

    return modules


# ============================================================================
# FUNCIONES DE EJECUCIÓN
# ============================================================================

def train_segmentation():
    """Ejecuta el entrenamiento de segmentación usando multi-GPU si está disponible."""
    try:
        import torch
        import subprocess
        import os
        
        print("🔄 ENTRENAMIENTO DE SEGMENTACIÓN (MULTI-GPU)")
        print("=" * 40)
        
        # Verificar disponibilidad de GPUs
        if not torch.cuda.is_available():
            print("❌ CUDA no disponible. Usando CPU (no recomendado)...")
            # Fallback a entrenamiento tradicional
            try:
                from trainer import train_segmentation as train_seg
                from settings import get_segmentation_config
                config = get_segmentation_config()
                success = train_seg(config)
                if success:
                    print("✅ Entrenamiento completado exitosamente!")
                else:
                    print("❌ Error en el entrenamiento")
            except ImportError as e:
                print(f"❌ Módulos necesarios no disponibles: {e}")
            return
            
        gpu_count = torch.cuda.device_count()
        print(f"🔍 GPUs detectadas: {gpu_count}")
        
        # Verificar si trainer.py soporta DDP
        if not os.path.exists("trainer.py"):
            print("❌ trainer.py no encontrado")
            return
            
        # Leer trainer.py para verificar si tiene soporte DDP
        with open("trainer.py", "r") as f:
            trainer_content = f.read()
            
        has_ddp_support = ("torch.distributed" in trainer_content and 
                          "DistributedDataParallel" in trainer_content and
                          "RANK" in trainer_content)
        
        if has_ddp_support and gpu_count >= 1:
            # Usar entrenamiento distribuido
            if gpu_count < 2:
                print("⚠️  Solo se detectó 1 GPU. Usando entrenamiento distribuido con 1 GPU...")
                nproc = 1
            else:
                nproc = min(gpu_count, 2)  # Usar máximo 2 GPUs
                print(f"🚀 Usando {nproc} GPUs para entrenamiento distribuido")
            
            # Comando torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                "trainer.py"
            ]
            
            print(f"💻 Ejecutando comando: {' '.join(cmd)}")
            print("⏳ Iniciando entrenamiento distribuido...")
            print("-" * 40)
            
            # Ejecutar torchrun
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            print("-" * 40)
            if result.returncode == 0:
                print("✅ Entrenamiento de segmentación completado exitosamente!")
            else:
                print(f"❌ Error en el entrenamiento. Código de salida: {result.returncode}")
                
        else:
            # Fallback a entrenamiento tradicional
            if not has_ddp_support:
                print("⚠️  trainer.py no tiene soporte DDP. Usando entrenamiento tradicional...")
            
            from trainer import train_segmentation as train_seg
            from settings import get_segmentation_config
            
            config = get_segmentation_config()
            success = train_seg(config)
            
            if success:
                print("✅ Entrenamiento completado exitosamente!")
            else:
                print("❌ Error en el entrenamiento")

    except ImportError as e:
        print(f"❌ Módulos necesarios no disponibles: {e}")
        print("   Necesarios: settings.py, trainer.py, models.py, datasets.py")
    except FileNotFoundError:
        print("❌ 'torchrun' no encontrado. Asegúrate de tener PyTorch instalado correctamente.")
        print("💡 Instala PyTorch con: pip install torch torchvision")
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento: {e}")


def train_harmonization():
    """Ejecuta el entrenamiento de harmonización usando torchrun para multi-GPU."""
    try:
        import subprocess
        import torch
        
        print("🎨 ENTRENAMIENTO DE HARMONIZACIÓN (MULTI-GPU)")
        print("=" * 40)
        
        # Verificar disponibilidad de GPUs
        if not torch.cuda.is_available():
            print("❌ CUDA no disponible. Se requiere GPU para entrenamiento distribuido.")
            return
            
        gpu_count = torch.cuda.device_count()
        print(f"🔍 GPUs detectadas: {gpu_count}")
        
        if gpu_count < 2:
            print("⚠️  Solo se detectó 1 GPU. Usando entrenamiento distribuido con 1 GPU...")
            nproc = 1
        else:
            nproc = min(gpu_count, 2)  # Usar máximo 2 GPUs
            print(f"🚀 Usando {nproc} GPUs para entrenamiento distribuido")
        
        # Comando torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "harmonization.py"
        ]
        
        print(f"💻 Ejecutando comando: {' '.join(cmd)}")
        print("⏳ Iniciando entrenamiento distribuido...")
        print("-" * 40)
        
        # Ejecutar torchrun
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        print("-" * 40)
        if result.returncode == 0:
            print("✅ Entrenamiento de harmonización completado exitosamente!")
        else:
            print(f"❌ Error en el entrenamiento. Código de salida: {result.returncode}")

    except ImportError as e:
        print(f"❌ Módulos necesarios no disponibles: {e}")
    except FileNotFoundError:
        print("❌ 'torchrun' no encontrado. Asegúrate de tener PyTorch instalado correctamente.")
        print("💡 Instala PyTorch con: pip install torch torchvision")
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento distribuido: {e}")


def run_demo():
    """Ejecuta el demo de inferencia."""
    try:
        from inference import demo_inference

        print("🎬 DEMO DE INFERENCIA")
        print("=" * 40)

        demo_inference()

    except ImportError as e:
        print(f"❌ Módulo inference.py no disponible: {e}")


def setup_system():
    """Configura datasets y ejemplos."""
    try:
        print("🛠️ CONFIGURACIÓN DEL SISTEMA")
        print("=" * 40)

        # Setup de harmonización si está disponible
        try:
            from harmonization import create_sample_harmonization_dataset
            print("📊 Configurando dataset de harmonización...")
            create_sample_harmonization_dataset()
        except ImportError:
            print("⚠️  Módulo harmonization.py no disponible para setup")

        # Setup de ejemplos de inferencia si está disponible
        try:
            from inference import create_inference_examples
            print("🎬 Configurando ejemplos de inferencia...")
            create_inference_examples()
        except ImportError:
            print("⚠️  Módulo inference.py no disponible para ejemplos")

        print("✅ Configuración completada")

    except Exception as e:
        print(f"❌ Error en la configuración: {e}")


def verify_system():
    """Verifica el sistema completo."""
    try:
        print("🔍 VERIFICACIÓN DEL SISTEMA")
        print("=" * 40)

        tests = []

        # Verificación de COCO
        try:
            from utils import quick_coco_test
            tests.append(("Estructura COCO", quick_coco_test))
        except ImportError:
            print("⚠️  utils.py no disponible para verificar COCO")

        # Verificación de modelos
        try:
            from models import test_model_forward
            tests.append(("Forward pass modelos", test_model_forward))
        except ImportError:
            print("⚠️  models.py no disponible para verificar modelos")

        # Verificación de procesamiento
        try:
            from utils import test_image_processing
            tests.append(("Procesamiento imágenes", test_image_processing))
        except ImportError:
            pass

        # Verificación de datasets
        try:
            from datasets import test_coco_dataset
            tests.append(("Dataset COCO", test_coco_dataset))
        except ImportError:
            print("⚠️  datasets.py no disponible para verificar dataset")

        # Ejecutar tests disponibles
        if tests:
            results = {}
            for test_name, test_func in tests:
                print(f"\n📋 Verificando {test_name}...")
                try:
                    results[test_name] = test_func()
                except Exception as e:
                    print(f"❌ Error: {e}")
                    results[test_name] = False

            # Resumen
            print(f"\n📊 RESUMEN DE VERIFICACIÓN:")
            all_passed = True
            for test_name, passed in results.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
                if not passed:
                    all_passed = False

            if all_passed:
                print(f"\n🎉 ¡Todas las verificaciones exitosas!")
            else:
                print(f"\n⚠️  Algunas verificaciones fallaron")
        else:
            print("❌ No hay módulos disponibles para verificación")

    except Exception as e:
        print(f"❌ Error en la verificación: {e}")


def show_config():
    """Muestra las configuraciones del sistema."""
    try:
        from settings import print_config_summary

        print("📋 CONFIGURACIONES DEL SISTEMA")
        print("=" * 40)

        print_config_summary()

    except ImportError:
        print("❌ Módulo settings.py no disponible")


# ============================================================================
# MENÚ INTERACTIVO
# ============================================================================

def interactive_menu():
    """Menú interactivo principal."""
    modules = print_system_status()

    print(f"\n📋 MENÚ PRINCIPAL")
    print("Selecciona una opción:")

    # Definir opciones disponibles
    options = []

    # Opción 1: Segmentación
    if all([modules['models'], modules['datasets'], modules['trainer']]):
        print("1. Entrenar modelo de segmentación")
        options.append(('1', train_segmentation))
    else:
        print("1. [DESHABILITADO] Entrenar segmentación (faltan módulos)")

    # Opción 2: Harmonización
    if modules['harmonization']:
        print("2. Entrenar modelo de harmonización")
        options.append(('2', train_harmonization))
    else:
        print("2. [DESHABILITADO] Entrenar harmonización (falta harmonization.py)")

    # Opción 3: Demo
    if modules['inference']:
        print("3. Ejecutar demo de inferencia")
        options.append(('3', run_demo))
    else:
        print("3. [DESHABILITADO] Demo (falta inference.py)")

    # Opción 4: Setup
    print("4. Configurar sistema y datasets")
    options.append(('4', setup_system))

    # Opción 5: Verificación
    if modules['utils']:
        print("5. Verificar sistema completo")
        options.append(('5', verify_system))
    else:
        print("5. [DESHABILITADO] Verificación (falta utils.py)")

    # Opción 6: Configuraciones
    if modules['settings']:
        print("6. Mostrar configuraciones")
        options.append(('6', show_config))
    else:
        print("6. [DESHABILITADO] Configuraciones (falta settings.py)")

    # Opciones siempre disponibles
    print("7. Mostrar estado de módulos")
    options.append(('7', lambda: print_system_status()))

    print("8. Ayuda")
    options.append(('8', show_help))

    print("9. Salir")
    options.append(('9', lambda: exit_program()))

    # Procesar selección
    try:
        choice = input(f"\nSelecciona opción (1-9): ").strip()

        # Buscar y ejecutar opción
        for opt_num, func in options:
            if choice == opt_num:
                print()  # Línea en blanco
                func()
                print()  # Línea en blanco
                input("Presiona Enter para continuar...")
                interactive_menu()  # Volver al menú
                return

        # Si llegamos aquí, opción no válida
        print("❌ Opción no válida")
        input("Presiona Enter para continuar...")
        interactive_menu()

    except (KeyboardInterrupt, EOFError):
        exit_program()


def exit_program():
    """Sale del programa."""
    print("👋 ¡Hasta luego!")
    sys.exit(0)


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def show_help():
    """Muestra la ayuda completa."""
    print(__doc__)
    print("\nEjemplos de uso:")
    print("  python main.py                    # Menú interactivo")
    print("  python main.py segmentation       # Entrenar segmentación directamente")
    print("  python main.py harmonization      # Entrenar harmonización directamente")
    print("  python main.py demo               # Ejecutar demo")
    print("  python main.py setup              # Configurar datasets")
    print("  python main.py verify             # Verificar sistema")
    print("  python main.py config             # Mostrar configuraciones")
    print("  python main.py status             # Estado de módulos")


def initialize_system():
    """Inicializa el sistema según módulos disponibles."""
    print("🚀 INICIALIZANDO SISTEMA U-NET")
    print("=" * 40)

    # Usar settings si está disponible
    try:
        from settings import initialize_experiment
        initialize_experiment()
        return True
    except ImportError:
        print("⚠️  Módulo settings.py no disponible")
        print("🔧 Usando configuración básica...")

        # Crear directorios básicos
        basic_dirs = [
            'checkpoints', 'logs', 'plots', 'examples',
            'dataset', 'dataset/foregrounds', 'dataset/backgrounds',
            'input_images', 'output_images'
        ]

        for dir_path in basic_dirs:
            os.makedirs(dir_path, exist_ok=True)

        print("✅ Directorios básicos creados")
        print("💡 Para funcionalidad completa, asegúrate de tener settings.py")
        return False


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del sistema."""
    # Inicializar sistema
    settings_available = initialize_system()
    print()  # Línea en blanco

    # Procesar argumentos de línea de comandos
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        # Mapeo de modos a funciones
        mode_functions = {
            'segmentation': train_segmentation,
            'harmonization': train_harmonization,
            'demo': run_demo,
            'setup': setup_system,
            'verify': verify_system,
            'config': show_config,
            'status': print_system_status,
            'help': show_help,
            'menu': interactive_menu
        }

        if mode in mode_functions:
            mode_functions[mode]()
        else:
            print(f"❌ Modo no reconocido: {mode}")
            print("💡 Usa 'python main.py help' para ver opciones disponibles")
            sys.exit(1)
    else:
        # Modo por defecto: menú interactivo
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Programa cancelado por el usuario!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("💡 Usa 'python main.py help' para obtener ayuda")
        sys.exit(1)