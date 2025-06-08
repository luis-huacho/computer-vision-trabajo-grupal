import os
import sys
import subprocess
import argparse
import time
from datetime import datetime


class Colors:
    """Códigos de color para terminal."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def print_colored(message, color=Colors.NC, prefix=""):
    """Imprime mensaje con color."""
    print(f"{color}{prefix}{message}{Colors.NC}")


def print_info(message):
    """Imprime mensaje de información."""
    print_colored(message, Colors.BLUE, "[INFO] ")


def print_success(message):
    """Imprime mensaje de éxito."""
    print_colored(message, Colors.GREEN, "[SUCCESS] ")


def print_warning(message):
    """Imprime mensaje de advertencia."""
    print_colored(message, Colors.YELLOW, "[WARNING] ")


def print_error(message):
    """Imprime mensaje de error."""
    print_colored(message, Colors.RED, "[ERROR] ")


def get_log_filename(base_name):
    """Genera nombre de archivo de log con timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.log"


def clean_logs():
    """Limpia los archivos de log."""
    # Generar nombres de archivos con timestamp
    output_log = get_log_filename('output')
    errors_log = get_log_filename('errors')

    print_info(f"Archivos de log: {output_log}, {errors_log}")

    try:
        # Crear archivos vacíos
        with open(output_log, 'w') as f:
            pass
        with open(errors_log, 'w') as f:
            pass
        print_success("Archivos de log creados")
        return output_log, errors_log
    except Exception as e:
        print_error(f"Error creando archivos de log: {e}")
        return None, None


def show_system_info():
    """Muestra información del sistema."""
    print_info("Información del sistema:")
    print(f"  - Python: {sys.version.split()[0]}")
    print(f"  - Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Directorio: {os.getcwd()}")

    # Información de GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  - GPU: {gpu_count} dispositivo(s) CUDA")
            for i in range(gpu_count):
                print(f"    - {torch.cuda.get_device_name(i)}")
        else:
            print("  - GPU: CPU mode")
    except ImportError:
        print("  - GPU: No se pudo verificar")

    print()


def run_training(output_log, errors_log, verbose=False):
    """Ejecuta el entrenamiento."""
    print_info("Iniciando entrenamiento del modelo U-Net...")
    print_info(f"Output log: {output_log}")
    print_info(f"Errors log: {errors_log}")

    if not verbose:
        print_info(f"Monitorear progreso: tail -f {output_log}")

    print()

    try:
        if verbose:
            # Ejecutar con logs en tiempo real
            print_info("Modo verbose activado - mostrando logs en tiempo real")
            print_colored("=" * 60, Colors.CYAN)

            # Usar subprocess con stdout/stderr combinados
            process = subprocess.Popen(
                [sys.executable, 'main.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Guardar output y mostrarlo en tiempo real
            with open(output_log, 'w') as log_file:
                for line in process.stdout:
                    print(line.rstrip())
                    log_file.write(line)
                    log_file.flush()

            # Esperar a que termine
            return_code = process.wait()

        else:
            # Ejecutar en background con redirección
            with open(output_log, 'w') as out_file, open(errors_log, 'w') as err_file:
                result = subprocess.run(
                    [sys.executable, 'main.py'],
                    stdout=out_file,
                    stderr=err_file
                )
                return_code = result.returncode

        # Verificar resultado
        if return_code == 0:
            print_success("Entrenamiento completado exitosamente!")
            return True
        else:
            print_error(f"El entrenamiento falló con código de salida: {return_code}")
            print_info(f"Revisa {errors_log} para más detalles")
            return False

    except KeyboardInterrupt:
        print_warning("\nEntrenamiento cancelado por el usuario")
        return False
    except Exception as e:
        print_error(f"Error ejecutando el entrenamiento: {e}")
        return False


def move_results_to_timestamped_dirs():
    """Mueve los resultados a subdirectorios con timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print_info("Organizando resultados del entrenamiento...")

    # Mover plots
    plots_dir = 'plots'
    plots_timestamp_dir = os.path.join(plots_dir, timestamp)

    if os.path.exists(plots_dir):
        # Obtener archivos en plots (excluyendo subdirectorios)
        plot_files = [f for f in os.listdir(plots_dir)
                      if os.path.isfile(os.path.join(plots_dir, f))
                      and f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg'))]

        if plot_files:
            try:
                os.makedirs(plots_timestamp_dir, exist_ok=True)
                moved_plots = 0
                for file in plot_files:
                    src = os.path.join(plots_dir, file)
                    dst = os.path.join(plots_timestamp_dir, file)
                    os.rename(src, dst)
                    moved_plots += 1
                print_success(f"Movidos {moved_plots} archivos de plots a: plots/{timestamp}/")
            except Exception as e:
                print_error(f"Error moviendo plots: {e}")
        else:
            print_info("No se encontraron archivos de plots para mover")

    # Mover checkpoints
    checkpoints_dir = 'checkpoints'
    checkpoints_timestamp_dir = os.path.join(checkpoints_dir, timestamp)

    if os.path.exists(checkpoints_dir):
        # Obtener archivos en checkpoints (excluyendo subdirectorios)
        checkpoint_files = [f for f in os.listdir(checkpoints_dir)
                            if os.path.isfile(os.path.join(checkpoints_dir, f))
                            and f.lower().endswith(('.pth', '.pt', '.ckpt'))]

        if checkpoint_files:
            try:
                os.makedirs(checkpoints_timestamp_dir, exist_ok=True)
                moved_checkpoints = 0
                for file in checkpoint_files:
                    src = os.path.join(checkpoints_dir, file)
                    dst = os.path.join(checkpoints_timestamp_dir, file)
                    os.rename(src, dst)
                    moved_checkpoints += 1
                print_success(f"Movidos {moved_checkpoints} checkpoints a: checkpoints/{timestamp}/")
            except Exception as e:
                print_error(f"Error moviendo checkpoints: {e}")
        else:
            print_info("No se encontraron checkpoints para mover")

    return timestamp


def show_summary(output_log, errors_log):
    """Muestra resumen del entrenamiento."""
    print_info("Resumen del entrenamiento:")

    # Mostrar tamaño de archivos de log
    for log_file in [output_log, errors_log]:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            size_mb = size / (1024 * 1024)
            print(f"  - {log_file}: {size_mb:.2f} MB")

            # Verificar si hay contenido en errors.log
            if 'errors' in log_file and size > 0:
                print_warning("Se encontraron errores en el log de errores")
                print("  - Últimas líneas:")
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-5:]:
                            print(f"    {line.rstrip()}")
                except:
                    pass
            elif 'errors' in log_file:
                print_success("No se encontraron errores")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Script para ejecutar el entrenamiento del modelo U-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_training.py                    # Ejecutar entrenamiento normal
  python run_training.py --verbose         # Mostrar logs en tiempo real
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar logs en tiempo real')
    parser.add_argument('--no-color', action='store_true',
                        help='Desactivar colores en la salida')

    args = parser.parse_args()

    # Desactivar colores si se solicita
    if args.no_color:
        Colors.RED = Colors.GREEN = Colors.YELLOW = Colors.BLUE = Colors.CYAN = Colors.NC = ''

    # Header
    print_colored("=" * 50, Colors.CYAN)
    print_colored("   U-Net Training Script", Colors.CYAN)
    print_colored("=" * 50, Colors.CYAN)
    print()

    # Ejecutar pasos principales
    show_system_info()

    # Crear archivos de log con timestamp
    output_log, errors_log = clean_logs()
    if not output_log or not errors_log:
        print_error("No se pudieron crear los archivos de log")
        sys.exit(1)

    print()
    if not args.verbose:
        print_info("Presiona Ctrl+C para cancelar el entrenamiento")
        time.sleep(2)

    # Ejecutar entrenamiento
    success = run_training(output_log, errors_log, verbose=args.verbose)

    # Si el entrenamiento fue exitoso, organizar resultados
    if success:
        print()
        results_timestamp = move_results_to_timestamped_dirs()
        print_info(f"Resultados organizados en subdirectorios: {results_timestamp}")

    print()
    show_summary(output_log, errors_log)

    print()
    print_colored("=" * 50, Colors.CYAN)
    if success:
        print_success("Script completado exitosamente")
    else:
        print_error("Script completado con errores")
    print_colored("=" * 50, Colors.CYAN)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()