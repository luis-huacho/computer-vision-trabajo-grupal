import torch
import numpy as np
import cv2
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class ModelInference:
    """
    Clase para realizar inferencia con el modelo entrenado.
    Incluye capacidades de harmonización usando módulos externos.
    """

    def __init__(self, segmentation_model_path, harmonization_model_path=None, device='cuda'):
        self.device = device

        # Import ImageProcessor desde utils
        try:
            from utils import ImageProcessor
            self.processor = ImageProcessor()
        except ImportError:
            print("⚠️  ImageProcessor no disponible desde utils")
            self.processor = None

        # Cargar modelo de segmentación
        try:
            from models import UNetAutoencoder
            self.segmentation_model = UNetAutoencoder(pretrained=False, use_attention=True)
            checkpoint = torch.load(segmentation_model_path, map_location=device)
            self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])
            self.segmentation_model.to(device)
            self.segmentation_model.eval()
            print(f"✅ Modelo de segmentación cargado: {segmentation_model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo de segmentación: {e}")
            self.segmentation_model = None

        # Cargar modelo de harmonización si está disponible
        self.harmonization_inference = None
        try:
            from harmonization import HarmonizationInference
            if harmonization_model_path and os.path.exists(harmonization_model_path):
                self.harmonization_inference = HarmonizationInference(harmonization_model_path, device)
                print(f"✅ Modelo de harmonización cargado: {harmonization_model_path}")
            elif harmonization_model_path:
                print(f"⚠️  Modelo de harmonización especificado pero no encontrado: {harmonization_model_path}")
        except ImportError:
            print("⚠️  Módulo de harmonización no disponible")

    def remove_background(self, image_path, output_path=None, image_size=384):
        """
        Remueve el fondo de una imagen manteniendo las dimensiones originales.
        """
        if self.segmentation_model is None:
            print("❌ Modelo de segmentación no disponible")
            return None

        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        # Usar redimensionamiento con padding si está disponible
        if self.processor:
            dummy_mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
            image_processed, _, restore_metadata = self.processor.resize_with_padding(
                image, dummy_mask, image_size
            )
        else:
            # Fallback simple
            image_processed = cv2.resize(image, (image_size, image_size))
            restore_metadata = None

        # Normalizar
        image_normalized = image_processed.astype(np.float32) / 255.0

        # Convertir a tensor
        input_tensor = torch.FloatTensor(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)
            output = output.squeeze(0).cpu().numpy()

        # Post-procesamiento
        rgb_channels = output[:3].transpose(1, 2, 0)
        alpha_channel = output[3]

        # Restaurar al tamaño original usando metadatos si están disponibles
        if self.processor and restore_metadata:
            rgb_restored, alpha_restored = self.processor.restore_original_size(
                rgb_channels, alpha_channel, restore_metadata
            )
        else:
            # Fallback simple
            rgb_restored = cv2.resize(rgb_channels, (original_size[1], original_size[0]))
            alpha_restored = cv2.resize(alpha_channel, (original_size[1], original_size[0]))

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

    def harmonize_composition(self, foreground_rgba, background_rgb, output_path=None):
        """
        Armoniza una composición de foreground + background.
        """
        if self.harmonization_inference is None:
            print("⚠️  Modelo de harmonización no disponible. Usando composición básica.")
            if self.processor:
                return self.processor.composite_foreground_background(foreground_rgba, background_rgb)
            else:
                # Fallback muy básico
                return self._simple_composite(foreground_rgba, background_rgb)

        # Crear composición inicial
        if self.processor:
            composite = self.processor.composite_foreground_background(foreground_rgba, background_rgb)
        else:
            composite = self._simple_composite(foreground_rgba, background_rgb)

        # Usar el módulo de harmonización
        harmonized = self.harmonization_inference.harmonize_composition(composite, output_path)

        return harmonized

    def _simple_composite(self, foreground_rgba, background_rgb):
        """Composición básica sin alpha blending sofisticado."""
        # Asegurar que las dimensiones coincidan
        if foreground_rgba.shape[:2] != background_rgb.shape[:2]:
            h, w = background_rgb.shape[:2]
            foreground_rgba = cv2.resize(foreground_rgba, (w, h))

        # Extraer componentes
        fg_rgb = foreground_rgba[:, :, :3].astype(np.float32) / 255.0
        alpha = foreground_rgba[:, :, 3:4].astype(np.float32) / 255.0
        bg_rgb = background_rgb.astype(np.float32) / 255.0

        # Alpha blending básico
        composite = fg_rgb * alpha + bg_rgb * (1 - alpha)

        # Convertir de vuelta a uint8
        composite = (composite * 255).astype(np.uint8)

        return composite

    def remove_background_and_harmonize(self, image_path, background_path, output_path=None, image_size=384):
        """
        Proceso completo: remoción de fondo + harmonización con nuevo fondo.
        """
        # Paso 1: Remover fondo
        foreground_rgba = self.remove_background(image_path, image_size=image_size)
        if foreground_rgba is None:
            return None

        # Paso 2: Cargar nuevo fondo
        background = cv2.imread(background_path)
        if background is None:
            raise ValueError(f"No se pudo cargar el fondo: {background_path}")

        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        # Redimensionar fondo al tamaño del foreground
        background = cv2.resize(background, (foreground_rgba.shape[1], foreground_rgba.shape[0]))

        # Paso 3: Harmonizar composición
        result = self.harmonize_composition(foreground_rgba, background, output_path)

        return result

    def batch_process(self, input_dir, output_dir):
        """
        Procesa un directorio completo de imágenes.
        """
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        successful = 0
        failed = 0

        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_no_bg.png")

            try:
                result = self.remove_background(input_path, output_path)
                if result is not None:
                    print(f"✅ Procesado: {img_file}")
                    successful += 1
                else:
                    print(f"❌ Error procesando: {img_file}")
                    failed += 1
            except Exception as e:
                print(f"❌ Error procesando {img_file}: {e}")
                failed += 1

        print(f"\n📊 Resumen del procesamiento en lote:")
        print(f"   ✅ Exitosos: {successful}")
        print(f"   ❌ Fallidos: {failed}")
        print(f"   📁 Directorio de salida: {output_dir}")


def demo_inference():
    """
    Función de demostración para usar el modelo entrenado.
    """
    print("🎬 DEMO DE INFERENCIA")
    print("=" * 30)

    # Paths de modelos
    try:
        from settings import SegmentationConfig, HarmonizationConfig
        segmentation_model_path = SegmentationConfig.BEST_MODEL_PATH
        harmonization_model_path = HarmonizationConfig.BEST_MODEL_PATH
    except ImportError:
        segmentation_model_path = 'checkpoints/best_segmentation.pth'
        harmonization_model_path = 'checkpoints/best_harmonizer.pth'

    if not os.path.exists(segmentation_model_path):
        print("❌ Modelo de segmentación entrenado no encontrado.")
        print(f"   Esperado en: {segmentation_model_path}")
        print("   Primero ejecuta el entrenamiento.")
        return

    # Crear objeto de inferencia
    print("🔄 Inicializando inferencia...")
    inference = ModelInference(
        segmentation_model_path=segmentation_model_path,
        harmonization_model_path=harmonization_model_path
    )

    # Ejemplo de uso básico - solo remoción de fondo
    input_image = 'example_input.jpg'
    output_image = 'example_output.png'

    print(f"\n📝 Ejemplo 1: Remoción de fondo")
    if os.path.exists(input_image):
        print(f"   🔄 Procesando: {input_image}")
        result = inference.remove_background(input_image, output_image)
        if result is not None:
            print(f"   ✅ Fondo removido exitosamente")
            print(f"   💾 Resultado guardado en: {output_image}")
            print(f"   📊 Dimensiones del resultado: {result.shape}")
        else:
            print(f"   ❌ Error en el procesamiento")
    else:
        print(f"   ⚠️  Imagen de ejemplo no encontrada: {input_image}")
        print(f"   💡 Coloca una imagen con este nombre para probar")

    # Ejemplo de uso avanzado - remoción + harmonización
    background_image = 'example_background.jpg'
    harmonized_output = 'example_harmonized.jpg'

    print(f"\n📝 Ejemplo 2: Remoción + Harmonización")
    if os.path.exists(input_image) and os.path.exists(background_image):
        print(f"   🔄 Procesando: {input_image} + {background_image}")
        harmonized_result = inference.remove_background_and_harmonize(
            input_image, background_image, harmonized_output
        )
        if harmonized_result is not None:
            print(f"   ✅ Imagen harmonizada exitosamente")
            print(f"   💾 Resultado guardado en: {harmonized_output}")
            print(f"   📊 Dimensiones: {harmonized_result.shape}")
        else:
            print(f"   ❌ Error en el procesamiento")
    else:
        missing_files = []
        if not os.path.exists(input_image):
            missing_files.append(input_image)
        if not os.path.exists(background_image):
            missing_files.append(background_image)

        print(f"   ⚠️  Archivos faltantes para harmonización:")
        for file in missing_files:
            print(f"      - {file}")
        print(f"   💡 Coloca estos archivos para probar la harmonización")

    # Ejemplo de procesamiento en lote
    batch_input_dir = 'input_images'
    batch_output_dir = 'output_images'

    print(f"\n📝 Ejemplo 3: Procesamiento en lote")
    if os.path.exists(batch_input_dir) and len(os.listdir(batch_input_dir)) > 0:
        print(f"   🔄 Procesando directorio: {batch_input_dir}")
        inference.batch_process(batch_input_dir, batch_output_dir)
    else:
        print(f"   ⚠️  Directorio de entrada no encontrado o vacío: {batch_input_dir}")
        print(f"   💡 Crea el directorio y coloca imágenes para procesamiento en lote")

    print(f"\n🎉 Demo de inferencia completado!")
    print(f"💡 Consejos:")
    print(f"   • Usa imágenes con personas claramente visibles")
    print(f"   • Buena iluminación mejora los resultados")
    print(f"   • Fondos contrastantes funcionan mejor")


def create_inference_examples():
    """
    Crea ejemplos de archivos para demostrar la inferencia.
    """
    print("🛠️ CREANDO EJEMPLOS PARA INFERENCIA")
    print("=" * 40)

    # Crear directorio de ejemplos
    os.makedirs('examples', exist_ok=True)
    os.makedirs('input_images', exist_ok=True)

    print("✅ Directorios de ejemplos creados:")
    print("   📁 examples/ - Para archivos de ejemplo")
    print("   📁 input_images/ - Para procesamiento en lote")

    # Crear archivo README para ejemplos
    readme_content = """# Ejemplos de Inferencia

## Archivos necesarios para demos:

### Demo básico:
- `example_input.jpg` - Imagen con persona para procesar
- `example_background.jpg` - Fondo nuevo para harmonización

### Procesamiento en lote:
- `input_images/` - Directorio con múltiples imágenes

## Resultados esperados:
- `example_output.png` - Imagen sin fondo (RGBA)
- `example_harmonized.jpg` - Imagen harmonizada con nuevo fondo
- `output_images/` - Resultados del procesamiento en lote

## Consejos:
1. Usa imágenes con personas claramente visibles
2. Buena iluminación mejora los resultados
3. Fondos contrastantes funcionan mejor
4. Resolución mínima recomendada: 300x300 píxeles
"""

    with open('examples/README.md', 'w') as f:
        f.write(readme_content)

    print("✅ Archivo README.md creado en examples/")
    print("💡 Coloca tus imágenes de prueba según las instrucciones")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'demo':
            demo_inference()
        elif mode == 'setup':
            create_inference_examples()
        elif mode == 'test':
            # Prueba básica del módulo
            print("=== PRUEBA DEL MÓDULO DE INFERENCIA ===")

            # Verificar si hay modelos disponibles
            segmentation_path = 'checkpoints/best_segmentation.pth'
            harmonization_path = 'checkpoints/best_harmonizer.pth'

            if os.path.exists(segmentation_path):
                print("✅ Modelo de segmentación encontrado")
                try:
                    inference = ModelInference(segmentation_path, harmonization_path)
                    print("✅ Inferencia inicializada correctamente")
                except Exception as e:
                    print(f"❌ Error inicializando inferencia: {e}")
            else:
                print(f"❌ Modelo de segmentación no encontrado: {segmentation_path}")
                print("   Primero entrena el modelo")
        else:
            print(f"❌ Modo no reconocido: {mode}")
            print("Modos disponibles: demo, setup, test")
    else:
        # Modo por defecto: demo
        demo_inference()