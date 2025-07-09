import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


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


class ImageProcessor:
    """
    Clase para manejar redimensionamiento consistente de imágenes.
    Mantiene proporciones y permite restaurar tamaño original.
    """

    @staticmethod
    def resize_with_padding(image, mask, target_size):
        """
        Redimensiona imagen y máscara manteniendo proporciones y agregando padding.

        Args:
            image: Imagen de entrada (H, W, C)
            mask: Máscara de entrada (H, W)
            target_size: Tamaño objetivo (int)

        Returns:
            tuple: (imagen_redimensionada, máscara_redimensionada, metadatos_para_restaurar)
        """
        h, w = image.shape[:2]

        # Calcular escala para mantener proporciones
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Redimensionar manteniendo proporciones
        if len(image.shape) == 3:
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if len(mask.shape) == 2:
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Calcular padding para centrar
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        pad_h_bottom = target_size - new_h - pad_h
        pad_w_right = target_size - new_w - pad_w

        # Aplicar padding
        if len(image.shape) == 3:
            image_padded = cv2.copyMakeBorder(
                image_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            image_padded = cv2.copyMakeBorder(
                image_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
                cv2.BORDER_CONSTANT, value=0
            )

        mask_padded = cv2.copyMakeBorder(
            mask_resized, pad_h, pad_h_bottom, pad_w, pad_w_right,
            cv2.BORDER_CONSTANT, value=0
        )

        # Metadatos para restaurar
        restore_metadata = {
            'original_size': (h, w),
            'scale': scale,
            'new_size': (new_h, new_w),
            'padding': (pad_h, pad_w, pad_h_bottom, pad_w_right)
        }

        return image_padded, mask_padded, restore_metadata

    @staticmethod
    def restore_original_size(image, mask, restore_metadata):
        """
        Restaura imagen y máscara a su tamaño original.

        Args:
            image: Imagen procesada (target_size, target_size, C)
            mask: Máscara procesada (target_size, target_size)
            restore_metadata: Metadatos del redimensionamiento

        Returns:
            tuple: (imagen_original, máscara_original)
        """
        pad_h, pad_w, pad_h_bottom, pad_w_right = restore_metadata['padding']
        new_h, new_w = restore_metadata['new_size']
        original_h, original_w = restore_metadata['original_size']

        # Remover padding
        if len(image.shape) == 3:
            image_unpadded = image[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :]
        else:
            image_unpadded = image[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        if len(mask.shape) == 2:
            mask_unpadded = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        else:
            mask_unpadded = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        # Redimensionar al tamaño original
        if len(image.shape) == 3:
            image_restored = cv2.resize(image_unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_restored = cv2.resize(image_unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        mask_restored = cv2.resize(mask_unpadded, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        return image_restored, mask_restored

    @staticmethod
    def composite_foreground_background(foreground_rgba, background_rgb):
        """
        Compone foreground RGBA sobre background RGB usando alpha blending.

        Args:
            foreground_rgba: Imagen RGBA de foreground (H, W, 4)
            background_rgb: Imagen RGB de background (H, W, 3)

        Returns:
            Imagen RGB compuesta (H, W, 3)
        """
        # Asegurar que las dimensiones coincidan
        if foreground_rgba.shape[:2] != background_rgb.shape[:2]:
            h, w = background_rgb.shape[:2]
            foreground_rgba = cv2.resize(foreground_rgba, (w, h))

        # Extraer componentes
        fg_rgb = foreground_rgba[:, :, :3].astype(np.float32) / 255.0
        alpha = foreground_rgba[:, :, 3:4].astype(np.float32) / 255.0
        bg_rgb = background_rgb.astype(np.float32) / 255.0

        # Alpha blending
        composite = fg_rgb * alpha + bg_rgb * (1 - alpha)

        # Convertir de vuelta a uint8
        composite = (composite * 255).astype(np.uint8)

        return composite


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
            return torch.tensor(1.0, device=pred.device)

        return intersection / union

    @staticmethod
    def calculate_dice(pred, target, threshold=0.5):
        """Calcula Dice Coefficient."""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        intersection = (pred_binary * target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()

        if total == 0:
            return torch.tensor(1.0, device=pred.device)

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

    def save_checkpoint(self, model, optimizer, epoch, loss, metrics, is_best=False, model_name='model'):
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
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)

        # Guardar mejor modelo
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'best_{model_name}.pth')
            torch.save(checkpoint, best_path)

        # Guardar último modelo
        last_path = os.path.join(self.checkpoint_dir, f'last_{model_name}.pth')
        torch.save(checkpoint, last_path)

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Carga checkpoint del modelo."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['loss'], checkpoint['metrics']


class LossCalculator:
    """
    Calculadora de pérdidas compuestas para el entrenamiento.
    Combina múltiples tipos de pérdida para mejor calidad.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.3):
        self.alpha = alpha  # BCE weight
        self.beta = beta  # Dice weight
        self.gamma = gamma  # Perceptual weight
        self.delta = delta  # Edge weight

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice Loss para segmentación."""
        # FIX: Usar .reshape() en lugar de .view() para evitar problemas de memoria no contigua
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

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
        pred_edges_x = torch.nn.functional.conv2d(pred[:, 3:4], sobel_x, padding=1)
        pred_edges_y = torch.nn.functional.conv2d(pred[:, 3:4], sobel_y, padding=1)
        target_edges_x = torch.nn.functional.conv2d(target[:, 3:4], sobel_x, padding=1)
        target_edges_y = torch.nn.functional.conv2d(target[:, 3:4], sobel_y, padding=1)

        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)

        return self.mse_loss(pred_edges, target_edges)

    def calculate_loss(self, pred, target):
        """Calcula la pérdida total compuesta."""
        # Separar canales RGB y Alpha
        pred_rgb = pred[:, :3]
        pred_alpha = pred[:, 3:4]
        target_rgb = target[:, :3]
        target_alpha = target[:, 3:4]

        # Asegurar que los tensores sean contiguos para evitar errores de view/reshape
        pred_alpha = pred_alpha.contiguous()
        target_alpha = target_alpha.contiguous()
        pred_rgb = pred_rgb.contiguous()
        target_rgb = target_rgb.contiguous()

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

        # Verificar que la pérdida sea válida
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Si hay problemas, usar solo BCE loss como fallback
            total_loss = bce_loss

        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            'perceptual_loss': perceptual_loss,
            'edge_loss': edge_loss
        }


def quick_coco_test():
    """
    Prueba rápida para verificar estructura COCO sin cargar todo el dataset.
    """
    print("=== VERIFICACIÓN RÁPIDA DE ESTRUCTURA COCO ===\n")

    coco_root = 'COCO'
    issues = []

    # Verificar directorio principal
    if not os.path.exists(coco_root):
        issues.append(f"❌ Directorio principal no encontrado: {coco_root}")
        print(f"❌ Directorio principal no encontrado: {coco_root}")
        return False
    else:
        print(f"✅ Directorio principal encontrado: {coco_root}")

    # Verificar annotations
    ann_dir = os.path.join(coco_root, 'annotations')
    if not os.path.exists(ann_dir):
        issues.append(f"❌ Directorio de anotaciones no encontrado: {ann_dir}")
    else:
        print(f"✅ Directorio de anotaciones encontrado: {ann_dir}")

        # Verificar archivos específicos
        train_ann = os.path.join(ann_dir, 'person_keypoints_train2017.json')
        val_ann = os.path.join(ann_dir, 'person_keypoints_val2017.json')

        if os.path.exists(train_ann):
            size_mb = os.path.getsize(train_ann) / (1024 * 1024)
            print(f"✅ Anotaciones de entrenamiento encontradas: {train_ann} ({size_mb:.1f} MB)")
        else:
            issues.append(f"❌ Anotaciones de entrenamiento no encontradas: {train_ann}")

        if os.path.exists(val_ann):
            size_mb = os.path.getsize(val_ann) / (1024 * 1024)
            print(f"✅ Anotaciones de validación encontradas: {val_ann} ({size_mb:.1f} MB)")
        else:
            issues.append(f"❌ Anotaciones de validación no encontradas: {val_ann}")

    # Verificar directorios de imágenes
    train_dir = os.path.join(coco_root, 'train2017')
    val_dir = os.path.join(coco_root, 'val2017')

    if os.path.exists(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        print(f"✅ Directorio train2017 encontrado con {train_count:,} imágenes")
    else:
        issues.append(f"❌ Directorio train2017 no encontrado: {train_dir}")

    if os.path.exists(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        print(f"✅ Directorio val2017 encontrado con {val_count:,} imágenes")
    else:
        issues.append(f"❌ Directorio val2017 no encontrado: {val_dir}")

    # Resumen
    if issues:
        print(f"\n❌ Se encontraron {len(issues)} problemas:")
        for issue in issues:
            print(f"   {issue}")
        print("\n📋 Instrucciones para resolver:")
        print("   1. Asegúrate de que el directorio COCO esté en la ubicación correcta")
        print("   2. Descomprime los archivos ZIP si es necesario:")
        print("      - annotations_trainval2017.zip -> COCO/annotations/")
        print("      - train2017.zip -> COCO/train2017/")
        print("      - val2017.zip -> COCO/val2017/")
        return False
    else:
        print(f"\n✅ Estructura COCO verificada correctamente!")
        print(f"   ✅ Todos los archivos necesarios están presentes")
        print(f"   ✅ Listo para entrenar el modelo")
        return True


def analyze_coco_annotations():
    """
    Analiza las anotaciones COCO para entender mejor el dataset.
    """
    print("=== ANÁLISIS DE ANOTACIONES COCO ===\n")

    coco_root = 'COCO'
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')

    if not os.path.exists(train_ann_file):
        print(f"❌ Archivo de anotaciones no encontrado: {train_ann_file}")
        return

    try:
        print("📊 Cargando anotaciones...")
        with open(train_ann_file, 'r') as f:
            coco_data = json.load(f)

        # Estadísticas básicas
        total_images = len(coco_data['images'])
        total_annotations = len(coco_data['annotations'])

        print(f"📈 Estadísticas generales:")
        print(f"   - Total de imágenes: {total_images:,}")
        print(f"   - Total de anotaciones: {total_annotations:,}")

        # Filtrar solo personas con keypoints
        person_annotations = [ann for ann in coco_data['annotations']
                              if 'keypoints' in ann and ann.get('area', 0) > 500]

        print(f"   - Anotaciones de personas válidas: {len(person_annotations):,}")

        # Agrupar por imagen
        image_to_annotations = {}
        for ann in person_annotations:
            img_id = ann['image_id']
            if img_id not in image_to_annotations:
                image_to_annotations[img_id] = []
            image_to_annotations[img_id].append(ann)

        valid_images = len(image_to_annotations)
        print(f"   - Imágenes con personas válidas: {valid_images:,}")

        # Distribución de personas por imagen
        persons_per_image = [len(anns) for anns in image_to_annotations.values()]
        avg_persons = np.mean(persons_per_image)
        max_persons = max(persons_per_image)

        print(f"   - Promedio de personas por imagen: {avg_persons:.2f}")
        print(f"   - Máximo de personas en una imagen: {max_persons}")

        # Distribución de tamaños
        areas = [ann['area'] for ann in person_annotations]
        print(f"📏 Distribución de tamaños (área):")
        print(f"   - Área promedio: {np.mean(areas):.0f} píxeles²")
        print(f"   - Área mínima: {np.min(areas):.0f} píxeles²")
        print(f"   - Área máxima: {np.max(areas):.0f} píxeles²")
        print(f"   - Mediana: {np.median(areas):.0f} píxeles²")

        print(f"\n✅ Análisis completado. Dataset listo para entrenamiento.")

    except Exception as e:
        print(f"❌ Error analizando anotaciones: {e}")


def test_image_processing():
    """
    Función de prueba para verificar el procesamiento de imágenes.
    """
    print("Probando procesamiento de imágenes...")

    processor = ImageProcessor()

    # Crear imagen de prueba con diferentes dimensiones
    test_cases = [
        (480, 640, 3),  # Horizontal
        (640, 480, 3),  # Vertical
        (512, 512, 3),  # Cuadrada
        (1080, 1920, 3)  # Muy alta resolución
    ]

    target_size = 384

    for i, (h, w, c) in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {h}x{w}x{c}")

        # Crear imagen y máscara de prueba
        test_image = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        test_mask = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255

        try:
            # Procesar
            image_processed, mask_processed, metadata = processor.resize_with_padding(
                test_image, test_mask, target_size
            )

            print(f"  ✓ Original: {test_image.shape} -> Procesada: {image_processed.shape}")
            print(f"  ✓ Máscara: {test_mask.shape} -> Procesada: {mask_processed.shape}")

            # Verificar dimensiones
            if image_processed.shape[:2] == (target_size, target_size):
                print(f"  ✓ Dimensiones correctas")
            else:
                print(f"  ✗ Dimensiones incorrectas: {image_processed.shape}")
                continue

            # Probar restauración
            image_restored, mask_restored = processor.restore_original_size(
                image_processed, mask_processed, metadata
            )

            print(f"  ✓ Restaurada: {image_restored.shape}")

            # Verificar que se restauró al tamaño original
            if image_restored.shape == test_image.shape:
                print(f"  ✓ Restauración exitosa")
            else:
                print(f"  ✗ Error en restauración: {image_restored.shape} vs {test_image.shape}")

        except Exception as e:
            print(f"  ✗ Error en test case {i + 1}: {e}")

    # Probar composición
    print(f"\nProbando composición de imágenes...")
    try:
        fg_rgba = np.random.randint(0, 256, (384, 384, 4), dtype=np.uint8)
        bg_rgb = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)

        composite = processor.composite_foreground_background(fg_rgba, bg_rgb)

        print(f"  ✓ Composición exitosa: {composite.shape}")
        if composite.shape == (384, 384, 3):
            print(f"  ✓ Dimensiones de composición correctas")
        else:
            print(f"  ✗ Dimensiones de composición incorrectas")
    except Exception as e:
        print(f"  ✗ Error en composición: {e}")

    print("\n✓ Pruebas de procesamiento de imágenes completadas")
    return True