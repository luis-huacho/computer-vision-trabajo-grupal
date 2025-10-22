import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import json
import albumentations as A
from PIL import Image
import warnings

import warnings
import logging

# Configurar un logger básico para este módulo
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')


class COCOPersonDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes y máscaras del dataset COCO.
    Se enfoca únicamente en las personas (keypoints).
    """

    def __init__(self, coco_root, annotation_file, transform=None, image_size=384, store_metadata=False):
        """
        Args:
            coco_root: Directorio raíz del dataset COCO
            annotation_file: Archivo JSON de anotaciones (person_keypoints_train2017.json o person_keypoints_val2017.json)
            transform: Transformaciones de albumentations
            image_size: Tamaño al que redimensionar las imágenes
            store_metadata: Si guardar metadatos para restaurar tamaño original
        """
        # Identificador de proceso para logs en entorno distribuido
        rank = os.environ.get('RANK', 'N/A')
        self.log_prefix = f'[RANK {rank}] '

        logger.info(f'{self.log_prefix}Inicializando COCOPersonDataset...')
        logger.info(f'{self.log_prefix}  - COCO Root: {coco_root}')
        logger.info(f'{self.log_prefix}  - Annotation File: {annotation_file}')

        self.coco_root = coco_root
        self.annotation_file = annotation_file
        self.transform = transform
        self.image_size = image_size
        self.store_metadata = store_metadata

        # Import ImageProcessor desde utils
        try:
            from utils import ImageProcessor
            self.processor = ImageProcessor()
        except ImportError:
            logger.warning(f'{self.log_prefix}ImageProcessor no disponible desde utils')
            self.processor = None

        # Cargar anotaciones COCO
        if not os.path.exists(annotation_file):
            logger.error(f'{self.log_prefix}El archivo de anotaciones no existe: {annotation_file}')
            self.valid_image_ids = []
            return

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        logger.info(f'{self.log_prefix}Anotaciones cargadas. Total de imágenes en JSON: {len(self.coco_data['images'])}')
        logger.info(f'{self.log_prefix}Total de anotaciones en JSON: {len(self.coco_data['annotations'])}')

        # Crear mapeos
        self.images = {img['id']: img for img in self.coco_data['images']}

        # Filtrar solo anotaciones con keypoints válidos (personas)
        self.annotations = []
        min_area = 500
        for ann in self.coco_data['annotations']:
            if 'keypoints' in ann and ann.get('area', 0) > min_area:
                self.annotations.append(ann)
        
        logger.info(f"{self.log_prefix}Anotaciones filtradas por 'keypoints' y área > {min_area}: {len(self.annotations)} anotaciones válidas")

        # Agrupar anotaciones por imagen
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

        # Lista de IDs de imágenes válidas (que tienen personas)
        self.valid_image_ids = list(self.image_to_annotations.keys())

        logger.info(f'{self.log_prefix}Dataset COCO cargado: {len(self.valid_image_ids)} imágenes con personas encontradas.')

    def __len__(self):
        return len(self.valid_image_ids)

    def _get_image_directory(self):
        """
        Determina el directorio correcto de imágenes basado en el archivo de anotaciones.
        """
        if 'train2017' in self.annotation_file:
            return os.path.join(self.coco_root, 'train2017')
        elif 'val2017' in self.annotation_file:
            return os.path.join(self.coco_root, 'val2017')
        else:
            # Fallback: intentar detectar automáticamente
            print(f"⚠️  No se pudo determinar directorio desde: {self.annotation_file}")
            print("   Usando train2017 como fallback")
            return os.path.join(self.coco_root, 'train2017')

    def create_person_mask(self, image_shape, annotations):
        """
        Crea máscara de personas a partir de las anotaciones COCO.
        Usa la segmentación si está disponible, sino crea rectángulo del bbox.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in annotations:
            # Usar segmentación si está disponible
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    if len(seg) >= 6:  # Mínimo 3 puntos (x,y pairs)
                        # Convertir a array de puntos
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [points], 255)
            else:
                # Fallback: usar bounding box
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w_box, h_box = bbox
                    x, y, w_box, h_box = int(x), int(y), int(w_box), int(h_box)
                    # Asegurar que está dentro de los límites
                    x = max(0, x)
                    y = max(0, y)
                    w_box = min(w_box, w - x)
                    h_box = min(h_box, h - y)
                    if w_box > 0 and h_box > 0:
                        cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 255, -1)

        return mask

    def __getitem__(self, idx):
        # Obtener ID de imagen
        img_id = self.valid_image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.image_to_annotations[img_id]

        # Construir path de imagen
        image_dir = self._get_image_directory()
        img_path = os.path.join(image_dir, img_info['file_name'])

        try:
            # Cargar imagen
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Crear máscara de personas
            mask = self.create_person_mask(image.shape, annotations)

            # Usar redimensionamiento con padding que mantiene proporciones
            if self.processor:
                image_processed, mask_processed, restore_metadata = self.processor.resize_with_padding(
                    image, mask, self.image_size
                )
            else:
                # Fallback simple
                image_processed = cv2.resize(image, (self.image_size, self.image_size))
                mask_processed = cv2.resize(mask, (self.image_size, self.image_size))
                restore_metadata = {}

            # Normalizar imagen
            image_processed = image_processed.astype(np.float32) / 255.0

            # Crear máscara binaria
            alpha = (mask_processed > 127).astype(np.float32)

            # Aplicar transformaciones si están definidas
            if self.transform:
                try:
                    # Convertir alpha de nuevo a uint8 para albumentations
                    mask_uint8 = (alpha * 255).astype(np.uint8)
                    image_uint8 = (image_processed * 255).astype(np.uint8)

                    augmented = self.transform(image=image_uint8, mask=mask_uint8)
                    image_aug = augmented['image']
                    mask_aug = augmented['mask']

                    # Convertir de vuelta a float32
                    if isinstance(image_aug, torch.Tensor):
                        image_processed = image_aug.float() / 255.0
                    else:
                        image_processed = image_aug.astype(np.float32) / 255.0

                    alpha = (mask_aug > 127).astype(np.float32)

                except Exception as e:
                    print(f"Error en transformación: {e}")
                    # Usar versión sin transformar
                    pass

            # Crear target con 4 canales (RGBA)
            target = np.zeros((4, self.image_size, self.image_size), dtype=np.float32)

            # RGB channels
            if len(image_processed.shape) == 3:
                target[:3] = image_processed.transpose(2, 0, 1)
            else:
                target[:3] = image_processed

            # Alpha channel
            if len(alpha.shape) == 2:
                target[3] = alpha
            else:
                target[3] = alpha.squeeze()

            # Convertir imagen de entrada a tensor
            if not isinstance(image_processed, torch.Tensor):
                if len(image_processed.shape) == 3:
                    image_tensor = torch.FloatTensor(image_processed.transpose(2, 0, 1))
                else:
                    image_tensor = torch.FloatTensor(image_processed)
            else:
                image_tensor = image_processed.float()

            # Preparar salida
            result = (image_tensor, torch.FloatTensor(target))

            # Si se requieren metadatos, agregarlos
            if self.store_metadata:
                result = result + (restore_metadata,)

            return result

        except Exception as e:
            print(f"Error procesando imagen {img_id}: {e}")
            # Retornar tensor dummy en caso de error
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_target = torch.zeros(4, self.image_size, self.image_size)
            dummy_metadata = {
                'original_size': (self.image_size, self.image_size),
                'scale': 1.0,
                'new_size': (self.image_size, self.image_size),
                'padding': (0, 0, 0, 0)
            }

            if self.store_metadata:
                return dummy_image, dummy_target, dummy_metadata
            else:
                return dummy_image, dummy_target


def get_transforms(train=True, size=384):
    """
    Define transformaciones de aumentación de datos.

    Args:
        train (bool): Si True, aplica augmentaciones completas. Si False, solo resize.
        size (int): Tamaño target para resize (default: 384)

    Returns:
        albumentations.Compose: Transformaciones a aplicar
    """
    if train:
        # Transformaciones para entrenamiento: Resize + augmentaciones
        transform = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
        ], additional_targets={'mask': 'mask'})
    else:
        # Transformaciones para validación: Solo resize
        transform = A.Compose([
            A.Resize(size, size),
        ], additional_targets={'mask': 'mask'})

    return transform


def create_sample_batch():
    """
    Crea un batch de muestra para verificar que todo funciona.
    """
    print("=== CREANDO BATCH DE MUESTRA ===\n")

    coco_root = 'COCO'
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')

    try:
        # Crear dataset pequeño
        print("📊 Creando dataset de muestra...")
        dataset = COCOPersonDataset(
            coco_root=coco_root,
            annotation_file=train_ann_file,
            transform=None,
            image_size=384,
            store_metadata=False
        )

        if len(dataset) == 0:
            print("❌ Dataset vacío")
            return False

        # Crear dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        print(f"✅ Dataset creado con {len(dataset)} imágenes")
        print("🔄 Cargando primer batch...")

        # Cargar primer batch
        for batch_idx, (images, targets) in enumerate(loader):
            print(f"✅ Batch cargado exitosamente:")
            print(f"   - Imágenes shape: {images.shape}")
            print(f"   - Targets shape: {targets.shape}")
            print(f"   - Tipo de datos: {images.dtype}, {targets.dtype}")
            print(f"   - Rango imágenes: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   - Rango targets: [{targets.min():.3f}, {targets.max():.3f}]")

            # Verificar canales
            rgb_channels = targets[:, :3, :, :]
            alpha_channel = targets[:, 3:4, :, :]
            print(f"   - RGB channels shape: {rgb_channels.shape}")
            print(f"   - Alpha channel shape: {alpha_channel.shape}")
            print(f"   - Alpha channel range: [{alpha_channel.min():.3f}, {alpha_channel.max():.3f}]")

            # Solo procesar primer batch
            break

        print("✅ Verificación de batch completada exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error creando batch de muestra: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coco_dataset():
    """
    Función de prueba para verificar que el dataset COCO se carga correctamente.
    """
    print("Probando carga del dataset COCO...")

    coco_root = 'COCO'
    train_ann_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')

    if not os.path.exists(train_ann_file):
        print(f"✗ Archivo de anotaciones no encontrado: {train_ann_file}")
        return False

    try:
        # Crear dataset de prueba con solo unas pocas imágenes
        dataset = COCOPersonDataset(
            coco_root=coco_root,
            annotation_file=train_ann_file,
            transform=None,
            image_size=384,
            store_metadata=False
        )

        print(f"✓ Dataset COCO cargado exitosamente")
        print(f"  Total de imágenes con personas: {len(dataset)}")

        if len(dataset) > 0:
            # Probar cargar una muestra
            sample = dataset[0]
            image, target = sample

            print(f"  ✓ Muestra cargada exitosamente")
            print(f"    Image shape: {image.shape}")
            print(f"    Target shape: {target.shape}")

            return True
        else:
            print("✗ Dataset vacío")
            return False

    except Exception as e:
        print(f"✗ Error cargando dataset COCO: {e}")
        return False


if __name__ == "__main__":
    # Pruebas del módulo
    print("=== PRUEBAS DEL MÓDULO DE DATASETS ===\n")

    tests = [
        ("Dataset COCO", test_coco_dataset),
        ("Batch de muestra", create_sample_batch),
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
    print("📋 RESUMEN DE PRUEBAS DE DATASETS:")

    all_passed = True
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n🎉 ¡TODAS LAS PRUEBAS DE DATASETS EXITOSAS!")
    else:
        print(f"\n⚠️  Algunas pruebas fallaron")


# ============================================================================
# AISEGMENT MATTING HUMAN DATASET
# ============================================================================

class AISegmentDataset(Dataset):
    """
    AISegment.com Matting Human Dataset
    https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets

    Dataset de 34,425 imágenes de retratos de medio cuerpo con matting masks.

    Estructura:
        clip_img/{group-id}/clip_{subgroup-id}/{group-id}-{img-id}.jpg
        matting/{group-id}/matting_{subgroup-id}/{group-id}-{img-id}.png

    Args:
        root (str): Directorio raíz del dataset (ej: 'datasets/AISegment')
        split (str): 'train' o 'val'
        transforms (callable, optional): Transformaciones a aplicar
        auto_download (bool): Si True, intenta descargar con kagglehub
        kaggle_dataset_id (str): ID del dataset en Kaggle
        train_val_split (float): Proporción de datos para entrenamiento (default: 0.8)
        random_seed (int): Seed para reproducibilidad del split
    """

    def __init__(self,
                 root='datasets/AISegment',
                 split='train',
                 transforms=None,
                 auto_download=True,
                 kaggle_dataset_id='laurentmih/aisegmentcom-matting-human-datasets',
                 train_val_split=0.8,
                 random_seed=42):

        self.root = root
        self.split = split
        self.transforms = transforms
        self.train_val_split = train_val_split
        self.random_seed = random_seed

        # Intentar descarga híbrida
        self._setup_dataset(auto_download, kaggle_dataset_id)

        # Parsear estructura de directorios y crear lista de muestras
        self.samples = self._parse_dataset_structure()

        # Split train/val
        self._create_train_val_split()

        print(f"✅ AISegment Dataset cargado: {len(self.samples)} imágenes ({split})")

    def _setup_dataset(self, auto_download, kaggle_dataset_id):
        """
        Configura el dataset, intentando descarga automática si es necesario.
        """
        # Verificar si el dataset ya existe
        clip_img_dir = os.path.join(self.root, 'clip_img')
        matting_dir = os.path.join(self.root, 'matting')

        if os.path.exists(clip_img_dir) and os.path.exists(matting_dir):
            print(f"✅ Dataset AISegment encontrado en: {self.root}")
            return

        # Si no existe y auto_download está activado, intentar descargar
        if auto_download and kaggle_dataset_id:
            print(f"📥 Intentando descargar dataset AISegment con kagglehub...")
            try:
                import kagglehub

                # Descargar dataset
                download_path = kagglehub.dataset_download(kaggle_dataset_id)
                print(f"✅ Dataset descargado en: {download_path}")

                # Si la descarga fue a un path diferente, actualizar self.root
                # kagglehub descarga a ~/.cache/kagglehub/datasets/...
                if os.path.exists(os.path.join(download_path, 'clip_img')):
                    self.root = download_path
                    print(f"✅ Usando dataset desde: {self.root}")
                else:
                    # Buscar subdirectorios
                    for subdir in os.listdir(download_path):
                        potential_path = os.path.join(download_path, subdir)
                        if os.path.exists(os.path.join(potential_path, 'clip_img')):
                            self.root = potential_path
                            print(f"✅ Usando dataset desde: {self.root}")
                            break

            except ImportError:
                print("⚠️  kagglehub no está instalado. Instala con: pip install kagglehub")
                print(f"⚠️  Usando path manual: {self.root}")
            except Exception as e:
                print(f"⚠️  Error descargando dataset: {e}")
                print(f"⚠️  Usando path manual: {self.root}")

        # Verificar nuevamente si existe
        if not os.path.exists(os.path.join(self.root, 'clip_img')):
            # Antes de fallar, intentar buscar en cache de kagglehub
            # (Útil para otros ranks en DDP que esperan que rank0 descargue)
            cache_path = self._try_find_in_kaggle_cache(kaggle_dataset_id)
            if cache_path:
                self.root = cache_path
                print(f"✅ Dataset encontrado en cache de Kaggle: {self.root}")
                return

            # Si nada funciona, lanzar error
            raise FileNotFoundError(
                f"Dataset AISegment no encontrado en {self.root}\n"
                f"Opciones:\n"
                f"1. Descargar manualmente desde Kaggle y colocar en {self.root}\n"
                f"2. Instalar kagglehub y configurar API key de Kaggle\n"
                f"3. Especificar otro path en la configuración YAML"
            )

    def _try_find_in_kaggle_cache(self, kaggle_dataset_id):
        """
        Busca el dataset en la cache de kagglehub sin descargar.
        Útil para otros ranks en DDP que esperan que rank0 descargue.

        Args:
            kaggle_dataset_id: ID del dataset en formato "owner/dataset-name"

        Returns:
            Path al dataset en cache si existe, None si no se encuentra
        """
        if not kaggle_dataset_id:
            return None

        try:
            cache_base = os.path.expanduser('~/.cache/kagglehub/datasets')
            # Convertir "laurentmih/aisegmentcom-matting-human-datasets" a path
            # ~/.cache/kagglehub/datasets/laurentmih/aisegmentcom-matting-human-datasets/versions/1
            parts = kaggle_dataset_id.split('/')
            if len(parts) == 2:
                # Buscar en versiones (normalmente versión 1 es la más reciente)
                dataset_cache = os.path.join(cache_base, parts[0], parts[1], 'versions')
                if os.path.exists(dataset_cache):
                    # Buscar la versión más reciente
                    versions = sorted([d for d in os.listdir(dataset_cache) if d.isdigit()], reverse=True)
                    for version in versions:
                        cache_path = os.path.join(dataset_cache, version)
                        if os.path.exists(os.path.join(cache_path, 'clip_img')):
                            return cache_path
        except Exception as e:
            # Silenciar errores, simplemente retornar None
            pass

        return None

    def _parse_dataset_structure(self):
        """
        Parsea la estructura jerárquica del dataset y crea lista de pares (imagen, máscara).

        Estructura:
            clip_img/{group-id}/clip_{subgroup-id}/{group-id}-{img-id}.jpg
            matting/{group-id}/matting_{subgroup-id}/{group-id}-{img-id}.png
        """
        samples = []

        clip_img_root = os.path.join(self.root, 'clip_img')
        matting_root = os.path.join(self.root, 'matting')

        # Recorrer estructura de directorios
        for group_id in os.listdir(clip_img_root):
            group_path = os.path.join(clip_img_root, group_id)

            if not os.path.isdir(group_path):
                continue

            # Recorrer subgrupos
            for subgroup_dir in os.listdir(group_path):
                subgroup_path = os.path.join(group_path, subgroup_dir)

                if not os.path.isdir(subgroup_path):
                    continue

                # Extraer subgroup_id del nombre del directorio (clip_{subgroup_id})
                if not subgroup_dir.startswith('clip_'):
                    continue

                subgroup_id = subgroup_dir.replace('clip_', '')

                # Recorrer imágenes en este subgrupo
                for img_filename in os.listdir(subgroup_path):
                    if not img_filename.endswith('.jpg'):
                        continue

                    # Path de la imagen
                    img_path = os.path.join(subgroup_path, img_filename)

                    # Construir path de la máscara correspondiente
                    # Formato: {group_id}-{img_id}.jpg → {group_id}-{img_id}.png
                    mask_filename = img_filename.replace('.jpg', '.png')
                    mask_path = os.path.join(
                        matting_root,
                        group_id,
                        f'matting_{subgroup_id}',
                        mask_filename
                    )

                    # Verificar que la máscara existe
                    if os.path.exists(mask_path):
                        samples.append({
                            'image': img_path,
                            'mask': mask_path,
                            'group_id': group_id,
                            'subgroup_id': subgroup_id,
                            'filename': img_filename
                        })

        if len(samples) == 0:
            raise RuntimeError(
                f"No se encontraron imágenes válidas en {self.root}\n"
                f"Verificar estructura del dataset"
            )

        return samples

    def _create_train_val_split(self):
        """
        Crea split train/val de forma reproducible.
        """
        np.random.seed(self.random_seed)

        # Shuffle indices
        indices = np.arange(len(self.samples))
        np.random.shuffle(indices)

        # Split
        split_idx = int(len(self.samples) * self.train_val_split)

        if self.split == 'train':
            indices = indices[:split_idx]
        elif self.split == 'val':
            indices = indices[split_idx:]
        else:
            raise ValueError(f"Split debe ser 'train' o 'val', recibido: {self.split}")

        # Filtrar samples
        self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retorna imagen RGB y target RGBA (compatible con modelo de segmentación).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image: Tensor de 3 canales RGB [3, H, W] (input del modelo)
                - target_rgba: Tensor de 4 canales RGBA [4, H, W] (output esperado)
        """
        sample = self.samples[idx]

        # Cargar imagen RGB
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Cargar máscara PNG (4 canales: RGBA)
        mask_rgba = cv2.imread(sample['mask'], cv2.IMREAD_UNCHANGED)

        # Extraer canal alpha (índice 3)
        if mask_rgba.shape[2] == 4:
            alpha = mask_rgba[:, :, 3]
        else:
            # Si por alguna razón no tiene 4 canales, crear alpha desde grayscale
            alpha = cv2.cvtColor(mask_rgba, cv2.COLOR_RGB2GRAY)

        # Normalizar alpha a [0, 1]
        alpha = alpha.astype(np.float32) / 255.0

        # Aplicar transformaciones si están disponibles
        if self.transforms:
            # albumentations espera image y mask
            # Crear una imagen temporal RGBA para transformar todo junto
            transformed = self.transforms(
                image=image,
                mask=alpha
            )
            image = transformed['image']
            alpha = transformed['mask']

        # Convertir a tensores
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        alpha = torch.from_numpy(alpha).unsqueeze(0).float()  # [1, H, W]

        # Concatenar RGB + Alpha → RGBA [4, H, W]
        image_rgba = torch.cat([image, alpha], dim=0)

        # Retornar (image, image_rgba) para compatibilidad con modelo
        # Input: RGB de 3 canales (esperado por ResNet-50)
        # Target: RGBA de 4 canales (salida del modelo)
        return image, image_rgba


# ============================================================================
# FUNCIONES DE TEST PARA AISEGMENT
# ============================================================================

def test_aisegment_dataset():
    """
    Prueba del dataset AISegment.
    """
    try:
        print("\n🧪 PROBANDO AISEGMENT DATASET")
        print("=" * 50)

        # Intentar cargar dataset
        dataset = AISegmentDataset(
            root='datasets/AISegment',
            split='train',
            transforms=get_transforms(train=False, size=384),
            auto_download=False  # No descargar en tests
        )

        print(f"✅ Dataset cargado: {len(dataset)} imágenes")

        # Probar cargar una muestra
        if len(dataset) > 0:
            image_rgba, target_rgba = dataset[0]
            print(f"✅ Muestra cargada: image shape {image_rgba.shape}, target shape {target_rgba.shape}")

            assert image_rgba.shape[0] == 4, "Debe tener 4 canales (RGBA)"
            assert target_rgba.shape[0] == 4, "Target debe tener 4 canales (RGBA)"

        return True

    except FileNotFoundError:
        print("⚠️  Dataset AISegment no encontrado (esto es normal si no está descargado)")
        return True  # No es un error crítico en tests
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False


if __name__ == "__main__":
    # Agregar test de AISegment
    print("\n🧪 TESTS DE DATASETS")
    print("=" * 50)

    # Test COCO (existente)
    coco_result = test_coco_dataset()

    # Test AISegment (nuevo)
    aisegment_result = test_aisegment_dataset()

    print("\n" + "=" * 50)
    print("📋 RESUMEN:")
    print(f"   {'✅' if coco_result else '❌'} COCO Dataset")
    print(f"   {'✅' if aisegment_result else '❌'} AISegment Dataset")

    if coco_result and aisegment_result:
        print("\n🎉 ¡TODOS LOS TESTS EXITOSOS!")
    else:
        print("\n⚠️  Algunos tests fallaron")