#!/usr/bin/env python3
"""
Crea un dataset grande de harmonización usando TODAS las imágenes COCO disponibles
y técnicas de augmentación masiva.
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor
import albumentations as A

def setup_logger():
    """Configura el logger."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/create_large_dataset_{timestamp}.log'
    
    logger = logging.getLogger('large_dataset')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def get_augmentation_pipeline():
    """Define pipeline de augmentación agresiva."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.8),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.6),
        A.RandomShadow(p=0.3),
        A.RandomFog(p=0.2),
        A.RandomSunFlare(p=0.1),
    ])

def create_varied_foreground_masks():
    """Crea diferentes tipos de máscaras para foregrounds."""
    masks = []
    
    # 1. Personas (diferentes poses)
    person_templates = [
        # Persona de pie
        [(192, 100, 30), (192, 160, 80, 120), (150, 180, 25, 100), (234, 180, 25, 100), (175, 280, 34, 80), (209, 280, 34, 80)],
        # Persona sentada
        [(192, 120, 25), (192, 180, 70, 100), (140, 200, 20, 60), (244, 200, 20, 60), (170, 260, 30, 40), (214, 260, 30, 40)],
        # Persona con brazos extendidos
        [(192, 100, 30), (192, 160, 80, 120), (120, 170, 20, 80), (264, 170, 20, 80), (175, 280, 34, 80), (209, 280, 34, 80)],
    ]
    
    for template in person_templates:
        mask = np.zeros((384, 384), dtype=np.uint8)
        # Cabeza (círculo)
        cv2.circle(mask, (template[0][0], template[0][1]), template[0][2], 255, -1)
        # Torso y extremidades (rectángulos)
        for part in template[1:]:
            cv2.rectangle(mask, (part[0], part[1]), (part[0] + part[2], part[1] + part[3]), 255, -1)
        masks.append(('person', mask))
    
    # 2. Objetos (diferentes formas)
    object_shapes = [
        ('circle', lambda: create_circle_mask()),
        ('rectangle', lambda: create_rectangle_mask()),
        ('ellipse', lambda: create_ellipse_mask()),
        ('polygon', lambda: create_polygon_mask()),
        ('complex', lambda: create_complex_mask()),
    ]
    
    for shape_name, shape_func in object_shapes:
        for _ in range(3):  # 3 variaciones por forma
            masks.append((shape_name, shape_func()))
    
    return masks

def create_circle_mask():
    """Crea máscara circular con variaciones."""
    mask = np.zeros((384, 384), dtype=np.uint8)
    center = (192 + random.randint(-50, 50), 192 + random.randint(-50, 50))
    radius = random.randint(60, 150)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def create_rectangle_mask():
    """Crea máscara rectangular con variaciones."""
    mask = np.zeros((384, 384), dtype=np.uint8)
    w = random.randint(80, 200)
    h = random.randint(80, 200)
    x = random.randint(50, 384 - w - 50)
    y = random.randint(50, 384 - h - 50)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

def create_ellipse_mask():
    """Crea máscara elíptica con variaciones."""
    mask = np.zeros((384, 384), dtype=np.uint8)
    center = (192 + random.randint(-30, 30), 192 + random.randint(-30, 30))
    axes = (random.randint(50, 120), random.randint(70, 150))
    angle = random.randint(0, 360)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask

def create_polygon_mask():
    """Crea máscara poligonal con variaciones."""
    mask = np.zeros((384, 384), dtype=np.uint8)
    num_points = random.randint(5, 8)
    center = (192, 192)
    radius = random.randint(80, 140)
    
    points = []
    for i in range(num_points):
        angle = (2 * np.pi * i) / num_points + random.uniform(-0.3, 0.3)
        r = radius + random.randint(-20, 20)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def create_complex_mask():
    """Crea máscara compleja combinando formas."""
    mask = np.zeros((384, 384), dtype=np.uint8)
    
    # Forma principal
    cv2.circle(mask, (192, 150), 80, 255, -1)
    
    # Formas adicionales
    cv2.rectangle(mask, (150, 200), (234, 280), 255, -1)
    cv2.ellipse(mask, (192, 320), (40, 30), 0, 0, 360, 255, -1)
    
    return mask

def create_synthetic_foreground(mask_info, base_colors=None):
    """Crea un foreground sintético usando una máscara."""
    shape_type, mask = mask_info
    
    # Generar imagen base con textura
    if base_colors is None:
        base_colors = [
            (120, 80, 60),   # Tonos piel
            (80, 120, 160),  # Tonos ropa azul
            (160, 80, 80),   # Tonos ropa roja
            (80, 160, 80),   # Tonos ropa verde
            (100, 100, 100), # Tonos grises
            (150, 100, 50),  # Tonos marrones
        ]
    
    base_color = random.choice(base_colors)
    
    # Crear imagen con variación de color
    img = np.full((384, 384, 3), base_color, dtype=np.uint8)
    
    # Agregar textura/ruido
    noise = np.random.randint(-40, 40, (384, 384, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Agregar gradientes sutiles
    y_grad = np.linspace(-20, 20, 384).reshape(-1, 1)
    x_grad = np.linspace(-10, 10, 384).reshape(1, -1)
    gradient = (y_grad + x_grad).astype(np.int16)
    
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + gradient, 0, 255)
    
    # Aplicar máscara
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    
    return rgba

def process_single_coco_background(args):
    """Procesa una imagen COCO para crear múltiples backgrounds."""
    coco_path, output_dir, base_idx, augment_pipeline = args
    
    try:
        img = cv2.imread(coco_path, cv2.IMREAD_COLOR)
        if img is None:
            return 0
        
        created = 0
        base_name = os.path.splitext(os.path.basename(coco_path))[0]
        
        # Original redimensionado
        img_resized = cv2.resize(img, (384, 384))
        output_path = os.path.join(output_dir, f'bg_coco_{base_idx:06d}_{created:02d}.jpg')
        cv2.imwrite(output_path, img_resized)
        created += 1
        
        # Versiones augmentadas
        for aug_idx in range(4):  # 4 augmentaciones por imagen
            try:
                img_uint8 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                augmented = augment_pipeline(image=img_uint8)['image']
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                output_path = os.path.join(output_dir, f'bg_coco_{base_idx:06d}_{created:02d}.jpg')
                cv2.imwrite(output_path, augmented_bgr)
                created += 1
            except:
                continue
        
        return created
    except:
        return 0

def create_massive_backgrounds(coco_dir, output_dir, max_backgrounds=10000):
    """Crea backgrounds masivos desde COCO."""
    logger = logging.getLogger('large_dataset')
    
    logger.info(f"🖼️ CREANDO BACKGROUNDS MASIVOS DESDE COCO")
    logger.info(f"   COCO dir: {coco_dir}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Target: {max_backgrounds}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Encontrar todas las imágenes COCO
    all_images = []
    for subdir in ['train2017', 'val2017']:
        subdir_path = os.path.join(coco_dir, subdir)
        if os.path.exists(subdir_path):
            images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(images)
    
    logger.info(f"   Imágenes COCO encontradas: {len(all_images)}")
    
    # Limitar número de imágenes base
    max_base_images = max_backgrounds // 5  # 5 versiones por imagen
    if len(all_images) > max_base_images:
        all_images = random.sample(all_images, max_base_images)
    
    # Preparar argumentos para procesamiento paralelo
    augment_pipeline = get_augmentation_pipeline()
    args_list = [(img_path, output_dir, idx, augment_pipeline) 
                 for idx, img_path in enumerate(all_images)]
    
    # Procesamiento paralelo
    total_created = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_coco_background, args_list))
        total_created = sum(results)
    
    logger.info(f"   ✅ Creados {total_created} backgrounds desde COCO")
    return total_created

def create_massive_foregrounds(output_dir, num_foregrounds=2000):
    """Crea foregrounds masivos con variaciones."""
    logger = logging.getLogger('large_dataset')
    
    logger.info(f"🎨 CREANDO FOREGROUNDS MASIVOS")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Target: {num_foregrounds}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear plantillas de máscaras
    mask_templates = create_varied_foreground_masks()
    logger.info(f"   Plantillas de máscara: {len(mask_templates)}")
    
    # Generar augmentación para foregrounds
    fg_augment = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=30, p=0.9),
        A.RandomGamma(gamma_limit=(60, 140), p=0.6),
    ])
    
    created = 0
    
    # Crear múltiples variaciones de cada plantilla
    variations_per_template = num_foregrounds // len(mask_templates)
    
    for template_idx, mask_info in enumerate(mask_templates):
        shape_type, base_mask = mask_info
        
        for var_idx in range(variations_per_template):
            try:
                # Crear foreground base
                fg_rgba = create_synthetic_foreground((shape_type, base_mask))
                
                # Aplicar augmentación
                fg_rgb = fg_rgba[:, :, :3]
                mask = fg_rgba[:, :, 3]
                
                augmented = fg_augment(image=fg_rgb)['image']
                
                # Reconstituir RGBA
                final_rgba = np.zeros((384, 384, 4), dtype=np.uint8)
                final_rgba[:, :, :3] = augmented
                final_rgba[:, :, 3] = mask
                
                # Guardar
                output_path = os.path.join(output_dir, f'fg_{shape_type}_{template_idx:03d}_{var_idx:04d}.png')
                cv2.imwrite(output_path, final_rgba)
                created += 1
                
                if created % 500 == 0:
                    logger.info(f"   Progreso: {created}/{num_foregrounds}")
                
            except Exception as e:
                continue
    
    logger.info(f"   ✅ Creados {created} foregrounds")
    return created

def main():
    """Función principal para crear dataset masivo."""
    logger = setup_logger()
    
    try:
        # Configuración
        coco_dir = 'COCO'
        fg_dir = 'dataset/foregrounds'
        bg_dir = 'dataset/backgrounds'
        
        # Verificar COCO
        if not os.path.exists(coco_dir):
            logger.error(f"❌ Directorio COCO no encontrado: {coco_dir}")
            return
        
        # Backup si existe dataset actual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(fg_dir) or os.path.exists(bg_dir):
            backup_dir = f'dataset/backup_large_{timestamp}'
            os.makedirs(backup_dir, exist_ok=True)
            
            if os.path.exists(fg_dir):
                import shutil
                shutil.move(fg_dir, f'{backup_dir}/foregrounds_old')
            if os.path.exists(bg_dir):
                import shutil
                shutil.move(bg_dir, f'{backup_dir}/backgrounds_old')
            
            logger.info(f"📦 Backup creado en: {backup_dir}")
        
        # Crear backgrounds masivos
        bg_created = create_massive_backgrounds(coco_dir, bg_dir, max_backgrounds=10000)
        
        # Crear foregrounds masivos
        fg_created = create_massive_foregrounds(fg_dir, num_foregrounds=2000)
        
        # Cálculos finales
        dataset_size = fg_created * 2  # Según harmonization.py
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        logger.info(f"\n" + "="*60)
        logger.info(f"📋 REPORTE FINAL - DATASET MASIVO")
        logger.info(f"="*60)
        logger.info(f"✅ Foregrounds creados: {fg_created:,}")
        logger.info(f"✅ Backgrounds creados: {bg_created:,}")
        logger.info(f"📊 Tamaño total del dataset: {dataset_size:,}")
        logger.info(f"📊 Train samples: {train_size:,}")
        logger.info(f"📊 Val samples: {val_size:,}")
        logger.info(f"📊 Multiplicador vs original: {dataset_size // 20}x")
        logger.info(f"🎯 Dataset masivo creado exitosamente!")
        
        if dataset_size >= 5000:
            logger.info(f"🎉 ¡Dataset de calidad profesional logrado!")
        elif dataset_size >= 1000:
            logger.info(f"👍 Dataset de tamaño adecuado para entrenamiento")
        else:
            logger.warning(f"⚠️ Dataset aún pequeño, considera más datos")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()