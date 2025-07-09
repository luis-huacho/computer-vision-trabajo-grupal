#!/usr/bin/env python3
"""
Crea foregrounds sintéticos mejorados para harmonización.
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime

def setup_logger():
    """Configura el logger."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/create_improved_foregrounds_{timestamp}.log'
    
    logger = logging.getLogger('fg_creator')
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

def create_person_silhouette(width=384, height=384):
    """Crea una silueta más realista de persona."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Crear forma más realista de persona
    center_x = width // 2
    center_y = height // 2
    
    # Cabeza (círculo)
    head_radius = 25
    cv2.circle(img, (center_x, center_y - 120), head_radius, (120, 100, 80), -1)
    cv2.circle(mask, (center_x, center_y - 120), head_radius, 255, -1)
    
    # Torso (rectángulo redondeado)
    torso_width = 60
    torso_height = 100
    torso_top = center_y - 90
    torso_left = center_x - torso_width // 2
    
    cv2.rectangle(img, (torso_left, torso_top), 
                  (torso_left + torso_width, torso_top + torso_height), 
                  (80, 120, 160), -1)
    cv2.rectangle(mask, (torso_left, torso_top), 
                  (torso_left + torso_width, torso_top + torso_height), 
                  255, -1)
    
    # Brazos
    arm_width = 20
    arm_length = 80
    # Brazo izquierdo
    cv2.rectangle(img, (torso_left - arm_width, torso_top + 10), 
                  (torso_left, torso_top + 10 + arm_length), 
                  (80, 120, 160), -1)
    cv2.rectangle(mask, (torso_left - arm_width, torso_top + 10), 
                  (torso_left, torso_top + 10 + arm_length), 
                  255, -1)
    
    # Brazo derecho
    cv2.rectangle(img, (torso_left + torso_width, torso_top + 10), 
                  (torso_left + torso_width + arm_width, torso_top + 10 + arm_length), 
                  (80, 120, 160), -1)
    cv2.rectangle(mask, (torso_left + torso_width, torso_top + 10), 
                  (torso_left + torso_width + arm_width, torso_top + 10 + arm_length), 
                  255, -1)
    
    # Piernas
    leg_width = 25
    leg_height = 120
    leg_top = torso_top + torso_height
    
    # Pierna izquierda
    cv2.rectangle(img, (center_x - leg_width, leg_top), 
                  (center_x, leg_top + leg_height), 
                  (60, 80, 120), -1)
    cv2.rectangle(mask, (center_x - leg_width, leg_top), 
                  (center_x, leg_top + leg_height), 
                  255, -1)
    
    # Pierna derecha
    cv2.rectangle(img, (center_x, leg_top), 
                  (center_x + leg_width, leg_top + leg_height), 
                  (60, 80, 120), -1)
    cv2.rectangle(mask, (center_x, leg_top), 
                  (center_x + leg_width, leg_top + leg_height), 
                  255, -1)
    
    return img, mask

def create_geometric_shape(width=384, height=384, shape_type='circle'):
    """Crea formas geométricas variadas."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    center_x = width // 2
    center_y = height // 2
    
    if shape_type == 'circle':
        radius = np.random.randint(80, 150)
        cv2.circle(img, (center_x, center_y), radius, 
                   (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)), -1)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    elif shape_type == 'rectangle':
        w = np.random.randint(100, 200)
        h = np.random.randint(100, 200)
        top_left = (center_x - w//2, center_y - h//2)
        bottom_right = (center_x + w//2, center_y + h//2)
        cv2.rectangle(img, top_left, bottom_right, 
                     (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)), -1)
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
    elif shape_type == 'ellipse':
        axes = (np.random.randint(60, 120), np.random.randint(80, 150))
        angle = np.random.randint(0, 360)
        cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, 
                   (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)), -1)
        cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)
    
    return img, mask

def create_improved_foregrounds(target_dir='dataset/foregrounds', num_images=100):
    """Crea foregrounds sintéticos mejorados."""
    logger = setup_logger()
    
    logger.info(f"🎨 CREANDO FOREGROUNDS SINTÉTICOS MEJORADOS")
    logger.info(f"   Target: {target_dir}")
    logger.info(f"   Cantidad: {num_images}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    created_count = 0
    
    # 30% siluetas de personas
    person_count = int(num_images * 0.3)
    logger.info(f"   Creando {person_count} siluetas de persona...")
    
    for i in range(person_count):
        try:
            img, mask = create_person_silhouette()
            
            # Agregar variación de color
            color_variation = np.random.randint(-30, 30, (384, 384, 3))
            img = np.clip(img.astype(np.int16) + color_variation, 0, 255).astype(np.uint8)
            
            # Crear RGBA
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = mask
            
            # Guardar
            output_path = os.path.join(target_dir, f'fg_person_{created_count:04d}.png')
            cv2.imwrite(output_path, rgba)
            created_count += 1
            
        except Exception as e:
            logger.warning(f"   Error creando persona {i}: {e}")
    
    # 70% formas geométricas variadas
    shapes = ['circle', 'rectangle', 'ellipse']
    remaining = num_images - created_count
    
    logger.info(f"   Creando {remaining} formas geométricas...")
    
    for i in range(remaining):
        try:
            shape_type = shapes[i % len(shapes)]
            img, mask = create_geometric_shape(shape_type=shape_type)
            
            # Crear RGBA
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = mask
            
            # Guardar
            output_path = os.path.join(target_dir, f'fg_{shape_type}_{created_count:04d}.png')
            cv2.imwrite(output_path, rgba)
            created_count += 1
            
        except Exception as e:
            logger.warning(f"   Error creando forma {i}: {e}")
    
    logger.info(f"   ✅ Creados {created_count} foregrounds mejorados")
    
    return created_count

def main():
    """Función principal."""
    logger = setup_logger()
    
    try:
        # Verificar que existan backgrounds
        bg_dir = 'dataset/backgrounds'
        if not os.path.exists(bg_dir):
            logger.error("❌ No existe directorio de backgrounds")
            return
        
        bg_files = [f for f in os.listdir(bg_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(bg_files) == 0:
            logger.error("❌ No hay archivos de background")
            return
        
        logger.info(f"📊 Backgrounds disponibles: {len(bg_files)}")
        
        # Crear foregrounds mejorados
        fg_created = create_improved_foregrounds(num_images=100)
        
        # Calcular nuevo tamaño del dataset
        new_dataset_size = fg_created * 2
        train_size = int(0.8 * new_dataset_size)
        val_size = new_dataset_size - train_size
        
        logger.info(f"\n" + "="*60)
        logger.info(f"📋 REPORTE FINAL")
        logger.info(f"="*60)
        logger.info(f"✅ Foregrounds creados: {fg_created}")
        logger.info(f"✅ Backgrounds disponibles: {len(bg_files)}")
        logger.info(f"📊 Nuevo tamaño del dataset: {new_dataset_size}")
        logger.info(f"📊 Train/Val: {train_size}/{val_size}")
        logger.info(f"🎯 Dataset mejorado exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()