#!/usr/bin/env python3
"""
Crea un dataset robusto de harmonización usando datos COCO y segmentación existentes.
"""

import os
import cv2
import numpy as np
import json
import logging
from datetime import datetime
import shutil
from pathlib import Path

def setup_logger():
    """Configura el logger."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/create_harmonization_dataset_{timestamp}.log'
    
    logger = logging.getLogger('dataset_creator')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    logger.info(f"Iniciando creación de dataset de harmonización")
    logger.info(f"Log guardado en: {log_filename}")
    return logger

def find_existing_data():
    """Encuentra datos existentes de COCO y segmentación."""
    logger = logging.getLogger('dataset_creator')
    
    # Buscar directorios COCO
    coco_paths = []
    for potential_path in ['COCO', 'coco', '../COCO', 'dataset/COCO']:
        if os.path.exists(potential_path):
            coco_paths.append(potential_path)
    
    # Buscar outputs de segmentación
    segmentation_paths = []
    for potential_path in ['outputs', 'segmentation_outputs', 'results']:
        if os.path.exists(potential_path):
            segmentation_paths.append(potential_path)
    
    # Buscar máscaras generadas
    mask_paths = []
    for potential_path in ['masks', 'segmentation_masks', 'outputs/masks']:
        if os.path.exists(potential_path):
            mask_paths.append(potential_path)
    
    logger.info("=== DATOS EXISTENTES ENCONTRADOS ===")
    logger.info(f"📁 COCO paths: {coco_paths}")
    logger.info(f"📁 Segmentation paths: {segmentation_paths}")
    logger.info(f"📁 Mask paths: {mask_paths}")
    
    return {
        'coco_paths': coco_paths,
        'segmentation_paths': segmentation_paths,
        'mask_paths': mask_paths
    }

def analyze_coco_data(coco_paths):
    """Analiza los datos COCO disponibles."""
    logger = logging.getLogger('dataset_creator')
    
    coco_analysis = {}
    
    for coco_path in coco_paths:
        logger.info(f"\n🔍 Analizando COCO en: {coco_path}")
        
        # Buscar subdirectorios
        subdirs = []
        if os.path.exists(coco_path):
            subdirs = [d for d in os.listdir(coco_path) 
                      if os.path.isdir(os.path.join(coco_path, d))]
        
        logger.info(f"   Subdirectorios: {subdirs}")
        
        # Analizar imágenes en cada subdir
        for subdir in subdirs:
            subdir_path = os.path.join(coco_path, subdir)
            if subdir in ['train2017', 'val2017', 'test2017']:
                images = [f for f in os.listdir(subdir_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                logger.info(f"   {subdir}: {len(images)} imágenes")
                
                if len(images) > 0:
                    coco_analysis[f"{coco_path}/{subdir}"] = {
                        'path': subdir_path,
                        'count': len(images),
                        'sample_files': images[:5]
                    }
    
    return coco_analysis

def analyze_segmentation_data(segmentation_paths, mask_paths):
    """Analiza los datos de segmentación disponibles."""
    logger = logging.getLogger('dataset_creator')
    
    segmentation_analysis = {}
    
    # Analizar outputs de segmentación
    for seg_path in segmentation_paths:
        logger.info(f"\n🔍 Analizando segmentación en: {seg_path}")
        
        if os.path.exists(seg_path):
            files = [f for f in os.listdir(seg_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"   Archivos de segmentación: {len(files)}")
            
            if len(files) > 0:
                segmentation_analysis[seg_path] = {
                    'path': seg_path,
                    'count': len(files),
                    'sample_files': files[:5]
                }
    
    # Analizar máscaras
    for mask_path in mask_paths:
        logger.info(f"\n🔍 Analizando máscaras en: {mask_path}")
        
        if os.path.exists(mask_path):
            files = [f for f in os.listdir(mask_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"   Archivos de máscara: {len(files)}")
            
            if len(files) > 0:
                segmentation_analysis[f"{mask_path}_masks"] = {
                    'path': mask_path,
                    'count': len(files),
                    'sample_files': files[:5]
                }
    
    return segmentation_analysis

def create_foregrounds_from_segmentation(segmentation_analysis, target_dir, max_images=200):
    """Crea foregrounds usando datos de segmentación."""
    logger = logging.getLogger('dataset_creator')
    
    logger.info(f"\n🎨 CREANDO FOREGROUNDS DESDE SEGMENTACIÓN")
    logger.info(f"   Target: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    created_count = 0
    
    for source_name, data in segmentation_analysis.items():
        if created_count >= max_images:
            break
            
        source_path = data['path']
        logger.info(f"   Procesando: {source_name}")
        
        # Buscar pares imagen-máscara
        files = [f for f in os.listdir(source_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in files[:50]:  # Limitar por fuente
            if created_count >= max_images:
                break
                
            try:
                img_path = os.path.join(source_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Crear foreground con alpha channel
                # Si es una imagen segmentada, usar como máscara
                if 'mask' in source_name.lower():
                    # Es una máscara, buscar imagen original
                    original_name = filename.replace('_mask', '').replace('_seg', '')
                    original_path = None
                    
                    # Buscar imagen original en otros directorios
                    for other_name, other_data in segmentation_analysis.items():
                        if 'mask' not in other_name.lower():
                            test_path = os.path.join(other_data['path'], original_name)
                            if os.path.exists(test_path):
                                original_path = test_path
                                break
                    
                    if original_path:
                        original_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
                        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if original_img is not None and mask is not None:
                            # Redimensionar
                            original_img = cv2.resize(original_img, (384, 384))
                            mask = cv2.resize(mask, (384, 384))
                            
                            # Crear RGBA
                            rgba = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGBA)
                            rgba[:, :, 3] = mask
                            
                            # Guardar
                            output_path = os.path.join(target_dir, f'fg_seg_{created_count:04d}.png')
                            cv2.imwrite(output_path, rgba)
                            created_count += 1
                
                else:
                    # Es una imagen segmentada, crear máscara simple
                    img_resized = cv2.resize(img, (384, 384))
                    
                    # Crear máscara basada en contenido (simplificado)
                    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                    
                    # Crear RGBA
                    rgba = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGBA)
                    rgba[:, :, 3] = mask
                    
                    # Guardar
                    output_path = os.path.join(target_dir, f'fg_seg_{created_count:04d}.png')
                    cv2.imwrite(output_path, rgba)
                    created_count += 1
                    
            except Exception as e:
                logger.warning(f"   Error procesando {filename}: {e}")
                continue
    
    logger.info(f"   ✅ Creados {created_count} foregrounds desde segmentación")
    return created_count

def create_backgrounds_from_coco(coco_analysis, target_dir, max_images=500):
    """Crea backgrounds usando imágenes COCO."""
    logger = logging.getLogger('dataset_creator')
    
    logger.info(f"\n🖼️ CREANDO BACKGROUNDS DESDE COCO")
    logger.info(f"   Target: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    created_count = 0
    
    for source_name, data in coco_analysis.items():
        if created_count >= max_images:
            break
            
        source_path = data['path']
        logger.info(f"   Procesando: {source_name}")
        
        files = [f for f in os.listdir(source_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Tomar una muestra aleatoria
        import random
        files = random.sample(files, min(len(files), 100))
        
        for filename in files:
            if created_count >= max_images:
                break
                
            try:
                img_path = os.path.join(source_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Redimensionar
                img_resized = cv2.resize(img, (384, 384))
                
                # Guardar
                output_path = os.path.join(target_dir, f'bg_coco_{created_count:04d}.jpg')
                cv2.imwrite(output_path, img_resized)
                created_count += 1
                
            except Exception as e:
                logger.warning(f"   Error procesando {filename}: {e}")
                continue
    
    logger.info(f"   ✅ Creados {created_count} backgrounds desde COCO")
    return created_count

def create_enhanced_dataset():
    """Función principal para crear dataset mejorado."""
    logger = setup_logger()
    
    try:
        # 1. Encontrar datos existentes
        existing_data = find_existing_data()
        
        # 2. Analizar COCO
        coco_analysis = analyze_coco_data(existing_data['coco_paths'])
        
        # 3. Analizar segmentación
        segmentation_analysis = analyze_segmentation_data(
            existing_data['segmentation_paths'], 
            existing_data['mask_paths']
        )
        
        # 4. Crear directorios de destino
        fg_dir = 'dataset/foregrounds'
        bg_dir = 'dataset/backgrounds'
        
        # Backup del dataset actual
        if os.path.exists(fg_dir):
            backup_dir = f'dataset/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(backup_dir, exist_ok=True)
            if os.path.exists(f'{fg_dir}'):
                shutil.move(fg_dir, f'{backup_dir}/foregrounds_old')
            if os.path.exists(f'{bg_dir}'):
                shutil.move(bg_dir, f'{backup_dir}/backgrounds_old')
            logger.info(f"📦 Backup creado en: {backup_dir}")
        
        # 5. Crear nuevos foregrounds
        fg_created = 0
        if segmentation_analysis:
            fg_created = create_foregrounds_from_segmentation(
                segmentation_analysis, fg_dir, max_images=200
            )
        
        # 6. Crear nuevos backgrounds
        bg_created = 0
        if coco_analysis:
            bg_created = create_backgrounds_from_coco(
                coco_analysis, bg_dir, max_images=500
            )
        
        # 7. Reporte final
        logger.info(f"\n" + "="*60)
        logger.info(f"📋 REPORTE FINAL")
        logger.info(f"="*60)
        logger.info(f"✅ Foregrounds creados: {fg_created}")
        logger.info(f"✅ Backgrounds creados: {bg_created}")
        
        if fg_created > 0 and bg_created > 0:
            new_dataset_size = fg_created * 2
            train_size = int(0.8 * new_dataset_size)
            val_size = new_dataset_size - train_size
            
            logger.info(f"📊 Nuevo tamaño del dataset: {new_dataset_size}")
            logger.info(f"📊 Train/Val: {train_size}/{val_size}")
            logger.info(f"🎯 Dataset mejorado exitosamente!")
        else:
            logger.warning(f"⚠️ No se pudo crear dataset - datos insuficientes")
        
    except Exception as e:
        logger.error(f"❌ Error creando dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    create_enhanced_dataset()