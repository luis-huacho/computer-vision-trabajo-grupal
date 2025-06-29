#!/usr/bin/env python3
"""
Analizador detallado del dataset de harmonización.
Genera un reporte completo guardado en logs/.
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import json

def setup_logger():
    """Configura el logger para el análisis del dataset."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/dataset_harmonizer_analysis_{timestamp}.log'
    
    logger = logging.getLogger('dataset_analyzer')
    logger.setLevel(logging.INFO)
    
    # Evitar duplicar handlers
    if not logger.handlers:
        # Handler para archivo
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Iniciando análisis del dataset de harmonización")
    logger.info(f"Log guardado en: {log_filename}")
    return logger, log_filename

def analyze_dataset_structure():
    """Analiza la estructura básica del dataset."""
    logger = logging.getLogger('dataset_analyzer')
    
    # Directorios del dataset según harmonization.py
    foreground_dir = 'dataset/foregrounds'
    background_dir = 'dataset/backgrounds'
    
    logger.info("=== ANÁLISIS DE ESTRUCTURA DEL DATASET ===")
    logger.info(f"Directorio de foregrounds: {foreground_dir}")
    logger.info(f"Directorio de backgrounds: {background_dir}")
    
    analysis = {
        'foreground_dir': foreground_dir,
        'background_dir': background_dir,
        'foreground_exists': os.path.exists(foreground_dir),
        'background_exists': os.path.exists(background_dir),
        'foreground_files': [],
        'background_files': [],
        'total_foregrounds': 0,
        'total_backgrounds': 0,
        'dataset_size_calculation': 0
    }
    
    # Analizar foregrounds
    if analysis['foreground_exists']:
        fg_files = [f for f in os.listdir(foreground_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        analysis['foreground_files'] = fg_files
        analysis['total_foregrounds'] = len(fg_files)
        logger.info(f"✅ Directorio foregrounds existe")
        logger.info(f"📁 Archivos de foreground encontrados: {len(fg_files)}")
        for i, f in enumerate(fg_files[:10]):  # Mostrar primeros 10
            logger.info(f"   {i+1}. {f}")
        if len(fg_files) > 10:
            logger.info(f"   ... y {len(fg_files) - 10} más")
    else:
        logger.warning(f"❌ Directorio foregrounds NO existe")
    
    # Analizar backgrounds
    if analysis['background_exists']:
        bg_files = [f for f in os.listdir(background_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        analysis['background_files'] = bg_files
        analysis['total_backgrounds'] = len(bg_files)
        logger.info(f"✅ Directorio backgrounds existe")
        logger.info(f"📁 Archivos de background encontrados: {len(bg_files)}")
        for i, f in enumerate(bg_files[:10]):  # Mostrar primeros 10
            logger.info(f"   {i+1}. {f}")
        if len(bg_files) > 10:
            logger.info(f"   ... y {len(bg_files) - 10} más")
    else:
        logger.warning(f"❌ Directorio backgrounds NO existe")
    
    # Calcular tamaño del dataset según harmonization.py
    # len(self.foreground_files) * 2  (línea 196)
    if analysis['total_foregrounds'] > 0:
        analysis['dataset_size_calculation'] = analysis['total_foregrounds'] * 2
        logger.info(f"📊 Tamaño calculado del dataset: {analysis['dataset_size_calculation']}")
        logger.info(f"   (foregrounds × 2 = {analysis['total_foregrounds']} × 2)")
    
    return analysis

def analyze_image_properties(analysis):
    """Analiza las propiedades de las imágenes."""
    logger = logging.getLogger('dataset_analyzer')
    
    logger.info("\n=== ANÁLISIS DE PROPIEDADES DE IMÁGENES ===")
    
    foreground_props = []
    background_props = []
    
    # Analizar foregrounds
    if analysis['foreground_exists'] and analysis['total_foregrounds'] > 0:
        logger.info("📸 Analizando imágenes de foreground...")
        for i, filename in enumerate(analysis['foreground_files'][:5]):  # Analizar primeras 5
            filepath = os.path.join(analysis['foreground_dir'], filename)
            try:
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    props = {
                        'filename': filename,
                        'shape': img.shape,
                        'dtype': str(img.dtype),
                        'channels': len(img.shape) if len(img.shape) == 2 else img.shape[2],
                        'size_bytes': img.nbytes,
                        'has_alpha': len(img.shape) == 3 and img.shape[2] == 4
                    }
                    foreground_props.append(props)
                    logger.info(f"   {i+1}. {filename}: {props['shape']}, {props['channels']} canales, {'con alpha' if props['has_alpha'] else 'sin alpha'}")
            except Exception as e:
                logger.error(f"   Error leyendo {filename}: {e}")
    
    # Analizar backgrounds
    if analysis['background_exists'] and analysis['total_backgrounds'] > 0:
        logger.info("🖼️ Analizando imágenes de background...")
        for i, filename in enumerate(analysis['background_files'][:5]):  # Analizar primeras 5
            filepath = os.path.join(analysis['background_dir'], filename)
            try:
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if img is not None:
                    props = {
                        'filename': filename,
                        'shape': img.shape,
                        'dtype': str(img.dtype),
                        'channels': len(img.shape) if len(img.shape) == 2 else img.shape[2],
                        'size_bytes': img.nbytes
                    }
                    background_props.append(props)
                    logger.info(f"   {i+1}. {filename}: {props['shape']}, {props['channels']} canales")
            except Exception as e:
                logger.error(f"   Error leyendo {filename}: {e}")
    
    analysis['foreground_properties'] = foreground_props
    analysis['background_properties'] = background_props
    
    return analysis

def analyze_dataset_splits(analysis):
    """Analiza cómo se dividiría el dataset en train/val."""
    logger = logging.getLogger('dataset_analyzer')
    
    logger.info("\n=== ANÁLISIS DE DIVISIÓN TRAIN/VAL ===")
    
    if analysis['dataset_size_calculation'] > 0:
        # Según harmonization.py líneas 852-854
        total_size = analysis['dataset_size_calculation']
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        logger.info(f"📊 División del dataset:")
        logger.info(f"   Total samples: {total_size}")
        logger.info(f"   Train samples: {train_size} (80%)")
        logger.info(f"   Val samples: {val_size} (20%)")
        
        # Análisis de batches según configuración actual
        batch_size = 2  # Según harmonization.py línea 808
        
        train_batches = train_size // batch_size
        val_batches = val_size // batch_size
        val_batches_no_drop = (val_size + batch_size - 1) // batch_size  # Con drop_last=False
        
        logger.info(f"\n📦 Análisis de batches (batch_size={batch_size}):")
        logger.info(f"   Train batches: {train_batches}")
        logger.info(f"   Val batches (drop_last=True): {val_batches}")
        logger.info(f"   Val batches (drop_last=False): {val_batches_no_drop}")
        
        if val_batches == 0:
            logger.warning("⚠️ CON drop_last=True, NO HABRÁ BATCHES DE VALIDACIÓN!")
            logger.info("💡 Solución: usar drop_last=False en validación")
        
        analysis.update({
            'total_samples': total_size,
            'train_samples': train_size,
            'val_samples': val_size,
            'batch_size': batch_size,
            'train_batches': train_batches,
            'val_batches_drop_true': val_batches,
            'val_batches_drop_false': val_batches_no_drop
        })
    else:
        logger.warning("❌ No se puede analizar división: dataset vacío")
    
    return analysis

def test_dataset_loading(analysis):
    """Prueba cargar el dataset como lo haría harmonization.py."""
    logger = logging.getLogger('dataset_analyzer')
    
    logger.info("\n=== PRUEBA DE CARGA DEL DATASET ===")
    
    if not (analysis['foreground_exists'] and analysis['background_exists']):
        logger.error("❌ No se puede probar carga: directorios faltantes")
        return analysis
    
    if analysis['total_foregrounds'] == 0 or analysis['total_backgrounds'] == 0:
        logger.error("❌ No se puede probar carga: archivos faltantes")
        return analysis
    
    try:
        # Intentar importar HarmonizationDataset
        from harmonization import HarmonizationDataset
        
        logger.info("🔄 Creando instancia del dataset...")
        dataset = HarmonizationDataset(
            foreground_dir=analysis['foreground_dir'],
            background_dir=analysis['background_dir'],
            transform=None,
            image_size=384
        )
        
        dataset_len = len(dataset)
        logger.info(f"✅ Dataset creado exitosamente")
        logger.info(f"📏 Longitud reportada: {dataset_len}")
        
        # Probar cargar una muestra
        logger.info("🔄 Cargando muestra de prueba...")
        sample = dataset[0]
        composite, target = sample
        
        logger.info(f"✅ Muestra cargada exitosamente")
        logger.info(f"📊 Composite shape: {composite.shape}")
        logger.info(f"📊 Target shape: {target.shape}")
        logger.info(f"📊 Composite dtype: {composite.dtype}")
        logger.info(f"📊 Target dtype: {target.dtype}")
        logger.info(f"📊 Composite range: [{composite.min():.3f}, {composite.max():.3f}]")
        logger.info(f"📊 Target range: [{target.min():.3f}, {target.max():.3f}]")
        
        analysis.update({
            'dataset_loading_success': True,
            'dataset_length': dataset_len,
            'sample_composite_shape': tuple(composite.shape),
            'sample_target_shape': tuple(target.shape),
            'sample_composite_range': [float(composite.min()), float(composite.max())],
            'sample_target_range': [float(target.min()), float(target.max())]
        })
        
    except Exception as e:
        logger.error(f"❌ Error cargando dataset: {e}")
        analysis['dataset_loading_success'] = False
        analysis['dataset_loading_error'] = str(e)
    
    return analysis

def generate_summary_report(analysis, log_filename):
    """Genera un reporte resumen."""
    logger = logging.getLogger('dataset_analyzer')
    
    logger.info("\n" + "="*60)
    logger.info("📋 RESUMEN DEL ANÁLISIS DEL DATASET")
    logger.info("="*60)
    
    # Estado general
    dataset_ok = (analysis.get('foreground_exists', False) and 
                  analysis.get('background_exists', False) and
                  analysis.get('total_foregrounds', 0) > 0 and
                  analysis.get('total_backgrounds', 0) > 0)
    
    logger.info(f"🎯 Estado general del dataset: {'✅ OK' if dataset_ok else '❌ PROBLEMA'}")
    
    # Estadísticas
    logger.info(f"📊 Estadísticas:")
    logger.info(f"   - Foregrounds: {analysis.get('total_foregrounds', 0)}")
    logger.info(f"   - Backgrounds: {analysis.get('total_backgrounds', 0)}")
    logger.info(f"   - Tamaño dataset: {analysis.get('dataset_size_calculation', 0)} samples")
    logger.info(f"   - Train/Val: {analysis.get('train_samples', 0)}/{analysis.get('val_samples', 0)}")
    
    # Problemas identificados
    problems = []
    if not analysis.get('foreground_exists', False):
        problems.append("Directorio foregrounds no existe")
    if not analysis.get('background_exists', False):
        problems.append("Directorio backgrounds no existe")
    if analysis.get('total_foregrounds', 0) == 0:
        problems.append("No hay archivos de foreground")
    if analysis.get('total_backgrounds', 0) == 0:
        problems.append("No hay archivos de background")
    if analysis.get('val_batches_drop_true', 0) == 0:
        problems.append("Sin batches de validación (drop_last=True)")
    if not analysis.get('dataset_loading_success', False):
        problems.append("Error al cargar dataset")
    
    if problems:
        logger.warning("⚠️ Problemas identificados:")
        for i, problem in enumerate(problems, 1):
            logger.warning(f"   {i}. {problem}")
    else:
        logger.info("✅ No se detectaron problemas críticos")
    
    # Recomendaciones
    logger.info("\n💡 Recomendaciones:")
    if analysis.get('val_batches_drop_true', 0) == 0:
        logger.info("   - Usar drop_last=False en validación")
    if analysis.get('total_foregrounds', 0) < 10:
        logger.info("   - Agregar más imágenes de foreground")
    if analysis.get('total_backgrounds', 0) < 10:
        logger.info("   - Agregar más imágenes de background")
    if analysis.get('dataset_size_calculation', 0) < 50:
        logger.info("   - Dataset muy pequeño, considerar más imágenes")
    
    logger.info(f"\n📁 Reporte completo guardado en: {log_filename}")
    
    # Guardar análisis como JSON
    json_filename = log_filename.replace('.log', '_data.json')
    try:
        with open(json_filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"📄 Datos del análisis guardados en: {json_filename}")
    except Exception as e:
        logger.error(f"Error guardando JSON: {e}")

def main():
    """Función principal del análisis."""
    logger, log_filename = setup_logger()
    
    try:
        # Realizar análisis completo
        analysis = analyze_dataset_structure()
        analysis = analyze_image_properties(analysis)
        analysis = analyze_dataset_splits(analysis)
        analysis = test_dataset_loading(analysis)
        
        # Generar reporte final
        generate_summary_report(analysis, log_filename)
        
        logger.info("\n✅ Análisis completado exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error durante el análisis: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()