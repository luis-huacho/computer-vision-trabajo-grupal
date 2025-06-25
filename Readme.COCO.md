# 🏷️ Guía Completa - Dataset COCO para U-Net Background Removal

[![COCO Dataset](https://img.shields.io/badge/Dataset-COCO%202017-blue.svg)](https://cocodataset.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com)

Esta guía te llevará paso a paso para configurar y entrenar el modelo U-Net usando el **dataset COCO 2017** para eliminación de fondos de personas.

## 📋 Índice

- [🎯 Visión General](#-visión-general)
- [📥 Descarga e Instalación](#-descarga-e-instalación)
- [🔍 Verificación del Sistema](#-verificación-del-sistema)
- [🎯 Uso del Sistema](#-uso-del-sistema)
- [⚙️ Configuración del Entrenamiento](#-configuración-del-entrenamiento)
- [📊 Proceso de Entrenamiento](#-proceso-de-entrenamiento)
- [📈 Análisis y Métricas](#-análisis-y-métricas)
- [💡 Optimizaciones y Consejos](#-optimizaciones-y-consejos)
- [🔧 Solución de Problemas](#-solución-de-problemas)
- [📞 Soporte y Contacto](#-soporte-y-contacto)

## 🎯 Visión General

### ¿Por qué COCO Dataset?

El **COCO 2017 Dataset** es ideal para nuestro sistema de eliminación de fondos porque:

- **📊 64,115 imágenes** de entrenamiento con personas
- **🏷️ 149,813 anotaciones** de personas con keypoints
- **🎯 Diversidad excepcional**: Personas en múltiples contextos y poses
- **📐 Anotaciones precisas**: Segmentación de alta calidad
- **🌍 Estándar industrial**: Usado mundialmente para research

### Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Imágenes Total** | 64,115 (train) + 2,693 (val) |
| **Personas Anotadas** | ~149,813 instancias |
| **Promedio Personas/Imagen** | 3.32 |
| **Tamaño Total** | ~13 GB (descomprimido) |
| **Formatos** | JPG (imágenes) + JSON (anotaciones) |

## 📥 Descarga e Instalación

### 1. Preparar Entorno

```bash
# Verificar espacio en disco (mínimo 20GB recomendado)
df -h

# Crear directorio base
mkdir COCO && cd COCO
```

### 2. Descargar Archivos Requeridos

```bash
# Anotaciones (esenciales)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Imágenes de entrenamiento
wget http://images.cocodataset.org/zips/train2017.zip

# Imágenes de validación
wget http://images.cocodataset.org/zips/val2017.zip
```

**Tamaños de descarga:**
- `annotations_trainval2017.zip`: ~241 MB
- `train2017.zip`: ~12.9 GB  
- `val2017.zip`: ~788 MB
- **Total**: ~13.9 GB

### 3. Descomprimir Archivos

```bash
# Descomprimir anotaciones
unzip annotations_trainval2017.zip

# Descomprimir imágenes
unzip train2017.zip
unzip val2017.zip

# Limpiar archivos ZIP (opcional)
rm *.zip

# Volver al directorio principal
cd ..
```

### 4. Verificar Estructura Final

```bash
# La estructura debe ser:
COCO/
├── annotations/
│   ├── person_keypoints_train2017.json    # ~277 MB
│   ├── person_keypoints_val2017.json      # ~11 MB
│   └── [otros archivos de anotaciones]
├── train2017/                             # 64,115 imágenes .jpg
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
└── val2017/                               # 2,693 imágenes .jpg
    ├── 000000000139.jpg
    └── ...
```

## 🔍 Verificación del Sistema

### Comandos de Verificación

```bash
# Verificación rápida de estructura
python main.py quick

# Verificación completa del sistema
python main.py verify

# Análisis detallado del dataset
python main.py analyze
```

### Salidas Esperadas

#### Verificación Rápida (`python main.py quick`)
```
=== VERIFICACIÓN RÁPIDA DE ESTRUCTURA COCO ===

✅ Directorio principal encontrado: COCO
✅ Directorio de anotaciones encontrado: COCO/annotations
✅ Anotaciones de entrenamiento encontradas: COCO/annotations/person_keypoints_train2017.json (277.1 MB)
✅ Anotaciones de validación encontradas: COCO/annotations/person_keypoints_val2017.json (11.2 MB)
✅ Directorio train2017 encontrado con 64,115 imágenes
✅ Directorio val2017 encontrado con 2,693 imágenes

✅ Estructura COCO verificada correctamente!
   ✅ Todos los archivos necesarios están presentes
   ✅ Listo para entrenar el modelo
```

#### Análisis del Dataset (`python main.py analyze`)
```
=== ANÁLISIS DE ANOTACIONES COCO ===

📈 Estadísticas generales:
   - Total de imágenes: 64,115
   - Total de anotaciones: 262,465
   - Anotaciones de personas válidas: 149,813
   - Imágenes con personas válidas: 45,174
   - Promedio de personas por imagen: 3.32
   - Máximo de personas en una imagen: 13

📏 Distribución de tamaños (área):
   - Área promedio: 17,889 píxeles²
   - Área mínima: 544 píxeles²
   - Área máxima: 594,010 píxeles²
   - Mediana: 6,421 píxeles²

✅ Análisis completado. Dataset listo para entrenamiento.
```

## 🎯 Uso del Sistema

### Comandos Disponibles

| Comando | Descripción | Tiempo |
|---------|-------------|--------|
| `python main.py` | **Modo automático** - Verificación + entrenamiento | 2-4 horas |
| `python main.py verify` | Verificación completa del sistema | 2-3 minutos |
| `python main.py quick` | Verificación rápida de estructura COCO | 30 segundos |
| `python main.py analyze` | Análisis estadístico del dataset | 1-2 minutos |
| `python main.py batch` | Prueba de carga de datos | 30 segundos |
| `python main.py train` | Entrenamiento directo (sin verificación) | 2-4 horas |

### Flujo de Trabajo Recomendado

#### 1. Primera Vez - Verificación Completa
```bash
# Verificar todo el sistema
python main.py verify

# Si todo está bien, continuar con entrenamiento
python main.py train
```

#### 2. Verificación Rápida
```bash
# Solo verificar estructura de archivos
python main.py quick
```

#### 3. Análisis del Dataset
```bash
# Ver estadísticas detalladas
python main.py analyze
```

**Salida esperada:**
```
📈 Estadísticas generales:
   - Total de imágenes: 64,115
   - Total de anotaciones: 262,465
   - Anotaciones de personas válidas: 149,813
   - Imágenes con personas válidas: 45,174
   - Promedio de personas por imagen: 3.32
```

#### 4. Entrenamiento Automático (Recomendado)
```bash
# Verificación + entrenamiento con confirmación
python main.py
```

#### 5. Solo Entrenamiento
```bash
# Saltar verificaciones e ir directo al entrenamiento
python main.py train
```

## ⚙️ Configuración del Entrenamiento

### Parámetros por Defecto (main.py)

```python
config = {
    'batch_size': 16,           # Reducido para COCO (imágenes más variadas)
    'learning_rate': 1e-4,      # Learning rate conservador
    'weight_decay': 1e-6,       # Regularización ligera
    'num_epochs': 100,          # Épocas de entrenamiento
    'image_size': 384,          # Tamaño de procesamiento
    'num_workers': 8,           # Trabajadores para carga de datos
    'pin_memory': True,         # Optimización de memoria
}
```

### Personalizar Configuración

```python
# Para GPUs con poca memoria (≤6GB VRAM)
config['batch_size'] = 8
config['image_size'] = 256

# Para entrenamiento rápido
config['num_epochs'] = 50
config['learning_rate'] = 2e-4

# Para máxima calidad
config['batch_size'] = 32
config['image_size'] = 512
config['num_epochs'] = 200
```

### Configuración por Hardware

| Hardware | Batch Size | Image Size | Workers | Tiempo/Época |
|----------|------------|------------|---------|--------------|
| **RTX 4090** | 32 | 512 | 12 | ~8 min |
| **RTX 3080** | 24 | 384 | 8 | ~12 min |
| **GTX 1080** | 16 | 384 | 6 | ~18 min |
| **GTX 1060** | 8 | 256 | 4 | ~35 min |
| **CPU Only** | 4 | 256 | 2 | ~4 horas |

## 📊 Proceso de Entrenamiento

### Etapas del Entrenamiento

1. **Carga del Dataset** (2-3 minutos)
   - Lectura de anotaciones COCO
   - Filtrado de personas válidas
   - Creación de dataloaders

2. **Inicialización** (30 segundos)
   - Carga del modelo U-Net
   - Configuración de optimizador
   - Preparación de funciones de pérdida

3. **Entrenamiento** (2-4 horas)
   - 100 épocas por defecto
   - Validación cada época
   - Guardado automático del mejor modelo

4. **Finalización** (1 minuto)
   - Guardado del modelo final
   - Generación de gráficas
   - Resumen de métricas

### Monitoreo del Progreso

```bash
# Ver logs en tiempo real
tail -f logs/training_*.log

# Monitorear GPU
watch -n 2 nvidia-smi

# Verificar checkpoints guardados
ls -la checkpoints/
```

### Salida Típica del Entrenamiento

```
🚀 INICIANDO ENTRENAMIENTO U-NET
===================================

📊 Configuración:
   - Dataset: COCO 2017
   - Batch Size: 16
   - Learning Rate: 1e-4
   - Épocas: 100
   - Dispositivo: cuda:0

🔄 Cargando dataset COCO...
   ✅ Train: 45,174 imágenes con personas
   ✅ Val: 2,693 imágenes con personas

📈 Época 1/100:
   Train - Loss: 0.4521, Acc: 0.8234
   Val - Loss: 0.3876, IoU: 0.7543, Dice: 0.8012
   ⏱️ Tiempo: 12m 34s

📈 Época 2/100:
   Train - Loss: 0.3287, Acc: 0.8567
   Val - Loss: 0.2943, IoU: 0.8012, Dice: 0.8456
   ⏱️ Tiempo: 12m 28s

...

🎉 ¡Entrenamiento completado!
   📊 Mejor IoU: 0.8734 (Época 87)
   💾 Modelo guardado: checkpoints/best_model.pth
```

## 📈 Análisis y Métricas

### Métricas Principales

| Métrica | Descripción | Valor Objetivo |
|---------|-------------|----------------|
| **IoU** | Intersection over Union | ≥0.85 |
| **Dice Score** | Similaridad entre máscaras | ≥0.90 |
| **Pixel Accuracy** | Precisión a nivel de píxel | ≥0.95 |
| **Loss** | Función de pérdida | <0.15 |

### Gráficas Generadas

El sistema genera automáticamente:

- **`plots/training_curves.png`**: Curvas de pérdida y métricas
- **`plots/sample_predictions.png`**: Predicciones de muestra
- **`plots/confusion_matrix.png`**: Matriz de confusión
- **`plots/metrics_evolution.png`**: Evolución de métricas por época

### Análisis de Calidad Automático

```python
# El sistema evalúa automáticamente:
quality_metrics = {
    'coverage_threshold': 15.0,    # % mínimo de cobertura de persona
    'contrast_threshold': 60.0,    # Contraste mínimo de máscara
    'edge_threshold': 50.0,        # Definición mínima de bordes
    'resolution_threshold': 70.0   # Score mínimo de resolución
}
```

## 💡 Optimizaciones y Consejos

### Para Mejor Rendimiento

1. **SSD**: Usa SSD para almacenar COCO (mejora velocidad de carga)
2. **RAM**: 16+ GB recomendado para batch_size > 16
3. **GPU**: NVIDIA con 8+ GB VRAM para máxima velocidad
4. **Monitoring**: Usa `watch nvidia-smi` durante entrenamiento

### Para Mejor Calidad

1. **Épocas**: Mínimo 100 épocas para convergencia
2. **Learning Rate**: Usar 1e-4 o menos para estabilidad
3. **Image Size**: 384+ píxeles para mejor detalle
4. **Patience**: Esperar convergencia completa

### Para Desarrollo

1. **Verificación**: Siempre usar `python main.py verify` primero
2. **Logs**: Monitorear logs para detectar problemas temprano
3. **Checkpoints**: El mejor modelo se guarda automáticamente
4. **Resumir**: Puedes resumir entrenamiento desde `last_model.pth`

### Uso con Screen (Sesiones Largas)

```bash
# Crear sesión para entrenamiento largo
screen -S unet_training

# Ejecutar entrenamiento
python main.py train

# Despegarse (Ctrl+A, luego D)
# Reconectar más tarde
screen -r unet_training
```

### Optimización de Memoria

```python
# Para GPUs con poca memoria
config = {
    'batch_size': 8,           # Reducir batch size
    'image_size': 256,         # Reducir resolución
    'pin_memory': False,       # Desactivar pin memory
    'num_workers': 4,          # Menos workers
    'gradient_accumulation': 2  # Acumular gradientes
}
```

### Configuración Avanzada

```python
# Configuración para investigación
research_config = {
    'model_variant': 'attention_unet',  # Usar Attention U-Net
    'loss_function': 'focal_dice',      # Loss híbrido
    'optimizer': 'adamw',               # Optimizador avanzado
    'scheduler': 'cosine_annealing',    # Scheduler de learning rate
    'augmentation_level': 'heavy',      # Aumentación intensiva
    'mixed_precision': True,            # Entrenamiento mixto
}
```

## 🔧 Solución de Problemas

### Problemas de Dataset

#### "Dataset no encontrado"

```bash
# Verificar estructura
python main.py quick

# Salida esperada:
# ✅ Directorio principal encontrado: COCO
# ✅ Anotaciones de entrenamiento encontradas: ...
```

**Solución:**
```bash
# Verificar que los archivos existan
ls COCO/annotations/person_keypoints_train2017.json
ls COCO/train2017/ | head -5

# Si faltan, re-descargar
cd COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

#### "Anotaciones corruptas"

```bash
# Verificar integridad JSON
python -c "import json; json.load(open('COCO/annotations/person_keypoints_train2017.json'))"
```

**Solución:**
```bash
# Re-descargar solo anotaciones
cd COCO
rm -rf annotations/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Problemas de Memoria

#### "CUDA out of memory"

```bash
# Error típico:
# RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Soluciones:**
```python
# 1. Reducir batch_size
config['batch_size'] = 8  # O incluso 4

# 2. Reducir image_size
config['image_size'] = 256

# 3. Usar gradient accumulation
config['gradient_accumulation_steps'] = 2

# 4. Limpiar cache
torch.cuda.empty_cache()
```

#### "RAM insuficiente"

```bash
# Síntomas: Sistema muy lento, swap usage alto
```

**Soluciones:**
```python
# Reducir workers
config['num_workers'] = 2

# Desactivar pin_memory
config['pin_memory'] = False

# Usar batch_size menor
config['batch_size'] = 4
```

### Problemas de Entrenamiento

#### "Loss no converge"

```bash
# Síntomas: Loss se mantiene alto después de muchas épocas
```

**Diagnóstico:**
```bash
# Verificar datos
python main.py analyze

# Ver distribución de dataset
python main.py batch
```

**Soluciones:**
```python
# 1. Reducir learning rate
config['learning_rate'] = 5e-5

# 2. Aumentar épocas
config['num_epochs'] = 200

# 3. Cambiar optimizador
config['optimizer'] = 'sgd'
config['momentum'] = 0.9

# 4. Verificar augmentación
config['augmentation'] = 'light'  # Menos agresiva
```

#### "Overfitting"

```bash
# Síntomas: Train loss baja, val loss alta
```

**Soluciones:**
```python
# 1. Aumentar regularización
config['weight_decay'] = 1e-4

# 2. Dropout
config['dropout'] = 0.3

# 3. Early stopping
config['early_stopping_patience'] = 10

# 4. Más augmentación
config['augmentation'] = 'heavy'
```

### Problemas de Sistema

#### "Python module not found"

```bash
# Error: ModuleNotFoundError: No module named 'torch'
```

**Solución:**
```bash
# Verificar entorno virtual
which python
pip list | grep torch

# Reinstalar dependencias
pip install -r requirements.txt

# O forzar reinstalación
pip install --force-reinstall torch torchvision
```

#### "GPU no detectada"

```bash
# Verificar CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solución:**
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar CUDA toolkit
nvcc --version

# Reinstalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problemas de Rendimiento

#### "Entrenamiento muy lento"

**Diagnóstico:**
```bash
# Verificar uso de GPU
nvidia-smi

# Verificar IO del disco
iostat -x 1

# Verificar carga de CPU
htop
```

**Optimizaciones:**
```python
# 1. Aumentar workers si CPU/IO permite
config['num_workers'] = 12

# 2. Usar pin_memory si hay RAM suficiente
config['pin_memory'] = True

# 3. Optimizar dataloader
config['persistent_workers'] = True
config['prefetch_factor'] = 2

# 4. Usar mixed precision
config['mixed_precision'] = True
```

#### "Carga de dataset lenta"

**Soluciones:**
```bash
# 1. Mover COCO a SSD
mv COCO /path/to/ssd/COCO
ln -s /path/to/ssd/COCO ./COCO

# 2. Precargar datos en RAM (si es posible)
# Crear dataset en memoria
python -c "
import torch
from datasets import COCOPersonDataset
dataset = COCOPersonDataset(preload=True)
torch.save(dataset, 'preloaded_dataset.pt')
"
```

## 📞 Soporte y Contacto

### Problemas Frecuentes

- **Dataset no encontrado**: Verificar estructura con `python main.py quick`
- **Memoria insuficiente**: Reducir `batch_size` y/o `image_size`
- **Entrenamiento lento**: Verificar uso de GPU con `nvidia-smi`
- **Calidad baja**: Aumentar épocas y/o tamaño de imagen

### Checklist de Verificación

Antes de reportar problemas, verificar:

- [ ] Estructura COCO correcta (`python main.py quick`)
- [ ] Dependencias instaladas (`pip list | grep torch`)
- [ ] GPU disponible (`nvidia-smi`)
- [ ] Espacio en disco suficiente (`df -h`)
- [ ] Configuración apropiada para tu hardware
- [ ] Logs de error completos

### Recursos Adicionales

- 📖 **Documentación Original**: `README.md`
- 🎭 **Aplicación Web**: `README-app.md`
- 🔧 **Código Principal**: `main.py` (adaptado para COCO)
- 📊 **Dataset COCO**: [cocodataset.org](https://cocodataset.org/)
- 🛠️ **Utilidades**: `Docs/Utils.md`

### Logs Importantes

```bash
# Logs de entrenamiento
ls logs/training_*.log

# Logs de verificación
ls logs/verification_*.log

# Logs de sistema
ls logs/system_*.log

# Ver últimos errores
grep -i error logs/*.log | tail -10
```

### Desarrolladores

**Luis Huacho y Dominick Alvarez**  
Maestría en Informática - PUCP  
Especialización en Computer Vision y Deep Learning

**Contacto:**
- 📧 Email: [contacto-disponible-en-repositorio]
- 🔗 GitHub: [link-del-repositorio]
- 📚 Documentación: Ver archivos README adicionales

---

## 🏁 Inicio Rápido - 5 Pasos

```bash
# 1. Clonar y preparar entorno
git clone <repo> && cd unet-background-removal
python -m venv venv && source venv/bin/activate
pip install torch torchvision opencv-python albumentations numpy matplotlib streamlit

# 2. Descargar COCO (si no lo tienes)
mkdir COCO && cd COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip annotations_trainval2017.zip && unzip train2017.zip && unzip val2017.zip
cd ..

# 3. Verificar todo
python main.py verify

# 4. Entrenar modelo
python main.py train

# 5. Usar aplicación web
streamlit run app.py
```

¡El sistema está listo para funcionar con tu dataset COCO! 🚀

### Comandos de Emergencia

```bash
# Si algo sale mal, estos comandos te salvarán:

# Verificación ultra-rápida
python main.py quick

# Limpiar y reiniciar
rm -rf checkpoints/ logs/ plots/
python main.py verify

# Forzar reinstalación
pip install --force-reinstall -r requirements.txt

# Entrenamiento con configuración mínima
python -c "
import main
config = {'batch_size': 4, 'image_size': 256, 'num_epochs': 10}
main.train_model(config)
"
```

**¿Aún tienes problemas?** Revisa la sección de Solución de Problemas o consulta los logs en `logs/` para más detalles.

**¿Primera vez con COCO?** Este README tiene todo lo que necesitas. ¡Síguelo paso a paso!

**¿Listo para producción?** Una vez entrenado, consulta `README-app.md` para desplegar la aplicación web.

---

**Tip Pro**: Usa `screen` para entrenamientos largos y `watch nvidia-smi` para monitorear tu GPU. ¡El entrenamiento puede tomar horas, pero el resultado vale la pena! 💪