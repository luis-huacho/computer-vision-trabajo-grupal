# U-Net Autoencoder para Remoción de Fondo - Versión COCO

Un modelo de deep learning que utiliza arquitectura U-Net con Attention Gates para remover automáticamente el fondo de imágenes, manteniendo únicamente las personas detectadas. **Versión adaptada para dataset COCO**.

**Desarrollado por:** Luis Huacho y Dominick Alvarez  
**Institución:** Maestría en Informática, PUCP  
**Dataset:** COCO 2017 - Person Keypoints

## 🎯 Características Principales

- **Arquitectura Híbrida**: U-Net con Autoencoder para segmentación y reconstrucción
- **Dataset COCO**: Entrenado con COCO 2017 Person Keypoints (118K+ imágenes)
- **Attention Gates**: Enfoque automático en regiones de personas
- **Transfer Learning**: ResNet34 pre-entrenado como backbone
- **Segmentación Avanzada**: Usa anotaciones de segmentación COCO cuando están disponibles
- **Preservación de Dimensiones**: Mantiene el tamaño original de la imagen

## 📋 Requisitos del Sistema

### Software Requerido
```bash
Python >= 3.8
CUDA >= 11.0 (opcional, para GPU)
```

### Dependencias Python
```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
albumentations>=1.0.0
numpy>=1.21.0
matplotlib>=3.4.0
streamlit>=1.25.0
scikit-learn>=0.24.0
Pillow>=8.3.0
```

## 🚀 Instalación y Configuración

### 1. Clonar Repositorio

```bash
git clone <repository-url>
cd unet-background-removal
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Activar entorno (Windows)
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
# Instalar dependencias básicas
pip install -r requirements.txt

# O instalar manualmente
pip install torch torchvision opencv-python albumentations numpy matplotlib streamlit scikit-learn Pillow
```

### 4. Preparar Dataset COCO

#### Opción A: Descarga Automática (Recomendado)

```bash
# Crear directorio COCO
mkdir COCO
cd COCO

# Descargar anotaciones (253 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Descargar imágenes de entrenamiento (18 GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Descargar imágenes de validación (778 MB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

cd ..
```

#### Opción B: Si ya tienes los archivos

```bash
# Asegúrate de que la estructura sea:
COCO/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
│   └── *.jpg (118,287 archivos)
└── val2017/
    └── *.jpg (5,000 archivos)
```

### 5. Verificar Instalación

```bash
# Verificación rápida de estructura
python main.py quick

# Verificación completa del sistema
python main.py verify

# Análisis detallado del dataset
python main.py analyze
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

4. **Finalización**
   - Generación de gráficas
   - Guardado de checkpoints
   - Resumen de métricas

### Métricas de Evaluación

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| **IoU** | > 0.85 | Intersection over Union |
| **Dice** | > 0.90 | Coeficiente de Dice |
| **Pixel Accuracy** | > 0.95 | Precisión a nivel de píxel |
| **Loss** | < 0.1 | Pérdida compuesta |

### Monitoreo del Entrenamiento

```bash
# Ver logs en tiempo real
tail -f logs/training_YYYYMMDD_HHMMSS.log

# Ver progreso en archivos
ls -la checkpoints/
ls -la plots/
```

## 🔧 Solución de Problemas

### Problemas Comunes y Soluciones

#### ❌ "Directorio COCO no encontrado"
```bash
# Verificar estructura
ls -la COCO/
ls -la COCO/annotations/
ls -la COCO/train2017/ | head
ls -la COCO/val2017/ | head

# Si falta, descargar dataset
python main.py quick  # Te mostrará qué falta
```

#### ❌ "CUDA out of memory"
```python
# Reducir batch size en main.py línea ~XXX
config['batch_size'] = 8  # En lugar de 16
config['image_size'] = 256  # En lugar de 384
```

#### ❌ "Dataset vacío"
```bash
# Verificar anotaciones
python main.py analyze

# Debería mostrar:
# - Anotaciones de personas válidas: 149,813
# - Imágenes con personas válidas: 45,174
```

#### ❌ "Error cargando batch"
```bash
# Probar carga individual
python main.py batch

# Reducir workers si hay problemas
config['num_workers'] = 0  # En main.py
```

#### ❌ "Entrenamiento muy lento"
```bash
# Verificar que usa GPU
nvidia-smi  # Debería mostrar uso de GPU

# Si usa CPU, verificar instalación CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Comandos de Diagnóstico

```bash
# Verificación completa del sistema
python main.py verify

# Ver uso de recursos durante entrenamiento
watch -n 1 nvidia-smi  # Para GPU
watch -n 1 'free -h && df -h'  # Para RAM y disco

# Verificar logs de errores
tail -f logs/training_*.log | grep ERROR

# Limpiar memoria si es necesario
python -c "import torch; torch.cuda.empty_cache()"
```

## 📁 Estructura de Archivos Generados

```
unet-background-removal/
├── COCO/                     # Dataset COCO (19+ GB)
│   ├── annotations/
│   ├── train2017/
│   └── val2017/
├── checkpoints/              # Modelos entrenados
│   ├── best_model.pth       # Mejor modelo (principal)
│   ├── last_model.pth       # Último checkpoint
│   └── YYYYMMDD_HHMMSS/     # Checkpoints con timestamp
├── plots/                   # Gráficas de entrenamiento
│   ├── training_history.png
│   └── YYYYMMDD_HHMMSS/     # Plots con timestamp
├── logs/                    # Logs de entrenamiento
│   └── training_*.log
├── main.py                  # Código principal (ADAPTADO PARA COCO)
├── app.py                   # Aplicación Streamlit
├── run_training.py          # Script automatizado
└── README.COCO.md          # Esta documentación
```

## 🚀 Después del Entrenamiento

### 1. Verificar Resultados

```bash
# Verificar que el modelo se guardó
ls -la checkpoints/best_model.pth

# Ver gráficas de entrenamiento
ls -la plots/training_history.png
```

### 2. Usar el Modelo Entrenado

#### Aplicación Web (Recomendado)
```bash
# Ejecutar interfaz Streamlit
streamlit run app.py

# Abrir en navegador: http://localhost:8501
```

#### Inferencia por Código
```python
from main import ModelInference

# Cargar modelo
inference = ModelInference('checkpoints/best_model.pth')

# Procesar imagen individual
result = inference.remove_background('input.jpg', 'output.png')

# Procesamiento en lote
inference.batch_process('input_dir/', 'output_dir/')
```

#### Script de Entrenamiento Automatizado
```bash
# Usar script con logs organizados
python run_training.py

# Con logs en tiempo real
python run_training.py --verbose
```

### 3. Evaluar Calidad

| Métrica | Valor Esperado | Significado |
|---------|----------------|-------------|
| **Train IoU** | > 0.85 | Modelo aprende correctamente |
| **Val IoU** | > 0.80 | Buena generalización |
| **Diferencia Train-Val** | < 0.10 | Sin overfitting |
| **Convergencia** | 50-70 épocas | Entrenamiento eficiente |

## 📊 Diferencias vs Dataset Supervisely

| Aspecto | Supervisely (Original) | COCO (Esta Versión) |
|---------|----------------------|-------------------|
| **Imágenes de Entrenamiento** | ~8,000 | ~45,000 |
| **Calidad de Anotaciones** | Muy alta | Alta |
| **Variedad de Poses** | Media | Muy alta |
| **Variedad de Fondos** | Media | Muy alta |
| **Tamaño de Dataset** | 2-3 GB | 19+ GB |
| **Tiempo de Entrenamiento** | 2-3 horas | 3-5 horas |
| **Calidad Esperada** | Excelente | Muy buena |

## 🎯 Optimizaciones para COCO

### Configuración para Diferentes Escenarios

#### 🚀 Entrenamiento Rápido (Prototipo)
```python
config = {
    'batch_size': 32,
    'learning_rate': 2e-4,
    'num_epochs': 30,
    'image_size': 256,
}
# Tiempo: ~45 minutos en GPU
```

#### ⚖️ Entrenamiento Balanceado (Recomendado)
```python
config = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'image_size': 384,
}
# Tiempo: ~3 horas en GPU
```

#### 🎯 Máxima Calidad (Producción)
```python
config = {
    'batch_size': 8,
    'learning_rate': 5e-5,
    'num_epochs': 200,
    'image_size': 512,
}
# Tiempo: ~8 horas en GPU
```

## 🔍 Validación y Testing

### Scripts de Validación

```bash
# Validación completa antes de entrenar
python main.py verify

# Verificar solo estructura COCO
python main.py quick

# Análisis estadístico del dataset
python main.py analyze

# Probar carga de un batch
python main.py batch
```

### Verificaciones Automáticas

El sistema incluye verificaciones automáticas para:

- ✅ Estructura de directorios COCO
- ✅ Presencia de archivos de anotaciones
- ✅ Integridad de imágenes
- ✅ Forward pass del modelo
- ✅ Carga de datos sin errores
- ✅ Compatibilidad GPU/CPU

## 💡 Consejos y Mejores Prácticas

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

## 📞 Soporte y Contacto

### Problemas Frecuentes

- **Dataset no encontrado**: Verificar estructura con `python main.py quick`
- **Memoria insuficiente**: Reducir `batch_size` y/o `image_size`
- **Entrenamiento lento**: Verificar uso de GPU con `nvidia-smi`
- **Calidad baja**: Aumentar épocas y/o tamaño de imagen

### Recursos Adicionales

- 📖 **Documentación Original**: `README.md`
- 🎭 **Aplicación Web**: `README-app.md`
- 🔧 **Código Principal**: `main.py` (adaptado para COCO)
- 📊 **Dataset COCO**: [cocodataset.org](https://cocodataset.org/)

### Desarrolladores

**Luis Huacho y Dominick Alvarez**  
Maestría en Informática - PUCP  
Especialización en Computer Vision y Deep Learning

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