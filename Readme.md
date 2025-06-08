# U-Net Autoencoder para Remoción de Fondo

Un modelo de deep learning que utiliza arquitectura U-Net con Attention Gates para remover automáticamente el fondo de imágenes, manteniendo únicamente las personas detectadas.

**Desarrollado por:** Luis Huacho y Dominick Alvarez  
**Institución:** Maestría en Informática, PUCP

## 🎯 Características Principales

- **Arquitectura Híbrida**: U-Net con Autoencoder para segmentación y reconstrucción
- **Attention Gates**: Enfoque automático en regiones de personas
- **Transfer Learning**: ResNet34 pre-entrenado como backbone
- **Múltiples Funciones de Pérdida**: BCE, Dice, Perceptual y Edge Loss
- **Preservación de Dimensiones**: Mantiene el tamaño original de la imagen

## 🏗️ Arquitectura del Modelo

```
Input (RGB) → ResNet34 Encoder → Bottleneck → Attention Decoder → Output (RGBA)
             ↓                                ↑
        Skip Connections ──────────────────────┘
```

### Componentes Clave
- **UNetEncoder**: ResNet34 con skip connections [64, 64, 128, 256, 512] canales
- **UNetDecoder**: Upsampling con Attention Gates
- **ImageProcessor**: Redimensionamiento con padding que preserva proporciones
- **LossCalculator**: Pérdida compuesta optimizada para segmentación

## 📋 Requisitos

```
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

## 🚀 Instalación y Uso

### 1. Configuración del Entorno

```bash
# Clonar repositorio
git clone <repository-url>
cd unet-background-removal

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparar Dataset

Descargar el dataset Supervisely Persons y estructurarlo:

```
persons/
└── project/
    ├── ds1/
    │   ├── img/       # Imágenes originales
    │   └── ann/       # Anotaciones JSON
    ├── ds2/
    └── ...
```

### 3. Entrenamiento

#### Opción A: Script de Entrenamiento (Recomendado)

```bash
# Entrenamiento normal con logs en archivo
python run_training.py

# Entrenamiento con logs en tiempo real
python run_training.py --verbose

# Sin colores en la salida
python run_training.py --no-color
```

#### Opción B: Entrenamiento Directo

```bash
# Ejecutar directamente el código principal
python main.py
```

### 4. Aplicación Web

```bash
# Ejecutar interfaz de Streamlit
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
unet-background-removal/
├── main.py                # Modelo principal y entrenamiento
├── run_training.py        # Script automatizado de entrenamiento
├── app.py                 # Aplicación Streamlit
├── README.md             # Este archivo
├── README-app.md         # Documentación de la app
├── requirements.txt      # Dependencias
├── persons/              # Dataset Supervisely
├── checkpoints/          # Modelos guardados
│   ├── best_model.pth   # Mejor modelo
│   └── YYYYMMDD_HHMMSS/ # Checkpoints con timestamp
├── plots/               # Gráficas de entrenamiento
│   └── YYYYMMDD_HHMMSS/ # Plots con timestamp
└── logs/               # Logs de entrenamiento
```

## 🔧 Configuración

### Parámetros de Entrenamiento (main.py)

```python
config = {
    'batch_size': 4,
    'learning_rate': 5e-5,
    'weight_decay': 1e-6,
    'num_epochs': 100,
    'image_size': 384,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### Funciones de Pérdida

- **BCE Loss** (α=1.0): Clasificación binaria del canal alpha
- **Dice Loss** (β=1.0): Similitud entre máscaras
- **Perceptual Loss** (γ=0.5): Calidad visual de canales RGB
- **Edge Loss** (δ=0.3): Preservación de contornos

## 📊 Resultados Esperados

### Métricas Objetivo
- **IoU**: > 0.85 en validación
- **Dice Coefficient**: > 0.90 en validación
- **Pixel Accuracy**: > 0.95 en validación

### Rendimiento
- **Entrenamiento**: 100 épocas en ~4-6 horas (RTX 3080)
- **Inferencia**: ~50ms por imagen (256x256) en GPU
- **Convergencia**: Típicamente en 50-70 épocas

## 🛠️ Características Técnicas

### Procesamiento de Imágenes
- **Redimensionamiento Inteligente**: Mantiene proporciones con padding
- **Restauración Exacta**: Regresa al tamaño original sin pérdida
- **Aumentación de Datos**: Flip, rotación, cambios de brillo/contraste

### Arquitectura Avanzada
- **Attention Mechanisms**: Focalización automática en personas
- **Skip Connections**: Preservación de detalles de alta resolución
- **Gradient Clipping**: Estabilización del entrenamiento
- **Cosine Annealing**: Scheduler de learning rate optimizado

## 🔍 Uso del Modelo

### Entrenamiento

```python
# El sistema automáticamente:
# 1. Carga y procesa el dataset Supervisely
# 2. Aplica aumentación de datos
# 3. Entrena con validación cruzada
# 4. Guarda el mejor modelo y checkpoints
# 5. Genera gráficas de progreso

# Ejecutar:
python run_training.py
```

### Inferencia

```python
from main import ModelInference

# Cargar modelo entrenado
inference = ModelInference('checkpoints/best_model.pth')

# Procesar imagen individual
result = inference.remove_background('input.jpg', 'output.png')

# Procesamiento en lote
inference.batch_process('input_dir/', 'output_dir/')
```

## 🛠️ Solución de Problemas

### Error de Memoria GPU
```python
# Reducir batch size en config
config['batch_size'] = 2
```

### Dataset No Encontrado
```bash
# Verificar estructura
ls persons/project/ds1/img/
ls persons/project/ds1/ann/
```

### Problemas de Convergencia
```python
# Ajustar learning rate
config['learning_rate'] = 1e-5
```

## 📚 Referencias

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Attention U-Net: Learning Where to Look](https://arxiv.org/abs/1804.03999)
- [Supervisely Persons Dataset](https://github.com/supervisely-ecosystem/persons)

## 🤝 Contribuciones

Desarrollado como parte de la investigación en Computer Vision y Deep Learning en la Maestría en Informática de la PUCP.

**Autores:**
- Luis Huacho
- Dominick Alvarez

## 📄 Licencia

Este proyecto está bajo la Licencia MIT para fines académicos y de investigación.