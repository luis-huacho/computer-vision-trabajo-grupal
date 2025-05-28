# U-Net Autoencoder para Remoción de Fondo

Un modelo de deep learning avanzado que utiliza arquitectura U-Net con Attention Gates para remover automáticamente el fondo de imágenes, manteniendo únicamente las personas detectadas.

## 🎯 Características Principales

- **Arquitectura Híbrida**: Combina U-Net con Autoencoder para segmentación y reconstrucción de alta calidad
- **Attention Mechanisms**: Implementa Attention Gates para enfocarse en regiones de personas
- **Transfer Learning**: Utiliza ResNet34 pre-entrenado como backbone del encoder
- **Múltiples Funciones de Pérdida**: Combina BCE, Dice, Perceptual y Edge Loss para resultados superiores
- **Sistema de Checkpoints**: Guarda automáticamente las mejores versiones del modelo
- **Logging Completo**: Registro detallado del proceso de entrenamiento
- **Métricas Avanzadas**: IoU, Dice Coefficient, Pixel Accuracy y más

## 🏗️ Arquitectura del Modelo

### Encoder Path (Downsampling)
```
Input (256x256x3) → ResNet34 Backbone → Bottleneck (16x16x1024)
                                     ↓
Skip Connections: [64, 64, 128, 256, 512] channels
```

### Decoder Path (Upsampling)
```
Bottleneck → Attention Gates + Skip Connections → Output (256x256x4)
           ↓
RGBA Output: RGB channels + Alpha mask
```

### Componentes Clave
- **Attention Blocks**: Enfocan el modelo en regiones relevantes
- **Double Convolution**: Bloques conv-bn-relu-conv-bn-relu
- **Skip Connections**: Preservan detalles de alta resolución
- **Multi-Scale Loss**: Optimiza a diferentes niveles de resolución

## 📋 Requisitos

### Dependencias Principales
```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
albumentations>=1.0.0
numpy>=1.21.0
matplotlib>=3.4.0
Pillow>=8.3.0
scikit-learn>=0.24.0
```

### Hardware Recomendado
- **GPU**: NVIDIA GTX 1080 Ti o superior (8GB+ VRAM)
- **RAM**: 16GB+ para procesamiento de datasets grandes
- **Almacenamiento**: 20GB+ para dataset y checkpoints

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/unet-background-removal.git
cd unet-background-removal
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar dataset Supervisely Persons**
```bash
# Descargar desde: https://github.com/supervisely-ecosystem/persons
# Extraer en: data/supervisely_persons/
```

## 📁 Estructura del Proyecto

```
unet-background-removal/
├── main.py                 # Modelo principal y entrenamiento
├── README.md              # Este archivo
├── concepts.md            # Conceptos técnicos y métricas
├── requirements.txt       # Dependencias
├── data/
│   └── supervisely_persons/
│       ├── images/        # Imágenes originales
│       └── annotations/   # Anotaciones JSON
├── checkpoints/           # Modelos guardados
│   ├── best_model.pth    # Mejor modelo
│   └── last_model.pth    # Último checkpoint
├── logs/                 # Logs de entrenamiento
├── plots/               # Gráficas de entrenamiento
└── output/             # Resultados de inferencia
```

## 🎓 Uso

### Entrenamiento

```python
# Configurar parámetros
config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'image_size': 256
}

# Ejecutar entrenamiento
python main.py
```

### Inferencia

```python
from main import ModelInference

# Cargar modelo entrenado
inference = ModelInference('checkpoints/best_model.pth')

# Remover fondo de una imagen
result = inference.remove_background('input.jpg', 'output.png')

# Procesamiento en lote
inference.batch_process('input_dir/', 'output_dir/')
```

## 📊 Métricas de Evaluación

### Métricas de Segmentación
- **IoU (Intersection over Union)**: Mide la superposición entre predicción y ground truth
- **Dice Coefficient**: Similitud entre máscaras binarias
- **Pixel Accuracy**: Precisión a nivel de pixel

### Métricas de Calidad Visual
- **BCE Loss**: Binary Cross Entropy para clasificación
- **Perceptual Loss**: Preserva características visuales naturales
- **Edge Loss**: Mantiene contornos nítidos

## 🔧 Configuración Avanzada

### Parámetros del Modelo
```python
model = UNetAutoencoder(
    pretrained=True,      # Usar ResNet pre-entrenado
    use_attention=True    # Activar Attention Gates
)
```

### Funciones de Pérdida
```python
loss_weights = {
    'alpha': 1.0,    # BCE Loss weight
    'beta': 1.0,     # Dice Loss weight
    'gamma': 0.5,    # Perceptual Loss weight
    'delta': 0.3     # Edge Loss weight
}
```

### Aumentación de Datos
```python
transforms = [
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.3),
    ShiftScaleRotate(p=0.5),
    RandomBrightnessContrast(p=0.5),
    HueSaturationValue(p=0.3),
    GaussianBlur(p=0.2)
]
```

## 📈 Resultados Esperados

### Métricas Objetivo
- **IoU**: > 0.85 en validación
- **Dice Coefficient**: > 0.90 en validación
- **Pixel Accuracy**: > 0.95 en validación

### Tiempo de Entrenamiento
- **100 épocas**: ~4-6 horas en RTX 3080
- **Convergencia**: Típicamente en 50-70 épocas
- **Inferencia**: ~50ms por imagen (256x256)

## 🛠️ Solución de Problemas

### Error de Memoria GPU
```python
# Reducir batch size
config['batch_size'] = 4

# Usar gradient checkpointing
torch.utils.checkpoint.checkpoint()
```

### Dataset No Encontrado
```bash
# Verificar estructura de directorios
ls data/supervisely_persons/images/
ls data/supervisely_persons/annotations/
```

### Problemas de Convergencia
```python
# Ajustar learning rate
config['learning_rate'] = 5e-5

# Usar scheduler más agresivo
scheduler = CosineAnnealingLR(optimizer, T_max=50)
```

## 📚 Referencias y Recursos

### Papers Clave
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### Dataset
- [Supervisely Persons Dataset](https://github.com/supervisely-ecosystem/persons)
- [Dataset Ninja - Supervisely Persons](https://datasetninja.com/supervisely-persons)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Supervisely Team** por el dataset de alta calidad
- **PyTorch Team** por el framework de deep learning
- **Albumentations** por las herramientas de augmentación
- **OpenCV Community** por las utilidades de procesamiento de imágenes

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:

- **Email**: tu-email@ejemplo.com
- **GitHub**: [@tu-usuario](https://github.com/tu-usuario)
- **LinkedIn**: [Tu Perfil](https://linkedin.com/in/tu-perfil)

---

**¿Encontraste este proyecto útil? ¡Dale una ⭐ en GitHub!**