# Conceptos Técnicos y Métricas

Este documento explica los conceptos fundamentales, arquitecturas y métricas utilizadas en el proyecto de remoción de fondo con U-Net Autoencoder.

## 📚 Conceptos Fundamentales

### U-Net Architecture

**Definición**: U-Net es una arquitectura de red neuronal convolucional diseñada originalmente para segmentación biomédica, caracterizada por su forma de "U" que combina un path de contracción (encoder) con un path de expansión (decoder).

**Componentes Clave**:
- **Contracting Path (Encoder)**: Captura el contexto mediante convoluciones y pooling
- **Expansive Path (Decoder)**: Permite localización precisa mediante upsampling
- **Skip Connections**: Conectan directamente capas del encoder con el decoder

**Ventajas**:
- Preserva detalles finos gracias a las skip connections
- Eficiente con pocos datos de entrenamiento
- Excelente para tareas de segmentación pixel-wise

### Autoencoder

**Definición**: Arquitectura neuronal que aprende representaciones comprimidas de los datos (encoding) y luego los reconstruye (decoding).

**Estructura**:
```
Input → Encoder → Latent Space → Decoder → Output
```

**En nuestro contexto**:
- **Encoder**: Extrae características de la imagen original
- **Latent Space**: Representación comprimida con información esencial
- **Decoder**: Reconstruye la imagen enfocándose solo en personas

### Attention Mechanisms

**Attention Gates**: Mecanismo que permite al modelo "prestar atención" a regiones específicas de la imagen.

**Funcionamiento**:
1. Recibe feature maps del encoder y decoder
2. Calcula coeficientes de atención
3. Multiplica los feature maps por estos coeficientes
4. Enfatiza regiones relevantes (personas) y suprime fondo

**Fórmula matemática**:
```
α = σ(W_g * g + W_x * x + b)
output = x * α
```

Donde:
- `g`: gating signal (del decoder)
- `x`: feature map (del encoder)
- `σ`: función sigmoid
- `α`: coeficientes de atención

### Transfer Learning

**Concepto**: Utilizar un modelo pre-entrenado en una tarea relacionada como punto de partida.

**En nuestro modelo**:
- **Backbone**: ResNet50 pre-entrenado en ImageNet
- **Ventajas**: Convergencia más rápida, mejor extracción de características
- **Adaptación**: Las últimas capas se ajustan para nuestra tarea específica

## 🎯 Funciones de Pérdida

### 1. Binary Cross-Entropy (BCE) Loss

**Propósito**: Clasificación binaria pixel-wise (persona vs fondo)

**Fórmula**:
```
BCE = -[y*log(p) + (1-y)*log(1-p)]
```

**Interpretación**:
- `y`: etiqueta real (0 o 1)
- `p`: probabilidad predicha
- **Valor óptimo**: 0.0
- **Rango típico**: 0.0 - 2.0

### 2. Dice Loss

**Propósito**: Maneja desbalance de clases en segmentación

**Fórmula**:
```
Dice = (2 * |A ∩ B|) / (|A| + |B|)
Dice Loss = 1 - Dice
```

**Interpretación**:
- `A`: predicción
- `B`: ground truth
- **Valor óptimo**: 0.0 (Dice Coefficient = 1.0)
- **Rango**: 0.0 - 1.0

**Ventajas**:
- Robusto ante desbalance de clases
- Se enfoca en la superposición de regiones

### 3. Perceptual Loss

**Propósito**: Preservar características visuales naturales

**Funcionamiento**:
- Utiliza features de una red pre-entrenada (VGG)
- Compara representaciones de alto nivel
- Mantiene estructura y textura visual

**Aplicación**:
```python
perceptual_loss = MSE(VGG_features(pred), VGG_features(target))
```

### 4. Edge Loss

**Propósito**: Preservar contornos nítidos en la segmentación

**Implementación**:
- Aplica filtros Sobel para detectar bordes
- Compara gradientes entre predicción y ground truth
- Penaliza bordes difusos o incorrectos

**Filtros Sobel**:
```
Sobel_x = [[-1, 0, 1],    Sobel_y = [[-1, -2, -1],
           [-2, 0, 2],               [ 0,  0,  0],
           [-1, 0, 1]]               [ 1,  2,  1]]
```

### Función de Pérdida Compuesta

**Combinación**:
```
Total_Loss = α*BCE + β*Dice + γ*Perceptual + δ*Edge
```

**Pesos típicos**:
- α = 1.0 (BCE)
- β = 1.0 (Dice)
- γ = 0.5 (Perceptual)
- δ = 0.3 (Edge)

## 📊 Métricas de Evaluación

### 1. Intersection over Union (IoU)

**Definición**: Mide la superposición entre predicción y ground truth

**Fórmula**:
```
IoU = |A ∩ B| / |A ∪ B|
```

**Interpretación**:
- **Rango**: 0.0 - 1.0
- **0.0**: Sin superposición
- **0.5**: Aceptable
- **0.7**: Bueno
- **0.85+**: Excelente

**Umbral de decisión**: Típicamente 0.5 para binarización

### 2. Dice Coefficient

**Definición**: Medida de similitud entre dos conjuntos

**Fórmula**:
```
Dice = (2 * |A ∩ B|) / (|A| + |B|)
```

**Interpretación**:
- **Rango**: 0.0 - 1.0
- **0.0**: Sin similitud
- **0.8**: Buena segmentación
- **0.9+**: Excelente segmentación

**Relación con IoU**:
```
Dice = 2*IoU / (1 + IoU)
```

### 3. Pixel Accuracy

**Definición**: Proporción de píxeles clasificados correctamente

**Fórmula**:
```
Pixel_Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretación**:
- **Rango**: 0.0 - 1.0
- **0.95+**: Muy buena precisión
- **Limitación**: Puede ser engañosa con clases desbalanceadas

### 4. Métricas Adicionales

#### Sensitivity (Recall)
```
Sensitivity = TP / (TP + FN)
```
Mide la capacidad de detectar píxeles de persona.

#### Specificity
```
Specificity = TN / (TN + FP)
```
Mide la capacidad de identificar correctamente el fondo.

#### Precision
```
Precision = TP / (TP + FP)
```
Proporción de píxeles predichos como persona que realmente lo son.

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Media armónica entre precision y recall.

## 🔧 Parámetros de Entrenamiento

### Learning Rate Scheduling

**Cosine Annealing with Warm Restarts**:
```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Comportamiento**:
- Reduce learning rate siguiendo una función coseno
- "Reinicia" periódicamente para escapar mínimos locales
- T_0: período inicial, T_mult: multiplicador de período

### Optimización

**Adam Optimizer**:
- **Learning Rate**: 1e-4 (típico para fine-tuning)
- **Weight Decay**: 1e-5 (regularización L2)
- **Betas**: (0.9, 0.999) (momentos por defecto)

### Regularización

**Dropout**: 
- Encoder: 0.1
- Bottleneck: 0.2
- Previene overfitting

**Batch Normalization**:
- Estabiliza entrenamiento
- Permite learning rates más altos
- Actúa como regularizador

## 📈 Interpretación de Resultados

### Curvas de Entrenamiento

#### Loss Curves
- **Descenso suave**: Entrenamiento estable
- **Oscilaciones**: Learning rate muy alto
- **Plateau**: Posible convergencia o learning rate muy bajo
- **Divergencia**: Gradientes inestables

#### Validation vs Training
- **Gap pequeño**: Buen balance, sin overfitting
- **Gap grande**: Overfitting, necesita más regularización
- **Val > Train**: Posible problema en datos o implementación

### Métricas por Época

#### IoU Progression
- **Inicio**: ~0.3 - 0.4
- **Convergencia**: 0.85+
- **Estancamiento en 0.7**: Posible problema de arquitectura

#### Dice Coefficient
- **Correlacionado con IoU**: Debe seguir tendencia similar
- **Más sensible**: Cambios más pronunciados que IoU
- **Objetivo**: 0.9+ para aplicaciones comerciales

## 🎨 Aumentación de Datos

### Transformaciones Geométricas
- **HorizontalFlip**: Duplica variabilidad, preserva personas
- **RandomRotate90**: Diferentes orientaciones
- **ShiftScaleRotate**: Variaciones realistas de pose

### Transformaciones Fotométricas
- **RandomBrightnessContrast**: Simula diferentes condiciones de luz
- **HueSaturationValue**: Variabilidad de color
- **GaussianBlur**: Robustez ante desenfoques

### Consideraciones Especiales
- **Preservar anotaciones**: Transformaciones deben aplicarse a imagen y máscara
- **Probabilidades balanceadas**: No todas las transformaciones en cada muestra
- **Coherencia temporal**: En videos, mantener consistencia

## ⚡ Optimizaciones de Rendimiento

### Mixed Precision Training

**Concepto**: Usar FP16 en lugar de FP32 para acelerar entrenamiento

**Implementación**:
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Beneficios**:
- 1.5-2x más rápido
- Usa menos memoria GPU
- Mantiene precisión numérica

### Gradient Checkpointing

**Propósito**: Reducir uso de memoria a costa de tiempo de cómputo

**Aplicación**:
```python
import torch.utils.checkpoint as checkpoint

def forward(self, x):
    x = checkpoint.checkpoint(self.encoder, x)
    return self.decoder(x)
```

**Trade-off**:
- ↓ Memoria: ~50% menos
- ↑ Tiempo: ~20% más lento

### Data Loading Optimization

**Estrategias**:
- **num_workers**: 4-8 procesos paralelos
- **pin_memory**: True para GPU
- **prefetch_factor**: 2-4 para cache

**Implementación**:
```python
DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

## 🧪 Experimentación y Debugging

### Sanity Checks

#### 1. Overfitting Test
```python
# Entrenar en 1 batch pequeño
small_dataset = Subset(dataset, range(8))
# Debe alcanzar loss ~0 rápidamente
```

#### 2. Learning Rate Range Test
```python
# Incrementar LR exponencialmente
for epoch in range(100):
    lr = 1e-6 * (10 ** (epoch / 100))
    # Plotear loss vs LR
```

#### 3. Gradient Flow
```python
# Verificar gradientes no son 0 o infinito
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Common Issues y Soluciones

#### Loss no disminuye
- **Verificar**: Learning rate, arquitectura, datos
- **Solución**: Reducir LR, simplificar modelo, verificar labels

#### Overfitting rápido
- **Síntomas**: Val loss aumenta mientras train loss baja
- **Solución**: Más dropout, regularización, más datos

#### Gradientes explosivos
- **Síntomas**: Loss se vuelve NaN
- **Solución**: Gradient clipping, LR más bajo

#### Convergencia lenta
- **Causas**: LR muy bajo, modelo muy profundo
- **Solución**: LR scheduling, skip connections

## 📋 Evaluación Avanzada

### Cross-Validation

**K-Fold Validation**:
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(dataset):
    # Entrenar y evaluar cada fold
```

**Estratified Split**: Para datasets desbalanceados
```python
from sklearn.model_selection import StratifiedKFold
# Mantiene proporción de clases en cada fold
```

### Benchmarking

#### Comparación con Baselines
- **Traditional Methods**: GrabCut, Watershed
- **Deep Learning**: DeepLab, Mask R-CNN
- **Specialized**: Background Matting, MODNet

#### Métricas de Tiempo
```python
import time

start_time = time.time()
output = model(input_batch)
inference_time = time.time() - start_time

fps = batch_size / inference_time
```

### Test-Time Augmentation (TTA)

**Concepto**: Promediar predicciones de múltiples transformaciones

**Implementación**:
```python
def tta_predict(model, image, transforms):
    predictions = []
    for transform in transforms:
        aug_image = transform(image)
        pred = model(aug_image)
        # Invertir transformación
        pred = inverse_transform(pred, transform)
        predictions.append(pred)
    
    return torch.mean(torch.stack(predictions), dim=0)
```

## 🎯 Métricas Específicas para Remoción de Fondo

### Trimap Accuracy

**Definición**: Precisión en regiones de transición (bordes)

**Cálculo**:
1. Generar trimap (foreground, background, unknown)
2. Evaluar precisión solo en región "unknown"
3. Más desafiante que métricas estándar

### Alpha Matte Quality

**Sum of Absolute Differences (SAD)**:
```
SAD = Σ|α_pred - α_true|
```

**Mean Squared Error (MSE)**:
```
MSE = (1/N) * Σ(α_pred - α_true)²
```

**Gradient Error**:
```
Grad_Error = Σ|∇α_pred - ∇α_true|
```

### Perceptual Metrics

#### Structural Similarity (SSIM)
```python
from skimage.metrics import structural_similarity

ssim_value = structural_similarity(pred_rgb, target_rgb, multichannel=True)
```

**Interpretación**:
- **Rango**: -1 a 1
- **0.9+**: Excelente similitud visual
- **0.7-0.9**: Buena calidad
- **<0.7**: Calidad pobre

#### Learned Perceptual Image Patch Similarity (LPIPS)
```python
import lpips

loss_fn = lpips.LPIPS(net='alex')
perceptual_distance = loss_fn(pred_tensor, target_tensor)
```

**Ventajas**:
- Correlaciona mejor con percepción humana
- Detecta diferencias semánticamente importantes

## 🔍 Análisis de Casos Edge

### Escenarios Desafiantes

#### 1. Múltiples Personas
- **Problema**: Separación entre personas
- **Solución**: Instance segmentation, mejores skip connections

#### 2. Oclusión Parcial
- **Problema**: Personas parcialmente ocultas
- **Solución**: Más datos de entrenamiento, context understanding

#### 3. Backgrounds Complejos
- **Problema**: Confusión con texturas similares a piel/ropa
- **Solución**: Más diversidad en datos, mejores features

#### 4. Iluminación Extrema
- **Problema**: Sombras duras, contraluz
- **Solución**: Augmentación fotométrica, normalización adaptiva

### Estrategias de Mejora

#### 1. Active Learning
- Identificar casos problemáticos automáticamente
- Priorizar anotación manual de casos difíciles

#### 2. Hard Negative Mining
- Enfocarse en ejemplos donde el modelo falla
- Rebalancear dataset hacia casos desafiantes

#### 3. Multi-Scale Training
- Entrenar con diferentes resoluciones
- Mejor generalización a diferentes tamaños

## 📊 Monitoreo en Producción

### Métricas de Sistema

#### Latencia
- **Target**: <100ms para aplicaciones en tiempo real
- **Medición**: Percentil 95, no solo promedio

#### Throughput
- **Métrica**: Imágenes por segundo (IPS)
- **Optimización**: Batch processing, model optimization

#### Uso de Recursos
- **GPU Memory**: Monitorear picos y leaks
- **CPU Usage**: Para pre/post-procesamiento
- **Disk I/O**: Para carga de modelos grandes

### Drift Detection

#### Data Drift
- Distribución de inputs cambia con el tiempo
- **Detección**: Statistical tests, embedding analysis

#### Model Drift
- Performance se degrada gradualmente
- **Detección**: A/B testing, performance monitoring

### Calidad Automática

#### No-Reference Metrics
- Métricas sin ground truth
- **Ejemplos**: Blur detection, edge consistency

#### Anomaly Detection
- Detección de outputs anómalos
- **Métodos**: Reconstruction error, density estimation

## 🚀 Optimización para Deployment

### Model Quantization

**Post-Training Quantization**:
```python
import torch.quantization

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
```

**Beneficios**:
- 4x menos memoria
- 2-4x más rápido en CPU
- Ligera pérdida de precisión

### Model Pruning

**Structured Pruning**:
```python
import torch.nn.utils.prune as prune

prune.ln_structured(model.conv1, name="weight", amount=0.3, n=2, dim=0)
```

**Unstructured Pruning**:
```python
prune.random_unstructured(model.conv1, name="weight", amount=0.3)
```

### Knowledge Distillation

**Concepto**: Entrenar modelo pequeño (student) con modelo grande (teacher)

```python
def distillation_loss(student_logits, teacher_logits, temperature=3):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(soft_student, soft_targets, reduction='batchmean')
```

## 📚 Recursos Adicionales

### Datasets Complementarios
- **COCO Person**: Más diversidad
- **ADE20K**: Escenas complejas
- **Open Images**: Escala masiva
- **PASCAL VOC**: Benchmark estándar

### Herramientas de Visualización
- **TensorBoard**: Métricas y gráficas
- **Weights & Biases**: Experiment tracking
- **Visdom**: Visualización en tiempo real

### Frameworks de Optimización
- **TensorRT**: NVIDIA GPU optimization
- **ONNX**: Cross-platform deployment
- **OpenVINO**: Intel hardware optimization

---

**Nota**: Este documento es dinámico y debe actualizarse conforme evolucione el proyecto y se descubran nuevas técnicas o mejores prácticas.