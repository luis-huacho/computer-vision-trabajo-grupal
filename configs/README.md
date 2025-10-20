# Configuraciones de Entrenamiento - Archivos YAML

Sistema de configuración flexible para experimentos de deep learning en segmentación de personas.

**Proyecto**: U-Net Background Removal with Harmonization
**Autores**: Luis Huacho y Dominick Alvarez - PUCP

---

## 📁 Archivos de Configuración Disponibles

### 1. `default.yaml`
**Configuración estándar de producción**

- **Arquitectura**: ResNet-50
- **Dataset**: COCO completo (sin muestreo)
- **Epochs**: 100
- **Batch Size**: 16
- **Workers**: 8
- **Uso**: Entrenamiento completo y robusto

```bash
python main.py train  # Usa default.yaml automáticamente
```

---

### 2. `resnet50_full.yaml`
**ResNet-50 con todas las características**

- **Arquitectura**: ResNet-50
- **Dataset**: COCO completo
- **Epochs**: 100
- **Características**: Mixed precision, attention gates, scheduler cosine annealing
- **Uso**: Entrenamiento de producción optimizado

```bash
python main.py train --config resnet50_full
```

---

### 3. `resnet34_quick.yaml`
**Prueba rápida con ResNet-34 y subset**

- **Arquitectura**: ResNet-34 (más ligero)
- **Dataset**: **Subset de 1000 imágenes**
- **Epochs**: 20
- **Batch Size**: 16
- **Uso**: Iteración rápida, experimentación, validación de pipeline

```bash
python main.py train --config resnet34_quick
```

**Ideal para**:
- ✅ Probar cambios en el código
- ✅ Validar que el pipeline funciona
- ✅ Experimentar con hiperparámetros
- ✅ Debugging

---

### 4. `resnet50_10percent.yaml`
**10% del dataset para validación intermedia**

- **Arquitectura**: ResNet-50
- **Dataset**: 10% del COCO (muestreo aleatorio)
- **Epochs**: 30
- **Batch Size**: 16
- **Uso**: Punto medio entre quick y full

```bash
python main.py train --config resnet50_10percent
```

**Ideal para**:
- ✅ Validar hiperparámetros antes de full training
- ✅ Comparar configuraciones sin esperar 100 epochs
- ✅ Obtener métricas representativas rápidamente

---

### 5. `debug.yaml`
**Configuración mínima para debugging**

- **Arquitectura**: ResNet-34
- **Dataset**: **Solo 100 imágenes**
- **Epochs**: 3
- **Batch Size**: 4
- **Workers**: 2
- **Mixed Precision**: Desactivado
- **Uso**: Verificar errores, debugging, desarrollo

```bash
python main.py train --config debug
```

**Ideal para**:
- ✅ Verificar que el código no tiene errores
- ✅ Probar nuevas features
- ✅ Debugging paso a paso
- ✅ CI/CD testing

---

## 🆕 Configs para AISegment Dataset

### 6. `aisegment_full.yaml`
**Dataset AISegment completo - ResNet-50**

- **Arquitectura**: ResNet-50
- **Dataset**: AISegment completo (34,425 imágenes)
- **Epochs**: 100
- **Batch Size**: 16
- **Descarga**: Automática con kagglehub
- **Uso**: Entrenamiento de producción con matting de alta calidad

```bash
python main.py train --config aisegment_full
```

**Características**:
- ✅ Descarga automática del dataset desde Kaggle
- ✅ Matting profesional con canal alpha suave
- ✅ 34,425 retratos de medio cuerpo de alta calidad
- ✅ Configuración optimizada para producción

**Prerequisito**: API key de Kaggle configurada (ver `docs/AISegment_Setup.md`)

---

### 7. `aisegment_10percent.yaml`
**10% del AISegment para validación intermedia**

- **Arquitectura**: ResNet-50
- **Dataset**: ~3,400 imágenes (10% de AISegment)
- **Epochs**: 50
- **Batch Size**: 16
- **Uso**: Validación de hiperparámetros

```bash
python main.py train --config aisegment_10percent
```

**Ideal para**:
- ✅ Probar configuraciones antes de full training
- ✅ Validar cambios en el modelo
- ✅ Experimentación rápida con dataset real

---

### 8. `aisegment_quick.yaml`
**Prueba rápida con 1000 imágenes - ResNet-34**

- **Arquitectura**: ResNet-34 (más ligero)
- **Dataset**: 1,000 imágenes de AISegment
- **Epochs**: 20
- **Batch Size**: 16
- **Uso**: Iteración rápida y experimentación

```bash
python main.py train --config aisegment_quick
```

**Ideal para**:
- ✅ Validar pipeline con AISegment
- ✅ Pruebas rápidas (~20 minutos)
- ✅ Experimentar con matting de retratos
- ✅ Comparar con COCO

---

## 🚀 Uso Rápido

### Opción 1: Config por defecto
```bash
python main.py train
```
Carga automáticamente `configs/default.yaml`

### Opción 2: Config por nombre
```bash
python main.py train --config resnet50_full
python main.py train --config resnet34_quick
python main.py train --config debug
```
Busca el archivo en `configs/{nombre}.yaml`

### Opción 3: Path completo
```bash
python main.py train --config-path /ruta/completa/mi_config.yaml
python main.py train --config-path configs/experimento_custom.yaml
```

### Listar configuraciones disponibles
```bash
python main.py help
```

---

## 📝 Estructura de un Archivo YAML

### Secciones Principales

```yaml
# ============================================================================
# 1. INFORMACIÓN DEL EXPERIMENTO
# ============================================================================
experiment:
  name: "Nombre del Experimento"
  description: "Descripción detallada"
  mode: "production"  # production, debug, quick_test

# ============================================================================
# 2. CONFIGURACIÓN DEL MODELO
# ============================================================================
model:
  architecture: "resnet50"  # resnet50 o resnet34
  image_size: 384
  use_pretrained: true      # Usar pesos pre-entrenados ImageNet
  use_attention: true       # Attention gates en decoder

# ============================================================================
# 3. CONFIGURACIÓN DEL DATASET
# ============================================================================
dataset:
  type: "coco"              # coco o supervisely
  root: "COCO"              # Directorio del dataset

  # Filtros de calidad
  min_person_area: 500      # Área mínima en píxeles
  min_keypoints: 3          # Mínimo de keypoints visibles
  train_val_split: 0.8      # 80% train, 20% val

  # MUESTREO/SUBSET (para experimentos rápidos)
  sampling:
    enabled: false          # Activar o desactivar
    mode: "full"            # full, subset, percentage
    subset_size: null       # Número de imágenes (si mode=subset)
    percentage: null        # Porcentaje 0-1 (si mode=percentage)
    strategy: "random"      # random, first, balanced
    random_seed: 42         # Para reproducibilidad

# ============================================================================
# 4. CONFIGURACIÓN DE ENTRENAMIENTO
# ============================================================================
training:
  # Hiperparámetros básicos
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.000001

  # DataLoader
  num_workers: 8
  pin_memory: true
  drop_last: true

  # Optimización
  gradient_clip_max_norm: 0.5
  mixed_precision: true

  # Learning Rate Scheduler
  scheduler:
    type: "cosine_annealing"  # cosine_annealing, step, plateau
    T_0: 10                    # Para cosine_annealing
    T_mult: 2
    # O para step scheduler:
    # step_size: 30
    # gamma: 0.1

  # Pesos de las funciones de pérdida
  loss_weights:
    alpha: 1.0   # BCE (Binary Cross Entropy)
    beta: 1.0    # Dice Loss
    gamma: 0.5   # Perceptual Loss (VGG)
    delta: 0.3   # Edge Loss

# ============================================================================
# 5. CONFIGURACIÓN DE VALIDACIÓN
# ============================================================================
validation:
  frequency: 1              # Validar cada N épocas

  early_stopping:
    enabled: true
    patience: 15            # Épocas sin mejora antes de parar
    min_delta: 0.0001       # Mejora mínima considerada

  # Métricas objetivo
  target_metrics:
    iou: 0.85
    dice: 0.90
    pixel_accuracy: 0.95

# ============================================================================
# 6. CONFIGURACIÓN DE CHECKPOINTS Y LOGGING
# ============================================================================
checkpoints:
  save_best: true
  save_last: true
  save_every_n_epochs: 5
  checkpoint_dir: "checkpoints/resnet50"

logging:
  log_every_n_batches: 10
  save_plots: true
  plot_dpi: 300

# ============================================================================
# 7. AUMENTACIÓN DE DATOS
# ============================================================================
augmentation:
  train:
    horizontal_flip: 0.5
    random_rotate90: 0.3
    shift_scale_rotate:
      enabled: true
      shift_limit: 0.1
      scale_limit: 0.2
      rotate_limit: 15
      p: 0.5
    brightness_contrast:
      enabled: true
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
```

---

## 🎯 Casos de Uso Comunes

### Caso 1: Experimentar con Hiperparámetros

**Problema**: Quiero probar diferentes learning rates sin esperar 100 epochs

**Solución**: Crear config con subset y menos epochs

```yaml
# configs/lr_experiment.yaml
experiment:
  name: "LR Experiment - 5e-5"

model:
  architecture: "resnet34"  # Más rápido

dataset:
  sampling:
    enabled: true
    mode: "percentage"
    percentage: 0.1  # Solo 10% del dataset

training:
  num_epochs: 20
  learning_rate: 0.00005  # Probar nuevo LR
```

```bash
python main.py train --config lr_experiment
```

---

### Caso 2: Comparar Arquitecturas

**Problema**: ¿ResNet-50 o ResNet-34 es mejor para mi caso?

**Solución**: Crear dos configs y comparar

```yaml
# configs/compare_resnet50.yaml
experiment:
  name: "Compare - ResNet50"
model:
  architecture: "resnet50"
dataset:
  sampling:
    enabled: true
    mode: "subset"
    subset_size: 5000
training:
  num_epochs: 30
checkpoints:
  checkpoint_dir: "checkpoints/compare_resnet50"
```

```yaml
# configs/compare_resnet34.yaml
experiment:
  name: "Compare - ResNet34"
model:
  architecture: "resnet34"
dataset:
  sampling:
    enabled: true
    mode: "subset"
    subset_size: 5000
training:
  num_epochs: 30
checkpoints:
  checkpoint_dir: "checkpoints/compare_resnet34"
```

```bash
python main.py train --config compare_resnet50
python main.py train --config compare_resnet34
# Comparar métricas en logs/
```

---

### Caso 3: Entrenamiento Completo de Producción

**Problema**: Configuración final para paper/producción

**Solución**: Usar config de producción con dataset completo

```bash
# Opción 1: Usar resnet50_full.yaml
python main.py train --config resnet50_full

# Opción 2: Crear config personalizado
cp configs/resnet50_full.yaml configs/production_final.yaml
# Editar production_final.yaml según necesidades
python main.py train --config production_final
```

---

### Caso 4: Debugging de Errores

**Problema**: Mi código falla y no sé dónde

**Solución**: Usar debug.yaml con datos mínimos

```bash
python main.py train --config debug
# Si falla, el error aparece rápido (3 epochs, 100 imágenes)
# Si funciona, escalar a resnet34_quick.yaml
```

---

## 📊 Muestreo/Subset de Datos

### ¿Por qué usar muestreo?

- ⚡ **Iteración rápida**: Probar cambios sin esperar horas
- 💰 **Ahorro de recursos**: GPU/CPU/tiempo
- 🔬 **Experimentación**: Probar múltiples configuraciones
- 🐛 **Debugging**: Encontrar errores rápidamente

### Modos de Muestreo

#### 1. `mode: "full"` - Dataset Completo
```yaml
sampling:
  enabled: false  # O enabled: true con mode: "full"
  mode: "full"
```
Usa todas las imágenes disponibles.

#### 2. `mode: "subset"` - Cantidad Fija
```yaml
sampling:
  enabled: true
  mode: "subset"
  subset_size: 1000  # Exactamente 1000 imágenes
  strategy: "random"  # Aleatorio
  random_seed: 42
```
Especifica número exacto de imágenes.

#### 3. `mode: "percentage"` - Porcentaje
```yaml
sampling:
  enabled: true
  mode: "percentage"
  percentage: 0.1  # 10% del dataset
  strategy: "random"
```
Útil cuando no sabes el tamaño total del dataset.

### Estrategias de Muestreo

- **`"random"`**: Selección aleatoria (recomendado)
- **`"first"`**: Primeras N imágenes (más rápido, menos representativo)
- **`"balanced"`**: Balanceo por clases (futuro)

---

## 🛠️ Crear Tu Propia Configuración

### Paso 1: Copiar Template
```bash
cp configs/default.yaml configs/mi_experimento.yaml
```

### Paso 2: Editar Según Necesidades
```yaml
experiment:
  name: "Mi Experimento - Loss Weights"
  description: "Probando diferentes pesos de pérdidas"

model:
  architecture: "resnet50"

dataset:
  sampling:
    enabled: true
    mode: "subset"
    subset_size: 2000  # Dataset reducido para pruebas

training:
  num_epochs: 40
  batch_size: 16

  # Modificar pesos de pérdidas
  loss_weights:
    alpha: 1.0
    beta: 2.0    # Aumentar peso de Dice
    gamma: 0.3   # Reducir perceptual
    delta: 0.5   # Aumentar edge

checkpoints:
  checkpoint_dir: "checkpoints/loss_experiment"
```

### Paso 3: Ejecutar
```bash
python main.py train --config mi_experimento
```

### Paso 4: Versionar
```bash
git add configs/mi_experimento.yaml
git commit -m "Add mi_experimento config"
```

---

## 📖 Referencia Rápida

### Parámetros Comunes

| Parámetro | Valores | Descripción |
|-----------|---------|-------------|
| `architecture` | `resnet50`, `resnet34` | Backbone del U-Net |
| `image_size` | `256`, `384`, `512` | Tamaño de entrada |
| `batch_size` | `4`, `8`, `16`, `32` | Depende de VRAM |
| `learning_rate` | `1e-4`, `5e-5`, `1e-3` | Ajustar según necesidad |
| `num_epochs` | `20`, `50`, `100` | Balance tiempo/calidad |
| `num_workers` | `0`, `4`, `8`, `16` | Depende de CPU |
| `mixed_precision` | `true`, `false` | Ahorra VRAM, más rápido |

### Scheduler Types

| Type | Uso | Parámetros |
|------|-----|------------|
| `cosine_annealing` | General (recomendado) | `T_0`, `T_mult` |
| `step` | Decaimiento por steps | `step_size`, `gamma` |
| `plateau` | Adaptativo por métrica | `patience`, `factor` |

### Sampling Modes

| Mode | Cuándo Usar | Parámetro |
|------|-------------|-----------|
| `full` | Producción, training final | - |
| `subset` | Conoces dataset, quieres N imágenes | `subset_size` |
| `percentage` | Dataset variable, quieres X% | `percentage` |

---

## ⚠️ Advertencias y Tips

### ✅ Buenas Prácticas

1. **Nombra descriptivamente**: `resnet50_lr5e5_bs32.yaml` mejor que `config1.yaml`
2. **Versionina configs**: Agregar a git para reproducibilidad
3. **Documenta cambios**: Usa `description` en el YAML
4. **Empieza pequeño**: Usa `debug` o `quick` antes de `full`
5. **Guarda métricas**: Checkpoints separados por experimento

### ❌ Errores Comunes

1. **Batch size muy grande**: OOM error
   - Solución: Reducir `batch_size` o activar `mixed_precision`

2. **Num workers muy alto**: Congela o ralentiza
   - Solución: `num_workers` = número de CPU cores / 2

3. **Dataset no existe**: FileNotFoundError
   - Solución: Verificar `dataset.root` apunta al directorio correcto

4. **Config no encontrado**: FileNotFoundError
   - Solución: Verificar que archivo existe en `configs/`
   - Usar `python main.py help` para listar disponibles

---

## 🔗 Referencias

- **Documentación completa**: Ver `CLAUDE.md` en raíz del proyecto
- **Código**: `config_loader.py`, `trainer.py`, `main.py`
- **Datasets**: `Readme.COCO.md` para setup de COCO

---

## 📞 Soporte

**Proyecto**: U-Net Background Removal with Harmonization
**Autores**: Luis Huacho y Dominick Alvarez
**Institución**: Maestría en Informática, PUCP

Para más información, consultar la documentación completa en `CLAUDE.md` y `Readme.md`.
