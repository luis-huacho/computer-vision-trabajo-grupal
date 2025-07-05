# Informe del Proceso de Entrenamiento de Segmentación

A continuación se detallan las características y los parámetros del proceso de entrenamiento para la segmentación de personas, basado en el análisis de los archivos `trainer.py` y `datasets.py`.

### Características Clave del Proceso de Entrenamiento

El entrenamiento está diseñado como un pipeline moderno y robusto de deep learning para segmentación de imágenes. Estas son sus características principales:

1.  **Modelo (U-Net con Attention):**
    *   Se utiliza un modelo `UNetAutoencoder`. La arquitectura U-Net es un estándar de oro para tareas de segmentación de imágenes porque es muy eficaz capturando tanto detalles finos como el contexto global de la imagen.
    *   Utiliza un **encoder pre-entrenado** (`pretrained=True`), lo que significa que se beneficia del conocimiento aprendido de un dataset masivo (como ImageNet) para extraer características de las imágenes de manera más efectiva desde el principio.
    *   Incorpora **mecanismos de atención** (`use_attention=True`), que permiten al modelo enfocarse en las regiones más relevantes de la imagen para mejorar la precisión de la segmentación.

2.  **Datos y Objetivo:**
    *   Utiliza el dataset **COCO (Common Objects in Context)**, filtrado para usar únicamente las imágenes que contienen personas.
    *   El objetivo del modelo no es solo predecir una máscara binaria. El `target` es una **imagen RGBA de 4 canales**:
        *   **Canales RGB:** La imagen original de la persona.
        *   **Canal Alpha (A):** La máscara de segmentación binaria.
    *   Esto significa que el modelo aprende a la vez a **reconstruir la apariencia de la persona y a delimitar su silueta**.

3.  **Aumentación de Datos:**
    *   Se aplica un conjunto de técnicas de aumentación de datos en tiempo real a las imágenes de entrenamiento para hacer el modelo más robusto y evitar el sobreajuste. Estas incluyen:
        *   Volteo Horizontal (`HorizontalFlip`)
        *   Rotaciones aleatorias de 90 grados (`RandomRotate90`)
        *   Cambios de escala, rotación y traslación (`ShiftScaleRotate`)
        *   Ajustes de brillo y contraste (`RandomBrightnessContrast`)
        *   Cambios en la tonalidad y saturación (`HueSaturationValue`)
        *   Desenfoque Gaussiano (`GaussianBlur`)

4.  **Función de Pérdida (Loss Function):**
    *   Se utiliza una función de pérdida personalizada (`LossCalculator` del archivo `utils.py`). Aunque no tenemos el código exacto, su objetivo es medir el error entre la imagen RGBA de 4 canales predicha por el modelo y la imagen RGBA real.
    *   Probablemente combina una pérdida de reconstrucción (como L1 o MSE) para los canales RGB y una pérdida de segmentación (como BCE o Dice Loss) para el canal Alpha.

5.  **Optimizador y Planificador de Tasa de Aprendizaje:**
    *   **Optimizador:** Se usa **Adam**, un optimizador muy popular y efectivo que adapta la tasa de aprendizaje para cada parámetro del modelo.
    *   **Planificador (Scheduler):** Se emplea `CosineAnnealingWarmRestarts`. Este es un planificador avanzado que reduce cíclicamente la tasa de aprendizaje siguiendo una curva de coseno y la "resetea" periódicamente. Ayuda al modelo a explorar mejor el espacio de soluciones y a no quedarse atascado en mínimos locales.

6.  **Métricas y Evaluación:**
    *   El rendimiento del modelo se mide con dos métricas estándar en segmentación:
        *   **IoU (Intersection over Union):** Mide la superposición entre la máscara predicha y la real.
        *   **Dice Coefficient:** Muy similar al IoU, también mide la superposición.
    *   Estas métricas se calculan en cada época tanto para los datos de entrenamiento como los de validación.

7.  **Guardado de Modelos (Checkpoints):**
    *   El script guarda automáticamente el modelo después de cada época.
    *   Lleva un registro del **mejor modelo hasta el momento** basándose en la métrica **IoU de validación**. Si el IoU de la época actual supera al mejor IoU histórico, se guarda una copia especial del modelo como `best_model.pth`.

### Parámetros de Entrenamiento

Aquí están los parámetros de entrenamiento con sus valores por defecto, tal como están definidos en `trainer.py`:

| Parámetro | Valor por Defecto | Descripción |
| :--- | :--- | :--- |
| `batch_size` | `16` | Número de imágenes procesadas en cada paso del entrenamiento. |
| `learning_rate` | `1e-4` (0.0001) | Tasa de aprendizaje inicial para el optimizador Adam. |
| `weight_decay` | `1e-6` (0.000001) | Técnica de regularización para prevenir el sobreajuste. |
| `num_epochs` | `100` | Número total de veces que el modelo verá el dataset completo. |
| `image_size` | `384` | Las imágenes se redimensionan a 384x384 píxeles antes de entrar al modelo. |
| `num_workers` | `8` | Número de subprocesos para cargar los datos en paralelo. Acelera la carga. |
| `pin_memory` | `True` | Optimización que acelera la transferencia de datos de la CPU a la GPU. |
| `device` | `cuda:{device_id}` | Se establece automáticamente para usar la GPU disponible. |

### Actualización de Parámetros (05/07/2025)

Se ha optado por un enfoque conservador para optimizar el uso de recursos en el servidor compartido, priorizando la estabilidad y un uso más eficiente de la GPU sin arriesgarse a errores de memoria.

Los siguientes parámetros han sido modificados en el archivo `trainer.py`:

*   **`batch_size`**: Se ha aumentado de `16` a **`32`**.
    *   **Justificación**: Un tamaño de lote mayor puede conducir a una estimación de gradiente más estable y a un uso más eficiente del hardware de la GPU, potencialmente acelerando la convergencia del entrenamiento. Este cambio se realiza asumiendo que hay suficiente VRAM disponible.

*   **`num_workers`**: Se ha aumentado de `8` a **`12`**.
    *   **Justificación**: Incrementar el número de workers puede acelerar el pipeline de carga de datos, asegurando que la GPU no se quede inactiva esperando los lotes de imágenes. Esto es especialmente útil si la CPU del servidor tiene suficientes núcleos para manejar la carga adicional.

*   **`image_size`**: Se mantiene en **`384x384`**.
    *   **Justificación**: Aunque aumentar la resolución podría mejorar la precisión en detalles finos, también incrementaría drásticamente (de forma cuadrática) el uso de memoria VRAM. Para evitar un error `CUDA out of memory`, se ha decidido no aumentar la resolución de la imagen al mismo tiempo que se aumenta el `batch_size`.

### Cómo Ejecutar el Entrenamiento de Segmentación

Para lanzar el entrenamiento de segmentación, se debe ejecutar el script `trainer.py`. A continuación se muestran los comandos recomendados según el entorno.

**Nota Importante:** Estos comandos inician **únicamente** el proceso de entrenamiento para el modelo de **segmentación**.

#### Ejecución en Múltiples GPUs

Para aprovechar todas las GPUs disponibles y acelerar el entrenamiento, se debe usar el lanzador `torchrun`.

```bash
torchrun trainer.py
```

#### Ejecución en una Sola GPU (Entorno Compartido)

Para limitar el entrenamiento a una única GPU específica y no saturar un servidor compartido, se recomienda usar la variable de entorno `CUDA_VISIBLE_DEVICES`. Este método es el más directo para controlar el uso de recursos.

```bash
# Ejemplo para usar solo la primera GPU (índice 0)
CUDA_VISIBLE_DEVICES=0 python trainer.py

# Ejemplo para usar solo la segunda GPU (índice 1)
CUDA_VISIBLE_DEVICES=1 python trainer.py
```

Alternativamente, se puede usar `torchrun` para lanzar un solo proceso:

```bash
torchrun --nproc_per_node=1 trainer.py
```
