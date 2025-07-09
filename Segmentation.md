# Informe del Proceso de Entrenamiento de Segmentación

A continuación se detallan las características y los parámetros del proceso de entrenamiento para la segmentación de personas, basado en el análisis de los archivos `trainer.py` y `datasets.py`.

### Flujo de Datos del Modelo (Input y Output)

Es fundamental entender el flujo de datos del modelo. Este funciona como un "traductor" que transforma una imagen de entrada estándar en una imagen de salida con información de segmentación y reconstrucción.

#### Input (Entrada)

El **input** del modelo es una **imagen a color estándar (RGB)** que ha sido procesada para el entrenamiento. El proceso para cada imagen es el siguiente:

1.  **Fuente:** Se toma una imagen del dataset COCO que contiene al menos una persona.
2.  **Redimensionamiento:** La imagen se redimensiona a un tamaño fijo (por defecto **384x384 píxeles**), utilizando padding para mantener la proporción original y evitar deformaciones.
3.  **Aumentación de Datos:** Se aplican transformaciones aleatorias (volteos, rotaciones, etc.) en el conjunto de entrenamiento para mejorar la robustez del modelo.
4.  **Normalización:** Los valores de los píxeles (0-255) se normalizan a un rango de **0.0 a 1.0**.
5.  **Formato (Tensor):** Finalmente, la imagen se convierte en un Tensor de PyTorch con el formato `[Canales, Altura, Anchura]`.

En resumen, el input es un **tensor de dimensiones `(3, 384, 384)`** que representa una imagen RGB normalizada.

#### Output (Salida)

El **output** del modelo es una **imagen sintética de 4 canales (RGBA)** del mismo tamaño que la entrada, donde cada canal tiene un propósito específico:

*   **Canales 1, 2 y 3 (RGB):** La **reconstrucción de la persona**. El modelo intenta generar una imagen de la persona aislada, sin el fondo original.
*   **Canal 4 (Alpha):** La **máscara de segmentación**. Este canal es una imagen en escala de grises donde los valores de los píxeles van de 0.0 a 1.0:
    *   Un valor cercano a **1.0 (blanco)** indica que el píxel **pertenece a la persona**.
    *   Un valor cercano a **0.0 (negro)** indica que el píxel **es parte del fondo**.

En resumen, el output es un **tensor de dimensiones `(4, 384, 384)`** que representa una imagen RGBA, conteniendo tanto la apariencia de la persona como su silueta (máscara).

### Arquitectura del Modelo (U-Net y sus Componentes Clave)

El modelo de segmentación utiliza la arquitectura `UNetAutoencoder`, que se basa en la popular red U-Net. Esta arquitectura es fundamental para tareas de segmentación de imágenes debido a su capacidad para combinar información de contexto global con detalles de localización precisos.

#### Componentes Principales: Encoder y Decoder

1.  **Encoder (Camino de Contracción):**
    *   Actúa como un extractor de características, reduciendo progresivamente la resolución espacial de la imagen de entrada mientras aumenta la profundidad de las características (número de canales).
    *   Utiliza un **encoder pre-entrenado** (`pretrained=True`), lo que le permite beneficiarse del conocimiento aprendido de grandes datasets (como ImageNet) para extraer características más robustas desde el inicio.
    *   Su objetivo es comprimir la información visual en una representación más abstracta y semántica.

2.  **Decoder (Camino de Expansión):**
    *   Toma la representación comprimida del encoder y la expande progresivamente para reconstruir la máscara de segmentación a la resolución original de la entrada.
    *   Su tarea es traducir las características de alto nivel en una predicción de píxel a píxel.

#### Skip Connections (Conexiones de Salto)

Las skip connections son la característica definitoria y más importante de la arquitectura U-Net. Son "atajos" que conectan directamente las capas del encoder con las capas correspondientes del decoder.

*   **Función:** Permiten que la información de alta resolución y los detalles espaciales finos (que se pierden durante la contracción en el encoder) sean transferidos directamente al decoder.
*   **Beneficio:** Al combinar la información semántica de alto nivel del decoder con los detalles espaciales del encoder, el modelo puede generar máscaras de segmentación mucho más precisas y con bordes bien definidos. Sin estas conexiones, el decoder solo podría producir máscaras borrosas.

#### Espacio Latente (Embedding)

En el punto más profundo y compacto de la U-Net, justo entre el final del encoder y el inicio del decoder (conocido como la capa de cuello de botella o *bottleneck*), se forma un **embedding** o representación latente de la imagen de entrada.

*   **Naturaleza:** Esta es una representación comprimida y de baja dimensión de la imagen original.
*   **Contenido:** Ha perdido gran parte de la información espacial detallada, pero ha codificado la **información semántica de alto nivel** de la imagen (el "qué" hay en la imagen, por ejemplo, la presencia de una persona).
*   **Rol:** Este embedding sirve como la base abstracta a partir de la cual el decoder comienza su proceso de reconstrucción. Es el "qué" de la imagen, mientras que las skip connections proporcionan el "dónde", permitiendo una reconstrucción precisa y detallada.

### Características Clave del Proceso de Entrenamiento

El entrenamiento está diseñado como un pipeline moderno y robusto de deep learning para segmentación de imágenes. Estas son sus características principales:

1.  **Datos y Objetivo:**
    *   Utiliza el dataset **COCO (Common Objects in Context)**, filtrado para usar únicamente las imágenes que contienen personas.
    *   El objetivo del modelo no es solo predecir una máscara binaria. El `target` es una **imagen RGBA de 4 canales**:
        *   **Canales RGB:** La imagen original de la persona.
        *   **Canal Alpha (A):** La máscara de segmentación binaria.
    *   Esto significa que el modelo aprende a la vez a **reconstruir la apariencia de la persona y a delimitar su silueta**.

2.  **Aumentación de Datos:**
    *   Se aplica un conjunto de técnicas de aumentación de datos en tiempo real a las imágenes de entrenamiento para hacer el modelo más robusto y evitar el sobreajuste. Estas incluyen:
        *   Volteo Horizontal (`HorizontalFlip`)
        *   Rotaciones aleatorias de 90 grados (`RandomRotate90`)
        *   Cambios de escala, rotación y traslación (`ShiftScaleRotate`)
        *   Ajustes de brillo y contraste (`RandomBrightnessContrast`)
        *   Cambios en la tonalidad y saturación (`HueSaturationValue`)
        *   Desenfoque Gaussiano (`GaussianBlur`)

3.  **Función de Pérdida (Loss Function):**
    *   Se utiliza una función de pérdida personalizada (`LossCalculator` del archivo `utils.py`). Aunque no tenemos el código exacto, su objetivo es medir el error entre la imagen RGBA de 4 canales predicha por el modelo y la imagen RGBA real.
    *   Probablemente combina una pérdida de reconstrucción (como L1 o MSE) para los canales RGB y una pérdida de segmentación (como BCE o Dice Loss) para el canal Alpha.

4.  **Optimizador y Planificador de Tasa de Aprendizaje:**
    *   **Optimizador:** Se usa **Adam**, un optimizador muy popular y efectivo que adapta la tasa de aprendizaje para cada parámetro del modelo.
    *   **Planificador (Scheduler):** Se emplea `CosineAnnealingWarmRestarts`. Este es un planificador avanzado que reduce cíclicamente la tasa de aprendizaje siguiendo una curva de coseno y la "resetea" periódicamente. Ayuda al modelo a explorar mejor el espacio de soluciones y a no quedarse atascado en mínimos locales.

5.  **Métricas y Evaluación:**
    *   El rendimiento del modelo se mide con dos métricas estándar en segmentación:
        *   **IoU (Intersection over Union):** Mide la superposición entre la máscara predicha y la real.
        *   **Dice Coefficient:** Muy similar al IoU, también mide la superposición.
    *   Estas métricas se calculan en cada época tanto para los datos de entrenamiento como los de validación.

6.  **Guardado de Modelos (Checkpoints):**
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

*   Se hizo rollback, porque la GPU se saturaba.

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

### Uso en Producción: Reconstrucción a Dimensiones Originales

Un requisito clave para utilizar el modelo en un pipeline de composición de imágenes es poder aplicar la máscara de segmentación a la imagen en su tamaño y proporción originales. El pipeline de datos está diseñado para permitir esto mediante el uso de metadatos.

#### El Mecanismo: Metadatos de Restauración

La clase `COCOPersonDataset` en `datasets.py` incluye un parámetro `store_metadata`. Cuando se activa (`store_metadata=True`), el `DataLoader` no solo devuelve la imagen de entrada y el objetivo, sino también un **diccionario de metadatos** por cada imagen. Este diccionario contiene la información necesaria para revertir las transformaciones de pre-procesamiento (redimensionamiento y padding).

El diccionario de metadatos (`restore_metadata`) contiene:

*   `original_size`: Las dimensiones (ancho, alto) de la imagen original.
*   `scale`: El factor de escala que se aplicó.
*   `padding`: El número de píxeles de relleno que se añadieron para convertir la imagen a las dimensiones de entrada del modelo (ej. 384x384).

#### Proceso de Reconstrucción (Inferencia)

Para obtener una imagen final segmentada con sus dimensiones originales, se deben seguir los siguientes pasos en un script de inferencia:

1.  **Pre-procesar la Imagen:** Cargar la imagen original y aplicarle las mismas transformaciones que en el entrenamiento (redimensionar con padding, normalizar). Es crucial **guardar los metadatos** de esta transformación.

2.  **Realizar la Inferencia:** Pasar la imagen procesada a través del modelo de segmentación para obtener el tensor de salida de 4 canales (RGBA) en baja resolución (ej. 384x384).

3.  **Extraer la Máscara:** Del tensor de salida, aislar el 4º canal (Alpha). Esta es la máscara de segmentación en baja resolución.

4.  **Revertir el Padding:** Utilizando los metadatos de `padding`, recortar las bandas de relleno de la máscara. La máscara ahora tendrá las dimensiones de la imagen escalada antes del relleno.

5.  **Revertir el Redimensionamiento:** Usando los metadatos de `original_size`, redimensionar la máscara recortada a las dimensiones exactas de la imagen original. El resultado es una **máscara de segmentación en alta resolución**.

6.  **Composición Final:**
    *   Tomar la imagen original de alta calidad.
    *   Asegurarse de que tenga un canal alfa (convertir a RGBA).
    *   Reemplazar su canal alfa con la máscara de alta resolución obtenida en el paso anterior.
    *   Guardar la imagen resultante en un formato que soporte transparencia, como **PNG**.

El resultado final es la imagen original con el fondo transparente, lista para ser utilizada en cualquier composición.