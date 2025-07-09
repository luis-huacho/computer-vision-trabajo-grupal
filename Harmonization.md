# Informe del Proceso de Entrenamiento de Armonización

A continuación se detallan las características y los parámetros del proceso de entrenamiento para la armonización de imágenes, basado en el análisis del archivo `harmonization.py`.

### Flujo de Datos del Modelo (Input y Output)

El modelo de armonización toma una imagen compuesta (persona sobre un fondo) y la transforma para que la persona se integre de manera más natural con el nuevo fondo, ajustando iluminación y color.

#### Input (Entrada)

El **input** del modelo es una **imagen RGB compuesta** que ha sido pre-procesada. El proceso para cada imagen es el siguiente:

1.  **Fuente:** Se genera una imagen compuesta sintética mezclando una imagen de *foreground* (persona con canal alfa) con una imagen de *background* (fondo).
2.  **Redimensionamiento:** La imagen compuesta se redimensiona a un tamaño fijo, por defecto **384x384 píxeles**.
3.  **Aumentación de Datos:** Se aplican transformaciones aleatorias (solo en el conjunto de entrenamiento) como ajustes de brillo, contraste, tono, saturación y color para hacer el modelo más robusto.
4.  **Normalización:** Los valores de los píxeles de la imagen (que originalmente van de 0 a 255) se normalizan a un rango de **0.0 a 1.0**.
5.  **Formato (Tensor):** Finalmente, la imagen se convierte en un Tensor de PyTorch con el formato `[Canales, Altura, Anchura]`.

En resumen, el input es un **tensor de dimensiones `(3, 384, 384)`** que representa una imagen RGB compuesta normalizada.

#### Output (Salida)

El **output** del modelo es una **imagen RGB armonizada** del mismo tamaño que la entrada. El modelo predice cómo debería verse la imagen compuesta para que la persona se integre de forma natural con el fondo.

*   **Canales 1, 2 y 3 (RGB):** La imagen compuesta, pero con los ajustes de iluminación y color aplicados para armonizar el *foreground* con el *background*.

En resumen, el output es un **tensor de dimensiones `(3, 384, 384)`** que representa la imagen RGB armonizada normalizada.

### Arquitectura del Modelo (UNetHarmonizer y sus Componentes Clave)

El modelo de armonización utiliza la arquitectura `UNetHarmonizer`, que es una variante de la popular red U-Net, adaptada para la tarea de ajuste de color e iluminación.

#### Componentes Principales: Encoder y Decoder

1.  **Encoder (Camino de Contracción):**
    *   Reduce progresivamente la resolución espacial de la imagen de entrada mientras extrae características de alto nivel.
    *   A diferencia del modelo de segmentación, este `UNetHarmonizer` construye su propio encoder (`_make_layer`) y no carga un encoder pre-entrenado de una red como VGG o ResNet directamente en su estructura principal para la extracción de características iniciales.

2.  **Decoder (Camino de Expansión):**
    *   Toma la representación comprimida del encoder y la expande progresivamente para reconstruir la imagen armonizada a la resolución original de la entrada.

#### Skip Connections (Conexiones de Salto)

Al igual que en una U-Net estándar, las `UNetHarmonizer` utiliza **skip connections** (conexiones de salto). Estas son "atajos" que conectan directamente las capas del encoder con las capas correspondientes del decoder.

*   **Función:** Permiten que la información de alta resolución y los detalles espaciales finos (que se pierden durante la contracción en el encoder) sean transferidos directamente al decoder.
*   **Beneficio:** Son cruciales para preservar los detalles de la imagen original mientras se aplican los ajustes de armonización, evitando que la imagen de salida se vea borrosa o pierda nitidez.

#### Espacio Latente (Embedding)

En el punto más profundo y compacto de la U-Net, justo entre el final del encoder y el inicio del decoder (conocido como la capa de cuello de botella o *bottleneck*), se forma un **embedding** o representación latente de la imagen de entrada.

*   **Naturaleza:** Esta es una representación comprimida y de baja dimensión de la imagen compuesta original.
*   **Contenido:** Contiene la información semántica de alto nivel necesaria para entender la relación entre el *foreground* y el *background* y cómo deben armonizarse.
*   **Rol:** Este embedding sirve como la base abstracta a partir de la cual el decoder comienza su proceso de reconstrucción de la imagen armonizada.

### Características Clave del Proceso de Entrenamiento

El entrenamiento de armonización es un proceso complejo que busca no solo la similitud de píxeles, sino también la coherencia visual y estilística.

1.  **Dataset y Creación de Composiciones:**
    *   Utiliza el `HarmonizationDataset`, que requiere directorios separados para imágenes de *foregrounds* (personas con canal alfa) y *backgrounds* (fondos).
    *   Durante el entrenamiento, se generan **imágenes compuestas sintéticas** mezclando aleatoriamente *foregrounds* con *backgrounds*.
    *   **Nota Importante:** El *target* (la verdad fundamental) para el entrenamiento es la **propia imagen compuesta sintética**. Esto significa que el modelo aprende a transformar la imagen compuesta para que se vea más natural, basándose en la suposición de que una composición bien hecha debería parecerse a sí misma después de la armonización, o que el modelo debe aprender a "corregir" las inconsistencias de la composición inicial.

2.  **Aumentación de Datos:**
    *   Se aplican transformaciones específicas para armonización, como `HorizontalFlip`, `RandomBrightnessContrast`, `HueSaturationValue`, `ColorJitter` y `RandomGamma`, para mejorar la robustez del modelo a variaciones de iluminación y color.

3.  **Función de Pérdida (HarmonizationLossCalculator):**
    *   Esta es una de las partes más sofisticadas del entrenamiento, ya que combina múltiples tipos de pérdidas para guiar al modelo hacia una armonización visualmente convincente. Las pérdidas se combinan con pesos (`alpha`, `beta`, `gamma`, `delta`).
    *   **Pérdida MSE (`mse_loss`):** Mide la diferencia cuadrática media píxel a píxel entre la imagen predicha y la imagen objetivo. Es una pérdida básica de reconstrucción.
    *   **Pérdida Perceptual (`perceptual_loss`):** Utiliza las características extraídas por una red VGG pre-entrenada (hasta la capa `conv3_3`). Esto ayuda al modelo a generar imágenes que no solo son correctas a nivel de píxel, sino que también son visualmente similares y realistas para el ojo humano.
    *   **Pérdida de Consistencia de Color (`color_consistency_loss`):** Intenta asegurar que los colores de la imagen armonizada sean coherentes. Realiza una conversión aproximada a un espacio de color LAB y pone un mayor énfasis en la consistencia de los componentes `a` y `b` (cromaticidad) que en la luminancia (`L`).
    *   **Pérdida de Estilo (`style_loss`):** Utiliza la matriz de Gram para comparar el "estilo" de la imagen predicha con el de la imagen objetivo. Esto ayuda a que la textura y el patrón general de la imagen armonizada coincidan con los del objetivo, contribuyendo a una integración más fluida.

4.  **Optimizador y Planificador de Tasa de Aprendizaje:**
    *   **Optimizador:** Se usa **Adam**, un optimizador adaptativo estándar.
    *   **Planificador (Scheduler):** Se emplea `CosineAnnealingWarmRestarts`, que ajusta cíclicamente la tasa de aprendizaje para mejorar la exploración del espacio de soluciones.

5.  **Métricas y Evaluación:**
    *   El rendimiento se mide principalmente por la pérdida total (`total_loss`), así como por las pérdidas individuales (MSE, perceptual) para monitorear el progreso.

6.  **Guardado de Modelos (Checkpoints):**
    *   El script guarda automáticamente el modelo después de cada época.
    *   Se guarda el **mejor modelo** basándose en la **pérdida de validación** (`val_loss`).

7.  **Entrenamiento Distribuido (DDP):**
    *   El entrenamiento está configurado para ejecutarse en un entorno multi-GPU utilizando `torch.distributed.DistributedDataParallel (DDP)` y `DistributedSampler`, lo que permite escalar el entrenamiento a múltiples dispositivos.

### Parámetros de Entrenamiento

Aquí están los parámetros de entrenamiento con sus valores por defecto, tal como están definidos en la función `train_harmonization_model` en `harmonization.py`:

| Parámetro | Valor por Defecto | Descripción |
| :--- | :--- | :--- |
| `batch_size` | `2` | Número de imágenes procesadas en cada paso del entrenamiento. (Nota: Valor bajo por defecto, puede requerir ajuste según el dataset y GPU). |
| `learning_rate` | `5e-5` (0.00005) | Tasa de aprendizaje inicial para el optimizador Adam. |
| `weight_decay` | `1e-6` (0.000001) | Técnica de regularización para prevenir el sobreajuste. |
| `num_epochs` | `50` | Número total de veces que el modelo verá el dataset completo. |
| `image_size` | `384` | Las imágenes se redimensionan a 384x384 píxeles antes de entrar al modelo. |
| `num_workers` | `4` | Número de subprocesos para cargar los datos en paralelo. |
| `pin_memory` | `True` | Optimización que acelera la transferencia de datos de la CPU a la GPU. |
| `device` | `cuda:{device_id}` | Se establece automáticamente para usar la GPU disponible. |

### Cómo Ejecutar el Entrenamiento de Armonización

Para lanzar el entrenamiento del modelo de armonización, se debe ejecutar el script `harmonization.py`. A continuación se muestran los comandos recomendados según el entorno.

**Nota Importante:** Estos comandos inician **únicamente** el proceso de entrenamiento para el modelo de **armonización**.

#### Ejecución en Múltiples GPUs

Para aprovechar todas las GPUs disponibles y acelerar el entrenamiento, se debe usar el lanzador `torchrun`.

```bash
torchrun harmonization.py
```

#### Ejecución en una Sola GPU (Entorno Compartido)

Para limitar el entrenamiento a una única GPU específica y no saturar un servidor compartido, se recomienda usar la variable de entorno `CUDA_VISIBLE_DEVICES`. Este método es el más directo para controlar el uso de recursos.

```bash
# Ejemplo para usar solo la primera GPU (índice 0)
CUDA_VISIBLE_DEVICES=0 python harmonization.py

# Ejemplo para usar solo la segunda GPU (índice 1)
CUDA_VISIBLE_DEVICES=1 python harmonization.py
```

Alternativamente, se puede usar `torchrun` para lanzar un solo proceso:

```bash
torchrun --nproc_per_node=1 harmonization.py
```

### Uso en Producción: Inferencia de Armonización

El archivo `harmonization.py` también incluye una clase `HarmonizationInference` diseñada para aplicar el modelo entrenado a nuevas composiciones. Esta clase permite cargar un modelo guardado y armonizar imágenes RGB compuestas.

#### Proceso de Inferencia:

1.  **Carga del Modelo:** Se inicializa `HarmonizationInference` con la ruta al modelo entrenado (`best_harmonizer.pth`).
2.  **Pre-procesamiento:** La imagen RGB compuesta de entrada se redimensiona a 384x384 y se normaliza para que coincida con el formato de entrada del modelo.
3.  **Inferencia:** La imagen pre-procesada se pasa a través del `UNetHarmonizer` para obtener la imagen armonizada en 384x384.
4.  **Post-procesamiento:**
    *   La imagen armonizada se escala de nuevo a las dimensiones originales de la entrada.
    *   Se puede aplicar un filtro de nitidez (`_apply_sharpening_filter`) para restaurar detalles.
    *   Se puede mezclar la imagen armonizada con la imagen original (`blend_factor`) para preservar ciertos detalles o texturas de la composición inicial.
5.  **Salida:** La imagen RGB armonizada final puede ser guardada o utilizada directamente para su visualización o composición posterior.
