# 🎭 Removedor de Fondo con IA - Aplicación Web

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

Aplicación web desarrollada con Streamlit que utiliza un modelo U-Net avanzado para remover automáticamente el fondo de imágenes, manteniendo solo las personas detectadas.

**Desarrollado por:** Luis Huacho y Dominick Alvarez  
**Institución:** Maestría en Informática, PUCP

## 🚀 Características

- 🤖 **IA Avanzada**: Modelo U-Net con Attention Gates
- 🎯 **Precisión Alta**: Entrenado en dataset Supervisely Persons
- 🖼️ **Procesamiento Inteligente**: Mantiene dimensiones originales
- 📱 **Interfaz Intuitiva**: Drag & drop para subir imágenes
- ⚡ **Optimizado**: Funciona en CPU y GPU
- 📊 **Modo Debug**: Visualiza todo el proceso paso a paso
- 💾 **Descarga Directa**: Resultado en PNG con transparencia

## 🛠️ Instalación Rápida

### Prerrequisitos

- Python 3.8 o superior
- Modelo entrenado (`checkpoints/best_model.pth`)

### 1. Preparar Entorno

```bash
# Clonar repositorio
git clone <repository-url>
cd unet-background-removal

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install streamlit torch torchvision opencv-python pillow numpy matplotlib
```

### 2. Verificar Modelo

```bash
# Asegurarse de que el modelo existe
ls checkpoints/best_model.pth

# Si no existe, entrenar primero:
python run_training.py
```

### 3. Ejecutar Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

## 🎯 Modo de Uso

### Interfaz Principal

1. **Subir Imagen**: Arrastra o selecciona una imagen (JPG, PNG, JPEG)
2. **Configurar**: Ajusta el tamaño de procesamiento (128-512px)
3. **Procesar**: Haz clic en "🚀 Procesar Imagen"
4. **Visualizar**: Observa el resultado paso a paso
5. **Descargar**: Obtén el archivo PNG sin fondo

### Modo Debug

La aplicación incluye un modo debug completo que muestra:

- **Imagen Original**: Como se subió
- **Imagen Redimensionada**: Con padding para el modelo
- **Máscara Generada**: Canal alpha del modelo
- **Resultado Procesado**: Antes de restaurar dimensiones
- **Resultado Final**: Restaurado al tamaño original

### Análisis Automático

- **Métricas de Calidad**: Puntuación automática del resultado
- **Estadísticas**: Cobertura de personas, contraste, definición
- **Recomendaciones**: Consejos para mejores resultados
- **Análisis por Zonas**: Distribución de personas en la imagen

## 📋 Requisitos del Sistema

### Mínimos (CPU)
- **RAM**: 4 GB
- **Procesador**: Intel i5 o AMD equivalente
- **Tiempo**: 15-30 segundos por imagen

### Recomendados (GPU)
- **RAM**: 8 GB
- **GPU**: NVIDIA GTX 1060+ (2GB VRAM)
- **Tiempo**: 1-5 segundos por imagen

## 🌐 Despliegue

### 1. Streamlit Community Cloud (Gratis)

```bash
# Subir a GitHub
git add .
git commit -m "Deploy app"
git push origin main

# Desplegar en share.streamlit.io
# 1. Ve a share.streamlit.io
# 2. Conecta tu repositorio
# 3. ¡Listo!
```

### 2. Docker Local

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
# Construir y ejecutar
docker build -t bg-removal-app .
docker run -p 8501:8501 bg-removal-app
```

### 3. Heroku

```bash
# Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Desplegar
heroku create tu-app-name
git push heroku main
```

## ⚙️ Configuración Avanzada

### Optimizar para CPU

```python
# En app.py, agregar al inicio:
import torch
torch.set_num_threads(2)
torch.backends.cudnn.enabled = False

# Reducir tamaño de procesamiento
image_size = 256  # En lugar de 384
```

### Personalizar Interfaz

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

### Variables de Entorno

```bash
# .env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 🔧 Solución de Problemas

### "Modelo no encontrado"

```bash
# Verificar archivo
ls -la checkpoints/best_model.pth

# Entrenar si es necesario
python run_training.py
```

### "Out of memory"

```python
# Reducir tamaño en configuración de sidebar
processing_size = st.slider("Tamaño de procesamiento", 128, 256, 128)
```

### "Port already in use"

```bash
# Usar puerto diferente
streamlit run app.py --server.port 8502

# O detener proceso
pkill -f streamlit
```

### Rendimiento Lento

```python
# Optimizaciones en app.py:
# 1. Usar cache para el modelo
@st.cache_resource
def load_model():
    return BackgroundRemoverDebug(model_path, device)

# 2. Optimizar threads
torch.set_num_threads(2)

# 3. Reducir tamaño de imagen
image_size = 256
```

## 📊 Métricas de Rendimiento

| Dispositivo | Tiempo/Imagen | RAM | Calidad |
|-------------|---------------|-----|---------|
| CPU (i5)    | 20-30s       | 3GB | Alta    |
| CPU (i7)    | 15-25s       | 3GB | Alta    |
| GPU (1060)  | 3-5s         | 2GB | Alta    |
| GPU (3080)  | 1-2s         | 2GB | Alta    |

## 🎨 Funcionalidades Destacadas

### Análisis Técnico Detallado

- **Dimensiones**: Original vs procesado
- **Estadísticas de Máscara**: Cobertura, contraste, definición
- **Análisis por Zonas**: Distribución 3x3 de la imagen
- **Histograma**: Distribución de valores en la máscara
- **Mapa de Calor**: Visualización de zonas con personas

### Consejos Inteligentes

El sistema proporciona recomendaciones automáticas:

- ✅ **Mejores prácticas**: Iluminación, contraste, resolución
- ⚠️ **Evitar**: Fondos similares, personas cortadas, baja resolución
- 📊 **Puntuación automática**: Análisis de calidad de 0-100

### Descargas Múltiples

- **Resultado Final**: PNG con transparencia en tamaño original
- **Máscara**: PNG en escala de grises
- **Resultado Procesado**: Versión intermedia antes de restaurar

## 🚀 Inicio Rápido - 3 Pasos

### 1. Preparar
```bash
git clone <repo> && cd unet-background-removal
pip install streamlit torch torchvision opencv-python pillow numpy
```

### 2. Verificar
```bash
ls checkpoints/best_model.pth  # Debe existir
```

### 3. Ejecutar
```bash
streamlit run app.py
```

¡La aplicación estará lista en `http://localhost:8501`!

## 📱 Acceso Móvil

La aplicación es completamente responsive y funciona en:

- 📱 **Móviles**: iPhone, Android
- 💻 **Tablets**: iPad, Android tablets
- 🖥️ **Desktop**: Windows, Mac, Linux

## 🤝 Contribuir

Este proyecto forma parte de la investigación en Computer Vision de la Maestría en Informática - PUCP.

**Desarrolladores:**
- **Luis Huacho** - Implementación del modelo y entrenamiento
- **Dominick Alvarez** - Arquitectura y optimización

### Mejoras Futuras

- [ ] Soporte para múltiples personas
- [ ] Detección de objetos adicionales
- [ ] Procesamiento en tiempo real
- [ ] API REST para integración
- [ ] Modo batch para múltiples imágenes

## 📞 Soporte

### Problemas Frecuentes

- **Modelo no carga**: Verificar `checkpoints/best_model.pth`
- **Lento en CPU**: Reducir tamaño a 256px o usar GPU
- **Error de memoria**: Usar CPU en lugar de GPU
- **Calidad baja**: Usar imágenes con mejor iluminación y contraste

### Contacto

- 🐛 **Issues**: GitHub Issues del repositorio
- 💬 **Discusiones**: GitHub Discussions
- 📧 **Email**: Contacto directo con los desarrolladores

## 📄 Licencia

Proyecto desarrollado bajo Licencia MIT para fines académicos y de investigación en la PUCP.
**Desarrollado con ❤️ en la PUCP**