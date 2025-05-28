# 🎭 Removedor de Fondo con IA - U-Net Autoencoder

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

Una aplicación web inteligente que utiliza un modelo U-Net con Attention Gates para remover automáticamente el fondo de imágenes, manteniendo solo las personas detectadas.

## 📸 Demo

![Demo Animation](https://via.placeholder.com/800x400/1f1f1f/ffffff?text=Demo+de+la+Aplicación)

## 🚀 Características

- 🤖 **Modelo IA avanzado**: U-Net con Attention Gates entrenado en dataset Supervisely
- 🖼️ **Procesamiento inteligente**: Detecta y mantiene solo personas en la imagen
- 🎨 **Interfaz intuitiva**: Drag & drop para subir imágenes
- 📱 **Responsive**: Funciona en desktop y móvil
- ⚡ **Rápido**: Procesamiento optimizado para CPU y GPU
- 💾 **Descarga directa**: Resultado en PNG con transparencia
- 📊 **Métricas**: Estadísticas del procesamiento

## 🛠️ Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip o conda
- Modelo entrenado (`best_model.pth`)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/bg-removal-ai.git
cd bg-removal-ai
```

### 2. Crear entorno virtual (recomendado)

```bash
# Con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Con conda
conda create -n bg-removal python=3.9
conda activate bg-removal
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Estructura del proyecto

```
bg-removal-ai/
├── app.py                 # Aplicación Streamlit principal
├── main.py               # Código de entrenamiento del modelo
├── requirements.txt      # Dependencias
├── README-ia.md         # Este archivo
├── checkpoints/         # Modelos entrenados
│   └── best_model.pth   # Modelo principal (requerido)
├── logs/                # Logs de entrenamiento
├── plots/               # Gráficas de entrenamiento
└── temp/                # Archivos temporales
```

## 🎯 Uso Rápido

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Pasos de uso

1. **Subir imagen**: Arrastra o selecciona una imagen (JPG, PNG, JPEG)
2. **Procesar**: Haz clic en "🚀 Procesar Imagen"
3. **Descargar**: Obtén el resultado sin fondo en PNG

## 📋 Requisitos del Sistema

### Mínimos (CPU)
- **RAM**: 4 GB
- **Procesador**: Intel i5 o AMD equivalente
- **Tiempo de procesamiento**: 15-30 segundos por imagen

### Recomendados (GPU)
- **RAM**: 8 GB
- **GPU**: NVIDIA GTX 1060 o superior (2GB VRAM)
- **Tiempo de procesamiento**: 1-5 segundos por imagen

## 🌐 Opciones de Despliegue

### 1. Streamlit Community Cloud (Gratis) ⭐

**Más fácil y gratuito**

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. ¡Listo! Tu app estará disponible públicamente

```yaml
# .streamlit/config.toml (opcional)
[server]
maxUploadSize = 200

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### 2. Docker (Local/Servidor)

**Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

**docker-compose.yml**
```yaml
version: '3.8'
services:
  bg-removal:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./checkpoints:/app/checkpoints
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

**Comandos:**
```bash
# Construir y ejecutar
docker-compose up --build

# Solo ejecutar
docker-compose up -d
```

### 3. Heroku

**Procfile**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt**
```
python-3.9.16
```

**Comandos:**
```bash
# Instalar Heroku CLI y ejecutar
heroku create tu-app-name
git push heroku main
```

### 4. Railway

1. Conecta tu repositorio en [railway.app](https://railway.app)
2. Configura las variables de entorno
3. ¡Despliega automáticamente!

### 5. Render

1. Conecta tu repositorio en [render.com](https://render.com)
2. Selecciona "Web Service"
3. Configura:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

## ⚙️ Optimizaciones

### Para CPU (Despliegue en la nube)

```python
# Agregar al inicio de app.py
import torch
torch.set_num_threads(2)  # Limitar threads
torch.backends.cudnn.enabled = False  # Deshabilitar CUDA

# En la función remove_background, cambiar:
image_size = 256  # Reducir de 384 a 256 para mayor velocidad
```

### Para GPU (Servidor local)

```python
# Verificar disponibilidad de GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Opcional: especificar GPU específica
# device = 'cuda:0'
```

## 🔧 Solución de Problemas

### Error: "Modelo no encontrado"

```bash
# Verificar que el archivo existe
ls -la checkpoints/best_model.pth

# Si no existe, necesitas entrenar el modelo primero
python main.py
```

### Error: "Out of memory"

```python
# Reducir tamaño de imagen
image_size = 256  # En lugar de 384

# Procesar en CPU
device = 'cpu'
```

### Error: "Module not found"

```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Verificar instalación de PyTorch
python -c "import torch; print(torch.__version__)"
```

### Error: "Port already in use"

```bash
# Usar puerto diferente
streamlit run app.py --server.port 8502

# O detener proceso existente
lsof -ti:8501 | xargs kill -9
```

## 📊 Métricas de Rendimiento

| Dispositivo | Tiempo/Imagen | RAM Usada | Calidad |
|-------------|---------------|-----------|---------|
| CPU (i5)    | 20-30s       | 2-4 GB    | Alta    |
| CPU (i7)    | 15-25s       | 2-4 GB    | Alta    |
| GPU (1060)  | 3-5s         | 1-2 GB    | Alta    |
| GPU (3080)  | 1-2s         | 1-2 GB    | Alta    |

## 🎨 Personalización

### Cambiar tema

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"           # Color principal
backgroundColor = "#FFFFFF"        # Fondo
secondaryBackgroundColor = "#F0F2F6"  # Fondo secundario
textColor = "#262730"              # Texto
```

### Modificar límites

```python
# En app.py, cambiar:
st.file_uploader(
    "Elige una imagen...",
    type=['png', 'jpg', 'jpeg'],
    # Cambiar límite de tamaño (en MB)
    help="Máximo 10MB por imagen"
)
```

## 📱 Variables de Entorno

```bash
# .env (opcional)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

## 🚀 Despliegue Rápido - Guía Paso a Paso

### Opción 1: Streamlit Cloud (Recomendado para principiantes)

1. **Preparar repositorio**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/tu-usuario/bg-removal-ai.git
   git push -u origin main
   ```

2. **Desplegar**:
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Conecta tu cuenta de GitHub
   - Selecciona tu repositorio
   - ¡Listo en 2 minutos!

### Opción 2: Docker (Para desarrolladores)

```bash
# Comando único
docker run -p 8501:8501 -v $(pwd)/checkpoints:/app/checkpoints tu-app:latest
```

### Opción 3: Servidor local

```bash
# Instalar y ejecutar
pip install -r requirements.txt
streamlit run app.py
```

## 📞 Soporte

### Problemas comunes

- **¿El modelo no carga?** → Verifica que `checkpoints/best_model.pth` existe
- **¿Muy lento?** → Reduce `image_size` a 256 o usa GPU
- **¿Error de memoria?** → Usa CPU en lugar de GPU
- **¿No encuentra módulos?** → Reinstala dependencias

### Contacto

- 🐛 **Issues**: [GitHub Issues](https://github.com/tu-usuario/bg-removal-ai/issues)
- 💬 **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/bg-removal-ai/discussions)
- 📧 **Email**: tu-email@ejemplo.com

## 🤝 Contribuir

¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## ⭐ Reconocimientos

- **Dataset**: Supervisely Person Dataset
- **Arquitectura**: U-Net con Attention Gates
- **Framework**: Streamlit, PyTorch
- **Inspiración**: Investigación en Computer Vision y Deep Learning

---

### 🎉 ¡Listo para usar!

Con esta guía puedes desplegar la aplicación en menos de 10 minutos. Para principiantes, recomendamos empezar con **Streamlit Community Cloud** por su simplicidad.

**¿Dudas?** Abre un [issue](https://github.com/tu-usuario/bg-removal-ai/issues) y te ayudaremos.

---

**Desarrollado con ❤️ y mucho ☕**