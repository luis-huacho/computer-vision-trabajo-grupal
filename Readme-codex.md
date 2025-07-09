# Guía rápida de comandos

A continuación tienes el “gran resumen” de los pasos y comandos mínimos que debes ejecutar para poner en marcha todo el flujo del proyecto (entrenamiento con COCO + verificación + demo o app web). Hemos ignorado por completo el fichero `Readme.md` de la raíz, y tomado la información de las demás guías/documentación del repositorio.

---

## 1. Clonar y preparar el entorno base

```bash
# 1. Clonar el repositorio y situarse en él
git clone <URL-DEL-REPO>
cd computer-vision-trabajo-grupal

# 2. Crear y activar un entorno virtual
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\Activate.ps1

# 3. Instalar dependencias core para entrenamiento/inferencia
pip install -r requirements.txt
```

---

## 2. Descargar y descomprimir el dataset COCO

```bash
# Entrar en la carpeta COCO (la crea si no existe)
mkdir -p COCO && cd COCO

# 2.1 Descargar anotaciones e imágenes
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# 2.2 Descomprimir todos los ZIPs y (opcional) borrar los ZIP para ahorrar espacio
unzip annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
rm *.zip

# Volver al directorio raíz del proyecto
cd ..
```

---

## 3. Verificar la estructura y el sistema

```bash
# Verificación rápida de estructura COCO (~30 s)
python main.py quick

# Verificación completa del sistema (~2–3 min)
python main.py verify

# (Opcional) Análisis estadístico del dataset
python main.py analyze
```

---

## 4. Entrenar los modelos

```bash
# 4.1 Modo automático: hace verify + train (segmentación + harmonización)
python main.py

# 4.2 Entrenamiento directo de segmentación (sin verify)
python main.py segmentation

# 4.3 Entrenamiento directo de harmonización (sin verify)
python main.py harmonization

# 4.4 Sólo entrenamiento (sin menú interactivo ni DDP automático)
python main.py train
```

---

## 5. Probar la inferencia (demo)

```bash
python main.py demo
```

---

## 6. (Opcional) Configurar datasets de harmonización y ejemplos

```bash
python main.py setup
```

---

## 7. Menú interactivo y utilidades

```bash
python main.py           # Arranca menú interactivo
python main.py help      # Muestra todas las opciones
python main.py status    # Estado de los módulos
python main.py config    # Imprime la configuración actual
```

---

## 8. Arrancar la aplicación web con Streamlit

```bash
# Instalar dependencias específicas de la app
pip install -r requirements-app.txt

# Ejecutar Streamlit
streamlit run app.py
```

---

### TL;DR — Pack completo en 3 pasos

```bash
# 1. Setup general
git clone <repo> && cd computer-vision-trabajo-grupal
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Dataset COCO
mkdir COCO && cd COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip *.zip && cd ..

# 3. Verificar + entrenar
python main.py verify
python main.py
```

---

Con esto tienes todos los comandos clave para iniciar el flujo completo: desde la creación del entorno y la descarga del dataset, hasta la verificación, el entrenamiento de los modelos y el lanzamiento de la app web. ¡Éxitos con el proyecto!