import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import sys
import os

# Añadir el directorio raíz del proyecto al sys.path para encontrar el módulo 'models'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models import UNetAutoencoder
from utils import ImageProcessor

st.set_page_config(page_title="Segmentación de Personas", layout="wide")

st.title("🤖 Aplicación de Segmentación de Personas")
st.write(
    "Sube una imagen que contenga una o más personas y la aplicación eliminará el fondo, "
    "dejando únicamente a las personas segmentadas. Utiliza un modelo U-Net con un backbone ResNet-50."
)

# --- Funciones Principales ---

@st.cache_resource
def load_model(model_path):
    """
    Carga el modelo de segmentación desde el checkpoint.
    La función se cachea para evitar recargar el modelo en cada ejecución.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Instanciar el modelo desde la clase importada
        model = UNetAutoencoder(pretrained=False, use_attention=True)
        
        # Cargar el checkpoint completo, que es un diccionario
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Extraer el state_dict del modelo, que está bajo la clave 'model_state_dict'
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en la ruta: {model_path}")
        st.info("Asegúrate de que el archivo 'best_segmentation.pth' se encuentra en el directorio 'checkpoints/'.")
        return None, None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        return None, None

def preprocess_image(image, image_size=384):
    """
    Preprocesa la imagen para que sea compatible con el modelo,
    replicando EXACTAMENTE el preprocesamiento del entrenamiento.
    Retorna tanto el tensor como los metadatos para la restauración.
    """
    # 1. Convertir PIL Image a numpy array
    image_np = np.array(image)

    # 2. Usar ImageProcessor para redimensionar con padding (como en el entrenamiento)
    dummy_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    processor = ImageProcessor()
    image_processed, _, restore_metadata = processor.resize_with_padding(
        image_np, dummy_mask, image_size
    )

    # 3. Normalizar a [0, 1] y convertir a tensor
    image_processed_float = image_processed.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_processed_float.transpose(2, 0, 1)).unsqueeze(0)

    return image_tensor, restore_metadata


def postprocess_and_visualize(original_image, model_output, restore_metadata):
    """
    Procesa la salida del modelo y genera imágenes de cada paso para depuración.
    Utiliza los metadatos para restaurar correctamente la máscara.
    """
    output_cpu = model_output.cpu().detach()[0]
    processor = ImageProcessor()

    # --- 1. Salida RGB cruda del modelo ---
    raw_rgb_pil = transforms.ToPILImage()(output_cpu[:3, :, :])

    # --- 2. Máscara de opacidad (alpha) cruda del modelo ---
    raw_mask_pil = transforms.ToPILImage()(output_cpu[3, :, :])

    # --- 3. Máscara restaurada al tamaño original (usando metadatos) ---
    raw_mask_np = np.array(raw_mask_pil)
    dummy_processed_image = np.zeros_like(raw_mask_np)
    _, restored_mask_np = processor.restore_original_size(
        dummy_processed_image, raw_mask_np, restore_metadata
    )
    restored_mask_pil = Image.fromarray(restored_mask_np)

    # --- 4. Aplicación de la máscara a la imagen original ---
    mask_np = restored_mask_np / 255.0
    original_np = np.array(original_image)
    mask_3d = np.stack([mask_np] * 3, axis=-1)
    segmented_image_np = (original_np * mask_3d).astype(np.uint8)
    final_result_pil = Image.fromarray(segmented_image_np)

    return {
        "raw_rgb": raw_rgb_pil,
        "raw_mask": raw_mask_pil,
        "restored_mask": restored_mask_pil,
        "final_result": final_result_pil
    }


# --- Carga del Modelo ---
# Construir la ruta absoluta al modelo para mayor robustez
MODEL_PATH = os.path.join(project_root, "checkpoints", "best_segmentation.pth")
model, device = load_model(MODEL_PATH)


# --- Interfaz de Usuario ---

if model:
    uploaded_file = st.file_uploader(
        "Elige una imagen...",
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen para segmentar. Los mejores resultados se obtienen con imágenes claras."
    )

    if uploaded_file is not None:
        # Leer la imagen subida
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(original_image, caption="🖼️ Imagen Original para Segmentar", use_container_width=True)
        st.markdown("---")

        # Botón para iniciar la segmentación
        if st.button("¡Segmentar Ahora!", use_container_width=True, type="primary"):
            with st.spinner('🧠 Realizando la segmentación y generando visualizaciones...'):
                # Preprocesar la imagen, obteniendo también los metadatos de restauración
                input_tensor, restore_metadata = preprocess_image(original_image)
                input_tensor = input_tensor.to(device)

                # Realizar la inferencia
                with torch.no_grad():
                    output_tensor = model(input_tensor)

                # Postprocesar y obtener todas las imágenes del proceso
                viz_dict = postprocess_and_visualize(original_image, output_tensor, restore_metadata)

                # Mostrar los resultados en un layout claro
                st.subheader("🔍 Visualización del Proceso de Segmentación")
                st.info("Aquí puedes ver los resultados de cada paso del modelo para diagnosticar la calidad.")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(viz_dict["raw_rgb"], caption="1. Salida RGB Cruda del Modelo", use_container_width=True)
                with col2:
                    st.image(viz_dict["raw_mask"], caption="2. Máscara de Opacidad Cruda (Capa Alpha)", use_container_width=True)

                col3, col4 = st.columns(2)
                with col3:
                    st.image(viz_dict["restored_mask"], caption="3. Máscara Restaurada (sin relleno)", use_container_width=True)
                with col4:
                    st.image(viz_dict["final_result"], caption="4. Resultado Final (Original × Máscara)", use_container_width=True)
else:
    st.warning("El modelo no pudo ser cargado. La aplicación no puede continuar.")
