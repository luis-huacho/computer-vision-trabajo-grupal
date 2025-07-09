import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet34
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import sys
import os

# Añadir el directorio raíz del proyecto al sys.path para encontrar el módulo 'models'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils import ImageProcessor

st.set_page_config(page_title="Segmentación de Personas - ResNet34", layout="wide")

st.title("🤖 Aplicación de Segmentación de Personas - ResNet34")
st.write(
    "Sube una imagen que contenga una o más personas y la aplicación eliminará el fondo, "
    "dejando únicamente a las personas segmentadas. Utiliza un modelo U-Net con un backbone ResNet-34."
)

# --- Definición del modelo ResNet34 ---

class AttentionBlock(nn.Module):
    """
    Attention Gate para U-Net.
    Permite al modelo enfocarse en regiones importantes (personas).
    """

    def __init__(self, gate_channels, in_channels, inter_channels):
        super(AttentionBlock, self).__init__()

        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.input_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_conv = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gate):
        gate_conv = self.gate_conv(gate)
        input_conv = self.input_conv(x)

        # Asegurar mismas dimensiones usando interpolación
        if gate_conv.shape[2:] != input_conv.shape[2:]:
            gate_conv = F.interpolate(gate_conv, size=input_conv.shape[2:], mode='bilinear', align_corners=False)

        combined = self.relu(gate_conv + input_conv)
        attention = self.sigmoid(self.bn(self.output_conv(combined)))

        return x * attention


class DoubleConv(nn.Module):
    """
    Bloque de doble convolución usado en U-Net.
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    """
    Encoder path del U-Net con skip connections.
    Utiliza ResNet-34 pre-entrenado como backbone para extracción de características.
    """

    def __init__(self, pretrained=True):
        super(UNetEncoder, self).__init__()

        # Usar ResNet-34 pre-entrenado como backbone
        resnet = resnet34(pretrained=pretrained)

        # Extraer capas del ResNet
        self.conv1 = resnet.conv1  # 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels (ResNet-34 BasicBlock)
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Capas adicionales para el bottleneck - actualizado para ResNet-34
        self.bottleneck = DoubleConv(512, 1024, dropout_rate=0.2)

    def forward(self, x):
        # Encoder path con skip connections
        skip_connections = []

        # Initial convolution
        x1 = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x1)  # Skip 1: 64 channels

        x2 = self.maxpool(x1)

        # ResNet layers (ResNet-34 con BasicBlock)
        x3 = self.layer1(x2)
        skip_connections.append(x3)  # Skip 2: 64 channels

        x4 = self.layer2(x3)
        skip_connections.append(x4)  # Skip 3: 128 channels

        x5 = self.layer3(x4)
        skip_connections.append(x5)  # Skip 4: 256 channels

        x6 = self.layer4(x5)
        skip_connections.append(x6)  # Skip 5: 512 channels

        # Bottleneck
        x7 = self.bottleneck(x6)

        return x7, skip_connections


class UNetDecoder(nn.Module):
    """
    Decoder path del U-Net con Attention Gates.
    Reconstruye la imagen enfocándose en las personas.
    """

    def __init__(self, use_attention=True):
        super(UNetDecoder, self).__init__()
        self.use_attention = use_attention

        # Upsampling layers - actualizado para ResNet-34
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Attention gates - actualizado para ResNet-34
        if self.use_attention:
            self.att1 = AttentionBlock(512, 512, 256)   # layer4: 512 channels
            self.att2 = AttentionBlock(256, 256, 128)   # layer3: 256 channels
            self.att3 = AttentionBlock(128, 128, 64)    # layer2: 128 channels
            self.att4 = AttentionBlock(64, 64, 32)      # layer1: 64 channels
            self.att5 = AttentionBlock(64, 64, 32)      # conv1: 64 channels

        # Convolution blocks - actualizado para ResNet-34
        self.conv1 = DoubleConv(1024, 512)  # 512 (up1) + 512 (skip[0]) = 1024
        self.conv2 = DoubleConv(512, 256)   # 256 (up2) + 256 (skip[1]) = 512
        self.conv3 = DoubleConv(256, 128)   # 128 (up3) + 128 (skip[2]) = 256
        self.conv4 = DoubleConv(128, 64)    # 64 (up4) + 64 (skip[3]) = 128
        self.conv5 = DoubleConv(128, 64)    # 64 (up5) + 64 (skip[4]) = 128

        # Output layer - 4 channels (RGB + Alpha)
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)

    def _match_tensor_size(self, x, target_tensor):
        """Ajusta el tamaño de x para que coincida con target_tensor usando interpolación."""
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, skip_connections):
        # Decoder path - las skip connections están en orden inverso
        skips = skip_connections[::-1]  # [512, 256, 128, 64, 64] (ResNet-34)

        # Up 1: 1024 -> 512
        x = self.up1(x)  # Upsample
        skip = skips[0]  # 512 channels (layer4)

        # Asegurar que las dimensiones coincidan
        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att1(skip, x)

        x = torch.cat([x, skip], dim=1)  # 512 + 512 = 1024
        x = self.conv1(x)  # 1024 -> 512

        # Up 2: 512 -> 256
        x = self.up2(x)
        skip = skips[1]  # 256 channels (layer3)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att2(skip, x)

        x = torch.cat([x, skip], dim=1)  # 256 + 256 = 512
        x = self.conv2(x)  # 512 -> 256

        # Up 3: 256 -> 128
        x = self.up3(x)
        skip = skips[2]  # 128 channels (layer2)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att3(skip, x)

        x = torch.cat([x, skip], dim=1)  # 128 + 128 = 256
        x = self.conv3(x)  # 256 -> 128

        # Up 4: 128 -> 64
        x = self.up4(x)
        skip = skips[3]  # 64 channels (layer1)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att4(skip, x)

        x = torch.cat([x, skip], dim=1)  # 64 + 64 = 128
        x = self.conv4(x)  # 128 -> 64

        # Up 5: 64 -> 64 (final upsampling)
        x = self.up5(x)
        skip = skips[4]  # 64 channels (conv1)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att5(skip, x)

        x = torch.cat([x, skip], dim=1)  # 64 + 64 = 128
        x = self.conv5(x)  # 128 -> 64

        # Final output
        x = self.final_conv(x)  # 64 -> 4 (RGBA)

        # Aplicar activaciones
        rgb = torch.sigmoid(x[:, :3])  # RGB channels
        alpha = torch.sigmoid(x[:, 3:4])  # Alpha channel

        return torch.cat([rgb, alpha], dim=1)


class UNetAutoencoder(nn.Module):
    """
    U-Net Autoencoder completo para remoción de fondo.
    Combina encoder y decoder para generar imágenes con personas sin fondo.
    """

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder, self).__init__()
        self.encoder = UNetEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded

# --- Funciones Principales ---

@st.cache_resource
def load_model(model_path):
    """
    Carga el modelo de segmentación desde el checkpoint.
    La función se cachea para evitar recargar el modelo en cada ejecución.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Instanciar el modelo desde la clase definida arriba
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
        st.info("Asegúrate de que el archivo 'best_segmentation.pth' se encuentra en el directorio 'checkpoints/resnet34/'.")
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
MODEL_PATH = os.path.join(project_root, "checkpoints", "resnet34", "best_segmentation.pth")
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
            with st.spinner('🧠 Realizando la segmentación con ResNet-34 y generando visualizaciones...'):
                # Preprocesar la imagen, obteniendo también los metadatos de restauración
                input_tensor, restore_metadata = preprocess_image(original_image)
                input_tensor = input_tensor.to(device)

                # Realizar la inferencia
                with torch.no_grad():
                    output_tensor = model(input_tensor)

                # Postprocesar y obtener todas las imágenes del proceso
                viz_dict = postprocess_and_visualize(original_image, output_tensor, restore_metadata)

                # Mostrar los resultados en un layout claro
                st.subheader("🔍 Visualización del Proceso de Segmentación - ResNet34")
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