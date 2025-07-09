import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50, resnet34
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import sys
import os
import time

# Añadir el directorio raíz del proyecto al sys.path para encontrar el módulo 'models'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models import UNetAutoencoder as UNetAutoencoder_ResNet50
from utils import ImageProcessor

st.set_page_config(page_title="Comparación de Modelos de Segmentación", layout="wide")

st.title("🤖 Comparación de Modelos de Segmentación: ResNet50 vs ResNet34")
st.write(
    "Esta aplicación permite comparar el rendimiento de dos modelos U-Net diferentes: "
    "uno con backbone ResNet-50 y otro con backbone ResNet-34. "
    "Sube una imagen y selecciona qué modelo(s) quieres probar."
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


class UNetEncoder_ResNet34(nn.Module):
    """
    Encoder path del U-Net con skip connections.
    Utiliza ResNet-34 pre-entrenado como backbone para extracción de características.
    """

    def __init__(self, pretrained=True):
        super(UNetEncoder_ResNet34, self).__init__()

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


class UNetDecoder_ResNet34(nn.Module):
    """
    Decoder path del U-Net con Attention Gates.
    Reconstruye la imagen enfocándose en las personas.
    """

    def __init__(self, use_attention=True):
        super(UNetDecoder_ResNet34, self).__init__()
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


class UNetAutoencoder_ResNet34(nn.Module):
    """
    U-Net Autoencoder completo para remoción de fondo.
    Combina encoder y decoder para generar imágenes con personas sin fondo.
    """

    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder_ResNet34, self).__init__()
        self.encoder = UNetEncoder_ResNet34(pretrained=pretrained)
        self.decoder = UNetDecoder_ResNet34(use_attention=use_attention)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded

# --- Funciones Principales ---

@st.cache_resource
def load_model_resnet50():
    """Carga el modelo ResNet50"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, "checkpoints", "best_segmentation.pth")
    
    try:
        model = UNetAutoencoder_ResNet50(pretrained=False, use_attention=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando modelo ResNet50: {e}")
        return None, None

@st.cache_resource
def load_model_resnet34():
    """Carga el modelo ResNet34"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, "checkpoints", "resnet34", "best_segmentation.pth")
    
    try:
        model = UNetAutoencoder_ResNet34(pretrained=False, use_attention=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando modelo ResNet34: {e}")
        return None, None

def preprocess_image(image, image_size=384):
    """
    Preprocesa la imagen para que sea compatible con el modelo,
    replicando EXACTAMENTE el preprocesamiento del entrenamiento.
    """
    # 1. Convertir PIL Image a numpy array
    image_np = np.array(image)

    # 2. Usar ImageProcessor para redimensionar con padding
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
    Procesa la salida del modelo y genera imágenes de cada paso.
    """
    output_cpu = model_output.cpu().detach()[0]
    processor = ImageProcessor()

    # --- 1. Salida RGB cruda del modelo ---
    raw_rgb_pil = transforms.ToPILImage()(output_cpu[:3, :, :])

    # --- 2. Máscara de opacidad (alpha) cruda del modelo ---
    raw_mask_pil = transforms.ToPILImage()(output_cpu[3, :, :])

    # --- 3. Máscara restaurada al tamaño original ---
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

def process_with_model(model, device, original_image, model_name):
    """
    Procesa una imagen con el modelo especificado y mide el tiempo.
    """
    start_time = time.time()
    
    # Preprocesar la imagen
    input_tensor, restore_metadata = preprocess_image(original_image)
    input_tensor = input_tensor.to(device)
    
    # Realizar la inferencia
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocesar
    viz_dict = postprocess_and_visualize(original_image, output_tensor, restore_metadata)
    
    processing_time = time.time() - start_time
    
    return viz_dict, processing_time

# --- Carga de Modelos ---
st.sidebar.header("⚙️ Configuración de Modelos")

# Cargar modelos
with st.sidebar:
    st.write("**Estado de los modelos:**")
    
    model_resnet50, device = load_model_resnet50()
    if model_resnet50:
        st.success("✅ ResNet50 cargado correctamente")
    else:
        st.error("❌ Error cargando ResNet50")
    
    model_resnet34, _ = load_model_resnet34()
    if model_resnet34:
        st.success("✅ ResNet34 cargado correctamente")
    else:
        st.error("❌ Error cargando ResNet34")

# Selector de modelos
model_options = []
if model_resnet50:
    model_options.append("ResNet50")
if model_resnet34:
    model_options.append("ResNet34")

if not model_options:
    st.error("❌ No se pudo cargar ningún modelo. Verifica que los archivos estén en las rutas correctas.")
    st.stop()

selected_models = st.sidebar.multiselect(
    "Selecciona los modelos a comparar:",
    options=model_options,
    default=model_options,
    help="Puedes seleccionar uno o ambos modelos para comparar"
)

if not selected_models:
    st.warning("⚠️ Por favor selecciona al menos un modelo para continuar.")
    st.stop()

# --- Interfaz de Usuario ---

uploaded_file = st.file_uploader(
    "Sube una imagen para segmentar:",
    type=["jpg", "jpeg", "png"],
    help="Sube una imagen que contenga personas para comparar el rendimiento de los modelos."
)

if uploaded_file is not None:
    # Leer la imagen subida
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.image(original_image, caption="🖼️ Imagen Original", use_container_width=True)
    st.markdown("---")

    # Botón para iniciar el procesamiento
    if st.button("🚀 Procesar con los modelos seleccionados", use_container_width=True, type="primary"):
        
        results = {}
        
        # Procesar con cada modelo seleccionado
        for model_name in selected_models:
            with st.spinner(f'🧠 Procesando con {model_name}...'):
                if model_name == "ResNet50":
                    viz_dict, processing_time = process_with_model(model_resnet50, device, original_image, model_name)
                elif model_name == "ResNet34":
                    viz_dict, processing_time = process_with_model(model_resnet34, device, original_image, model_name)
                
                results[model_name] = {
                    "viz_dict": viz_dict,
                    "processing_time": processing_time
                }
        
        # Mostrar resultados
        st.subheader("📊 Resultados de la Comparación")
        
        # Tabla de métricas
        if len(selected_models) > 1:
            st.write("**⏱️ Tiempo de procesamiento:**")
            metrics_data = []
            for model_name in selected_models:
                metrics_data.append({
                    "Modelo": model_name,
                    "Tiempo (segundos)": f"{results[model_name]['processing_time']:.3f}",
                    "Velocidad Relativa": f"{results[selected_models[0]]['processing_time'] / results[model_name]['processing_time']:.2f}x"
                })
            
            st.table(metrics_data)
            st.markdown("---")
        
        # Mostrar resultados de cada modelo
        if len(selected_models) == 1:
            # Un solo modelo - mostrar en formato completo
            model_name = selected_models[0]
            result = results[model_name]
            
            st.subheader(f"🔍 Resultados de {model_name}")
            st.info(f"Tiempo de procesamiento: {result['processing_time']:.3f} segundos")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(result["viz_dict"]["raw_rgb"], caption="1. Salida RGB del Modelo", use_container_width=True)
            with col2:
                st.image(result["viz_dict"]["raw_mask"], caption="2. Máscara de Opacidad", use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                st.image(result["viz_dict"]["restored_mask"], caption="3. Máscara Restaurada", use_container_width=True)
            with col4:
                st.image(result["viz_dict"]["final_result"], caption="4. Resultado Final", use_container_width=True)
        
        else:
            # Múltiples modelos - mostrar comparación lado a lado
            st.subheader("🔄 Comparación Lado a Lado")
            
            # Crear tabs para cada paso del proceso
            tab1, tab2, tab3, tab4 = st.tabs(["🎨 RGB Crudo", "🎭 Máscara Alpha", "🔧 Máscara Restaurada", "✨ Resultado Final"])
            
            with tab1:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["raw_rgb"], 
                                caption=f"{model_name} - RGB", 
                                use_container_width=True)
            
            with tab2:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["raw_mask"], 
                                caption=f"{model_name} - Máscara", 
                                use_container_width=True)
            
            with tab3:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["restored_mask"], 
                                caption=f"{model_name} - Restaurada", 
                                use_container_width=True)
            
            with tab4:
                cols = st.columns(len(selected_models))
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        st.image(results[model_name]["viz_dict"]["final_result"], 
                                caption=f"{model_name} - Final", 
                                use_container_width=True)

# --- Información adicional ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 Información de los Modelos")
    st.markdown("""
    **ResNet50:**
    - Más profundo (50 capas)
    - Mayor capacidad de representación
    - Más parámetros
    
    **ResNet34:**
    - Más ligero (34 capas)
    - Procesamiento más rápido
    - Menor uso de memoria
    """)