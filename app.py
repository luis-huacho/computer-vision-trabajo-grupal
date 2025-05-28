import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import numpy as np
import cv2
import os
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="Removedor de Fondo con IA",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar las clases del modelo desde main.py (asumiendo que están en el mismo directorio)
class AttentionBlock(nn.Module):
    """Attention Gate para U-Net."""
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
        
        if gate_conv.shape[2:] != input_conv.shape[2:]:
            gate_conv = F.interpolate(gate_conv, size=input_conv.shape[2:], mode='bilinear', align_corners=False)
        
        combined = self.relu(gate_conv + input_conv)
        attention = self.sigmoid(self.bn(self.output_conv(combined)))
        
        return x * attention

class DoubleConv(nn.Module):
    """Bloque de doble convolución usado en U-Net."""
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
    """Encoder path del U-Net con skip connections."""
    def __init__(self, pretrained=True):
        super(UNetEncoder, self).__init__()
        
        resnet = resnet34(pretrained=pretrained)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.bottleneck = DoubleConv(512, 1024, dropout_rate=0.2)
        
    def forward(self, x):
        skip_connections = []
        
        x1 = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x1)
        
        x2 = self.maxpool(x1)
        
        x3 = self.layer1(x2)
        skip_connections.append(x3)
        
        x4 = self.layer2(x3)
        skip_connections.append(x4)
        
        x5 = self.layer3(x4)
        skip_connections.append(x5)
        
        x6 = self.layer4(x5)
        skip_connections.append(x6)
        
        x7 = self.bottleneck(x6)
        
        return x7, skip_connections

class UNetDecoder(nn.Module):
    """Decoder path del U-Net con Attention Gates."""
    def __init__(self, use_attention=True):
        super(UNetDecoder, self).__init__()
        self.use_attention = use_attention
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        if self.use_attention:
            self.att1 = AttentionBlock(512, 512, 256)
            self.att2 = AttentionBlock(256, 256, 128)
            self.att3 = AttentionBlock(128, 128, 64)
            self.att4 = AttentionBlock(64, 64, 32)
            self.att5 = AttentionBlock(64, 64, 32)
        
        self.conv1 = DoubleConv(1024, 512)
        self.conv2 = DoubleConv(512, 256)
        self.conv3 = DoubleConv(256, 128)
        self.conv4 = DoubleConv(128, 64)
        self.conv5 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)
        
    def _match_tensor_size(self, x, target_tensor):
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x
        
    def forward(self, x, skip_connections):
        skips = skip_connections[::-1]
        
        x = self.up1(x)
        skip = skips[0]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att1(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        skip = skips[1]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att2(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        skip = skips[2]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att3(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        skip = skips[3]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att4(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv4(x)
        
        x = self.up5(x)
        skip = skips[4]
        skip = self._match_tensor_size(skip, x)
        if self.use_attention:
            skip = self.att5(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv5(x)
        
        x = self.final_conv(x)
        
        rgb = torch.sigmoid(x[:, :3])
        alpha = torch.sigmoid(x[:, 3:4])
        
        return torch.cat([rgb, alpha], dim=1)

class UNetAutoencoder(nn.Module):
    """U-Net Autoencoder completo para remoción de fondo."""
    def __init__(self, pretrained=True, use_attention=True):
        super(UNetAutoencoder, self).__init__()
        self.encoder = UNetEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder(use_attention=use_attention)
        
    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded

class BackgroundRemover:
    """Clase para realizar inferencia con el modelo entrenado."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = UNetAutoencoder(pretrained=False, use_attention=True)
        
        try:
            # Cargar modelo entrenado
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            self.model_loaded = False
        
    def remove_background(self, image, image_size=256):  # Reducir tamaño para CPU
        """Remueve el fondo de una imagen."""
        if not self.model_loaded:
            return None
            
        try:
            # Convertir PIL a numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Asegurar que la imagen esté en formato RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = image[:, :, :3]  # Quitar canal alpha si existe
            else:
                st.error("Formato de imagen no soportado")
                return None
                
            original_size = image_rgb.shape[:2]
            
            # Redimensionar para el modelo
            image_resized = cv2.resize(image_rgb, (image_size, image_size))
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Convertir a tensor
            input_tensor = torch.FloatTensor(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            # Optimización para CPU
            torch.set_num_threads(2)  # Limitar threads para estabilidad
            
            # Inferencia
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output.squeeze(0).cpu().numpy()
            
            # Post-procesamiento
            rgb_channels = output[:3].transpose(1, 2, 0)
            alpha_channel = output[3]
            
            # Redimensionar al tamaño original
            rgb_resized = cv2.resize(rgb_channels, (original_size[1], original_size[0]))
            alpha_resized = cv2.resize(alpha_channel, (original_size[1], original_size[0]))
            
            # Crear imagen RGBA
            result = np.zeros((original_size[0], original_size[1], 4), dtype=np.float32)
            result[:, :, :3] = rgb_resized
            result[:, :, 3] = alpha_resized
            
            # Convertir a uint8
            result = (result * 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")
            return None

@st.cache_resource
def load_model():
    """Cargar el modelo (con cache para evitar recargas)."""
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        st.error("❌ Modelo no encontrado en 'checkpoints/best_model.pth'")
        st.info("Asegúrate de que el modelo esté entrenado y guardado en la ruta correcta.")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return BackgroundRemover(model_path, device)

def main():
    """Función principal de la aplicación Streamlit."""
    
    # Header
    st.title("🎭 Removedor de Fondo con IA")
    st.markdown("**Sube una imagen y el modelo extraerá solo las personas, removiendo el fondo**")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        ### Cómo usar:
        1. Sube una imagen (JPG, PNG, JPEG)
        2. Haz clic en 'Procesar Imagen'
        3. Descarga el resultado
        
        ### Características:
        - Modelo U-Net con Attention Gates
        - Entrenado en dataset Supervisely
        - Optimizado para detectar personas
        """)
        
        # Información del sistema
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Usando: {device}")
    
    # Cargar modelo
    with st.spinner("Cargando modelo..."):
        bg_remover = load_model()
    
    if bg_remover is None:
        st.stop()
    
    st.success("✅ Modelo cargado correctamente")
    
    # Subir imagen
    st.header("📤 Subir Imagen")
    uploaded_file = st.file_uploader(
        "Elige una imagen...",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen que contenga personas para remover el fondo"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Imagen Original")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)
            
            # Información de la imagen
            st.info(f"**Dimensiones:** {original_image.size[0]} x {original_image.size[1]} píxeles")
        
        with col2:
            st.subheader("✨ Resultado")
            
            # Botón de procesamiento
            if st.button("🚀 Procesar Imagen", type="primary", use_container_width=True):
                with st.spinner("Procesando imagen... Esto puede tomar unos segundos."):
                    
                    # Procesar imagen
                    result = bg_remover.remove_background(original_image)
                    
                    if result is not None:
                        # Convertir resultado a PIL Image
                        result_pil = Image.fromarray(result, 'RGBA')
                        
                        # Mostrar resultado
                        st.image(result_pil, use_column_width=True)
                        
                        # Opción de descarga
                        buf = io.BytesIO()
                        result_pil.save(buf, format='PNG')
                        btn = st.download_button(
                            label="📥 Descargar Resultado",
                            data=buf.getvalue(),
                            file_name="imagen_sin_fondo.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        st.success("✅ ¡Procesamiento completado!")
                        
                        # Mostrar estadísticas
                        with st.expander("📊 Estadísticas del Procesamiento"):
                            alpha_channel = result[:, :, 3]
                            person_pixels = np.sum(alpha_channel > 127)
                            total_pixels = alpha_channel.size
                            coverage = (person_pixels / total_pixels) * 100
                            
                            st.metric("Cobertura de Personas", f"{coverage:.1f}%")
                            st.metric("Píxeles de Persona", f"{person_pixels:,}")
                            st.metric("Píxeles Totales", f"{total_pixels:,}")
                    else:
                        st.error("❌ Error al procesar la imagen")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Desarrollado con ❤️ usando Streamlit y PyTorch<br>
            Modelo U-Net con Attention Gates para remoción de fondo
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()