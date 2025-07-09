import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import warnings

warnings.filterwarnings('ignore')


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
        """
        Args:
            x: Feature map del skip connection
            gate: Feature map del decoder path
        """
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
    Utiliza ResNet-50 pre-entrenado como backbone para mejor extracción de características.
    """

    def __init__(self, pretrained=True):
        super(UNetEncoder, self).__init__()

        # Usar ResNet-50 pre-entrenado como backbone
        resnet = resnet50(pretrained=pretrained)

        # Extraer capas del ResNet
        self.conv1 = resnet.conv1  # 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels (ResNet-50 Bottleneck)
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # Capas adicionales para el bottleneck - actualizado para ResNet-50
        self.bottleneck = DoubleConv(2048, 4096, dropout_rate=0.2)

    def forward(self, x):
        # Encoder path con skip connections
        skip_connections = []

        # Initial convolution
        x1 = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x1)  # Skip 1: 64 channels

        x2 = self.maxpool(x1)

        # ResNet layers (ResNet-50 con Bottleneck blocks)
        x3 = self.layer1(x2)
        skip_connections.append(x3)  # Skip 2: 256 channels

        x4 = self.layer2(x3)
        skip_connections.append(x4)  # Skip 3: 512 channels

        x5 = self.layer3(x4)
        skip_connections.append(x5)  # Skip 4: 1024 channels

        x6 = self.layer4(x5)
        skip_connections.append(x6)  # Skip 5: 2048 channels

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

        # Upsampling layers - actualizado para ResNet-50
        self.up1 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)

        # Attention gates - actualizado para ResNet-50
        if self.use_attention:
            self.att1 = AttentionBlock(2048, 2048, 1024)  # layer4: 2048 channels
            self.att2 = AttentionBlock(1024, 1024, 512)   # layer3: 1024 channels
            self.att3 = AttentionBlock(512, 512, 256)     # layer2: 512 channels
            self.att4 = AttentionBlock(256, 256, 128)     # layer1: 256 channels
            self.att5 = AttentionBlock(64, 64, 32)        # conv1: 64 channels

        # Convolution blocks - actualizado para ResNet-50
        self.conv1 = DoubleConv(4096, 2048)  # 2048 (up1) + 2048 (skip[0]) = 4096
        self.conv2 = DoubleConv(2048, 1024)  # 1024 (up2) + 1024 (skip[1]) = 2048
        self.conv3 = DoubleConv(1024, 512)   # 512 (up3) + 512 (skip[2]) = 1024
        self.conv4 = DoubleConv(512, 256)    # 256 (up4) + 256 (skip[3]) = 512
        self.conv5 = DoubleConv(128, 64)     # 64 (up5) + 64 (skip[4]) = 128

        # Output layer - 4 channels (RGB + Alpha)
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)

    def _match_tensor_size(self, x, target_tensor):
        """Ajusta el tamaño de x para que coincida con target_tensor usando interpolación."""
        if x.shape[2:] != target_tensor.shape[2:]:
            x = F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x, skip_connections):
        # Decoder path - las skip connections están en orden inverso
        skips = skip_connections[::-1]  # [2048, 1024, 512, 256, 64] (ResNet-50)

        # Up 1: 4096 -> 2048
        x = self.up1(x)  # Upsample
        skip = skips[0]  # 2048 channels (layer4)

        # Asegurar que las dimensiones coincidan
        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att1(skip, x)

        x = torch.cat([x, skip], dim=1)  # 2048 + 2048 = 4096
        x = self.conv1(x)  # 4096 -> 2048

        # Up 2: 2048 -> 1024
        x = self.up2(x)
        skip = skips[1]  # 1024 channels (layer3)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att2(skip, x)

        x = torch.cat([x, skip], dim=1)  # 1024 + 1024 = 2048
        x = self.conv2(x)  # 2048 -> 1024

        # Up 3: 1024 -> 512
        x = self.up3(x)
        skip = skips[2]  # 512 channels (layer2)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att3(skip, x)

        x = torch.cat([x, skip], dim=1)  # 512 + 512 = 1024
        x = self.conv3(x)  # 1024 -> 512

        # Up 4: 512 -> 256
        x = self.up4(x)
        skip = skips[3]  # 256 channels (layer1)

        skip = self._match_tensor_size(skip, x)

        if self.use_attention:
            skip = self.att4(skip, x)

        x = torch.cat([x, skip], dim=1)  # 256 + 256 = 512
        x = self.conv4(x)  # 512 -> 256

        # Up 5: 256 -> 64 (final upsampling)
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


def test_model_forward():
    """
    Función de prueba para verificar que el modelo funciona correctamente.
    """
    print("Probando forward pass del modelo de segmentación...")

    # Crear modelo de segmentación
    model = UNetAutoencoder(pretrained=False, use_attention=True)
    model.eval()

    # Crear tensor de prueba
    test_input = torch.randn(1, 3, 384, 384)

    try:
        with torch.no_grad():
            output = model(test_input)

        print(f"✓ Forward pass de segmentación exitoso!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (1, 4, 384, 384)")

        if output.shape == (1, 4, 384, 384):
            print("✓ Dimensiones de salida correctas")
            return True
        else:
            print("✗ Dimensiones de salida incorrectas")
            return False

    except Exception as e:
        print(f"✗ Error en forward pass de segmentación: {e}")
        return False


if __name__ == "__main__":
    # Prueba del módulo
    print("=== PRUEBA DEL MÓDULO DE MODELOS ===")
    test_model_forward()