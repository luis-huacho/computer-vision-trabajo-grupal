#!/usr/bin/env python3
"""
Script de diagnóstico para el modelo de harmonización.
Analiza las dimensiones y detecta problemas en la arquitectura.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np

def test_vgg_encoder_dimensions():
    """Prueba las dimensiones del encoder VGG16."""
    print("=== DIAGNÓSTICO DEL ENCODER VGG16 ===")
    
    # Cargar VGG16
    vgg = vgg16(pretrained=False).features
    
    # Tamaños de entrada típicos
    test_sizes = [384, 256, 512]
    
    for size in test_sizes:
        print(f"\n📏 Probando con imagen de {size}x{size}:")
        test_input = torch.randn(1, 3, size, size)
        
        x = test_input
        print(f"  Input: {x.shape}")
        
        # Simular las capas como en harmonization.py
        try:
            # conv1: layers 0-3
            for i in range(4):
                x = vgg[i](x)
            print(f"  Después conv1 (0-3): {x.shape}")
            x1_pool = nn.MaxPool2d(2, 2)(x)
            print(f"  Después pool1: {x1_pool.shape}")
            
            # conv2: layers 4-8  
            x = x1_pool
            for i in range(4, 9):
                x = vgg[i](x)
            print(f"  Después conv2 (4-8): {x.shape}")
            x2_pool = nn.MaxPool2d(2, 2)(x)
            print(f"  Después pool2: {x2_pool.shape}")
            
            # conv3: layers 9-15
            x = x2_pool
            for i in range(9, 16):
                x = vgg[i](x)
            print(f"  Después conv3 (9-15): {x.shape}")
            x3_pool = nn.MaxPool2d(2, 2)(x)
            print(f"  Después pool3: {x3_pool.shape}")
            
            # conv4: layers 16-22
            x = x3_pool
            for i in range(16, 23):
                x = vgg[i](x)
            print(f"  Después conv4 (16-22): {x.shape}")
            x4_pool = nn.MaxPool2d(2, 2)(x)
            print(f"  Después pool4: {x4_pool.shape}")
            
            # conv5: layers 23-29
            x = x4_pool
            for i in range(23, 30):
                x = vgg[i](x)
            print(f"  Después conv5 (23-29): {x.shape}")
            x5_pool = nn.MaxPool2d(2, 2)(x)
            print(f"  Después pool5: {x5_pool.shape}")
            
            if x5_pool.shape[-1] <= 1:
                print(f"  ❌ PROBLEMA: Tamaño final muy pequeño: {x5_pool.shape}")
            else:
                print(f"  ✅ OK: Tamaño final aceptable: {x5_pool.shape}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

def test_harmonizer_architecture():
    """Prueba la arquitectura completa del harmonizer."""
    print("\n=== DIAGNÓSTICO DEL MODELO HARMONIZER ===")
    
    # Importar el modelo
    try:
        from harmonization import UNetHarmonizer
    except ImportError:
        print("❌ No se pudo importar UNetHarmonizer")
        return
    
    # Probar con diferentes tamaños
    test_sizes = [384, 256, 512, 128]
    
    for size in test_sizes:
        print(f"\n📏 Probando UNetHarmonizer con imagen de {size}x{size}:")
        
        try:
            # Crear modelo sin pretrained para evitar problemas de descarga
            model = UNetHarmonizer(pretrained=False, use_attention=False)
            model.eval()
            
            test_input = torch.randn(1, 3, size, size)
            print(f"  Input: {test_input.shape}")
            
            with torch.no_grad():
                output = model(test_input)
                print(f"  Output: {output.shape}")
                
            if output.shape == test_input.shape:
                print(f"  ✅ OK: Las dimensiones coinciden")
            else:
                print(f"  ❌ PROBLEMA: Las dimensiones no coinciden")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

def test_individual_components():
    """Prueba componentes individuales del modelo."""
    print("\n=== DIAGNÓSTICO DE COMPONENTES INDIVIDUALES ===")
    
    # Probar encoder básico (sin VGG)
    print("\n🔧 Probando encoder básico:")
    try:
        def make_basic_layer(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        conv1 = make_basic_layer(3, 64)
        conv2 = make_basic_layer(64, 128)
        conv3 = make_basic_layer(128, 256)
        conv4 = make_basic_layer(256, 512)
        conv5 = make_basic_layer(512, 512)
        pool = nn.MaxPool2d(2, 2)
        
        x = torch.randn(1, 3, 384, 384)
        print(f"  Input: {x.shape}")
        
        x1 = conv1(x)
        print(f"  Conv1: {x1.shape}")
        x1_pool = pool(x1)
        print(f"  Pool1: {x1_pool.shape}")
        
        x2 = conv2(x1_pool)
        print(f"  Conv2: {x2.shape}")
        x2_pool = pool(x2)
        print(f"  Pool2: {x2_pool.shape}")
        
        x3 = conv3(x2_pool)
        print(f"  Conv3: {x3.shape}")
        x3_pool = pool(x3)
        print(f"  Pool3: {x3_pool.shape}")
        
        x4 = conv4(x3_pool)
        print(f"  Conv4: {x4.shape}")
        x4_pool = pool(x4)
        print(f"  Pool4: {x4_pool.shape}")
        
        x5 = conv5(x4_pool)
        print(f"  Conv5: {x5.shape}")
        x5_pool = pool(x5)
        print(f"  Pool5: {x5_pool.shape}")
        
        if x5_pool.shape[-1] >= 1:
            print("  ✅ Encoder básico funciona correctamente")
        else:
            print("  ❌ Encoder básico también tiene problemas")
            
    except Exception as e:
        print(f"  ❌ ERROR en encoder básico: {e}")

def analyze_vgg_layers():
    """Analiza las capas VGG en detalle."""
    print("\n=== ANÁLISIS DETALLADO DE CAPAS VGG ===")
    
    vgg = vgg16(pretrained=False).features
    
    print("📋 Estructura de VGG16.features:")
    for i, layer in enumerate(vgg):
        print(f"  {i:2d}: {layer}")
    
    print(f"\n📊 Total de capas: {len(vgg)}")
    print(f"📊 Capas de convolución: {sum(1 for layer in vgg if isinstance(layer, nn.Conv2d))}")
    print(f"📊 Capas de pooling: {sum(1 for layer in vgg if isinstance(layer, nn.MaxPool2d))}")

def recommend_fixes():
    """Recomienda soluciones basadas en el diagnóstico."""
    print("\n=== RECOMENDACIONES DE CORRECCIÓN ===")
    
    print("🔧 Posibles soluciones:")
    print("  1. Usar stride=1 en lugar de stride=2 en algunos MaxPool2d")
    print("  2. Reducir el número de capas de pooling")
    print("  3. Usar padding adaptativo en las capas de convolución")
    print("  4. Cambiar a un encoder más simple sin VGG")
    print("  5. Usar imágenes de entrada más grandes (512x512 o 768x768)")
    print("  6. Implementar adaptive pooling para mantener dimensiones mínimas")

if __name__ == "__main__":
    print("🔍 DIAGNÓSTICO DEL MODELO DE HARMONIZACIÓN\n")
    
    # Ejecutar todas las pruebas
    test_vgg_encoder_dimensions()
    test_individual_components()
    analyze_vgg_layers()
    test_harmonizer_architecture()
    recommend_fixes()
    
    print("\n✅ Diagnóstico completado!")