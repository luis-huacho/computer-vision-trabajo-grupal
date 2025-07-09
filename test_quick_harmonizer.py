#!/usr/bin/env python3
"""
Test rápido del modelo de harmonización corregido.
"""

import torch
import torch.nn as nn

def test_harmonizer_quick():
    """Prueba rápida del modelo corregido."""
    print("=== PRUEBA RÁPIDA DEL HARMONIZER CORREGIDO ===")
    
    try:
        from harmonization import UNetHarmonizer
        
        # Crear modelo
        model = UNetHarmonizer(pretrained=False, use_attention=False)
        model.eval()
        
        # Probar con imagen 384x384
        test_input = torch.randn(1, 3, 384, 384)
        print(f"Input: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            print(f"Output: {output.shape}")
            
        if output.shape == test_input.shape:
            print("✅ SUCCESS: El modelo funciona correctamente!")
            return True
        else:
            print(f"❌ ERROR: Dimensiones incorrectas")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_harmonizer_quick()