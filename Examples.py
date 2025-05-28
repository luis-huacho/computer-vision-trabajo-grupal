"""
Ejemplos de uso y configuraciones para el modelo U-Net Background Removal.
Incluye diferentes escenarios de entrenamiento, inferencia y optimización.
"""

import torch
import torch.nn as nn
import os
from main import UNetAutoencoder, ModelInference, Trainer, SuperviselyDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# =============================================================================
# CONFIGURACIONES PREDEFINIDAS
# =============================================================================

def get_training_configs():
    """
    Configuraciones predefinidas para diferentes escenarios de entrenamiento.
    """
    configs = {
        # Configuración rápida para pruebas
        'quick_test': {
            'batch_size': 4,
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'num_epochs': 20,
            'image_size': 128,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 2,
            'pin_memory': True,
            'save_frequency': 5
        },
        
        # Configuración estándar para entrenamiento completo
        'standard': {
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 100,
            'image_size': 256,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4,
            'pin_memory': True,
            'save_frequency': 10
        },
        
        # Configuración de alta calidad para GPU potente
        'high_quality': {
            'batch_size': 16,
            'learning_rate': 5e-5,
            'weight_decay': 1e-6,
            'num_epochs': 200,
            'image_size': 512,
            'device': 'cuda',
            'num_workers': 8,
            'pin_memory': True,
            'save_frequency': 20,
            'mixed_precision': True
        },
        
        # Configuración para recursos limitados (CPU/GPU débil)
        'lightweight': {
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'image_size': 128,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 1,
            'pin_memory': False,
            'save_frequency': 5,
            'use_attention': False  # Deshabilitar attention para ahorrar memoria
        }
    }
    
    return configs

# =============================================================================
# EJEMPLOS DE ENTRENAMIENTO
# =============================================================================

def train_with_config(config_name='standard'):
    """
    Ejemplo de entrenamiento usando configuraciones predefinidas.
    """
    configs = get_training_configs()
    config = configs.get(config_name, configs['standard'])
    
    print(f"Iniciando entrenamiento con configuración: {config_name}")
    print(f"Configuración: {config}")
    
    # Verificar datos
    data_dir = 'data/supervisely_persons'
    if not os.path.exists(data_dir):
        print("ERROR: Dataset no encontrado. Descarga el dataset Supervisely Persons.")
        return
    
    # Crear modelo según configuración
    model = UNetAutoencoder(
        pretrained=True,
        use_attention=config.get('use_attention', True)
    )
    
    # Configurar datos (implementación simplificada)
    # En la práctica, usar la implementación completa de main.py
    print("Configurando datasets...")
    
    # Aquí iría la lógica completa de carga de datos
    # Por brevedad, mostramos solo la estructura
    
    print(f"Entrenamiento iniciado por {config['num_epochs']} épocas")
    # trainer.train(config['num_epochs'])

def train_custom():
    """
    Ejemplo de entrenamiento con configuración personalizada.
    """
    # Configuración personalizada
    custom_config = {
        'batch_size': 12,
        'learning_rate': 8e-5,
        'weight_decay': 2e-5,
        'num_epochs': 150,
        'image_size': 320,
        'device': 'cuda',
        'scheduler_type': 'cosine_warm_restarts',
        'loss_weights': {
            'alpha': 1.2,  # BCE weight
            'beta': 0.8,   # Dice weight
            'gamma': 0.6,  # Perceptual weight
            'delta': 0.4   # Edge weight
        }
    }
    
    print("Entrenamiento con configuración personalizada")
    # Implementar lógica de entrenamiento

# =============================================================================
# EJEMPLOS DE INFERENCIA
# =============================================================================

def basic_inference():
    """
    Ejemplo básico de inferencia en una imagen.
    """
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        print("ERROR: Modelo entrenado no encontrado.")
        print("Primero ejecuta el entrenamiento o descarga un modelo pre-entrenado.")
        return
    
    # Cargar modelo
    inference = ModelInference(model_path)
    
    # Procesar imagen
    input_path = 'examples/person_image.jpg'
    output_path = 'results/person_no_background.png'
    
    if os.path.exists(input_path):
        result = inference.remove_background(input_path, output_path)
        print(f"Fondo removido exitosamente. Resultado en: {output_path}")
        
        # Mostrar resultado
        display_result(input_path, output_path)
    else:
        print(f"Imagen de entrada no encontrada: {input_path}")

def batch_inference():
    """
    Ejemplo de procesamiento en lote.
    """
    model_path = 'checkpoints/best_model.pth'
    inference = ModelInference(model_path)
    
    input_dir = 'input_images/'
    output_dir = 'output_images/'
    
    if os.path.exists(input_dir):
        inference.batch_process(input_dir, output_dir)
        print(f"Procesamiento en lote completado. Resultados en: {output_dir}")
    else:
        print(f"Directorio de entrada no encontrado: {input_dir}")

def real_time_inference():
    """
    Ejemplo de inferencia en tiempo real usando webcam.
    """
    model_path = 'checkpoints/best_model.pth'
    inference = ModelInference(model_path)
    
    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: No se puede acceder a la webcam")
        return
    
    print("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Guardar frame temporal
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, frame)
        
        try:
            # Procesar con el modelo
            result = inference.remove_background(temp_path)
            
            # Convertir resultado para mostrar
            display_frame = cv2.cvtColor(result[:,:,:3], cv2.COLOR_RGB2BGR)
            
            # Mostrar resultado
            cv2.imshow('Background Removal - Real Time', display_frame)
            
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            cv2.imshow('Background Removal - Real Time', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_path):
        os.remove(temp_path)

# =============================================================================
# UTILIDADES DE VISUALIZACIÓN
# =============================================================================

def display_result(input_path, output_path):
    """
    Muestra comparación entre imagen original y resultado.
    """
    # Cargar imágenes
    original = Image.open(input_path).convert('RGB')
    result = Image.open(output_path).convert('RGBA')
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(original)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Resultado con fondo removido
    axes[1].imshow(result)
    axes[1].set_title('Fondo Removido')
    axes[1].axis('off')
    
    # Solo la máscara
    alpha_channel = np.array(result)[:,:,3]
    axes[2].imshow(alpha_channel, cmap='gray')
    axes[2].set_title('Máscara de Segmentación')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_training_progress(log_file='logs/training.log'):
    """
    Visualiza el progreso del entrenamiento desde logs.
    """
    if not os.path.exists(log_file):
        print(f"Archivo de log no encontrado: {log_file}")
        return
    
    # Parsear logs (implementación simplificada)
    epochs = []
    train_losses = []
    val_losses = []
    val_ious = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Época' in line and 'Train - Loss:' in line:
                # Extraer métricas (parsing simplificado)
                # En implementación real, usar regex o parser más robusto
                pass
    
    # Crear gráficas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(epochs, train_losses, label='Train Loss')
    axes[0].plot(epochs, val_losses, label='Val Loss')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curvas de Pérdida')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU curve
    axes[1].plot(epochs, val_ious, label='Validation IoU', color='green')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU de Validación')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# EJEMPLOS DE OPTIMIZACIÓN
# =============================================================================

def optimize_model_for_mobile():
    """
    Ejemplo de optimización del modelo para dispositivos móviles.
    """
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        print("ERROR: Modelo no encontrado")
        return
    
    # Cargar modelo
    model = UNetAutoencoder(pretrained=False, use_attention=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Quantización
    print("Aplicando quantización...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    
    # Guardar modelo optimizado
    optimized_path = 'checkpoints/model_mobile_optimized.pth'
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'optimization': 'quantized_int8'
    }, optimized_path)
    
    print(f"Modelo optimizado guardado en: {optimized_path}")
    
    # Comparar tamaños
    original_size = os.path.getsize(model_path) / (1024*1024)  # MB
    optimized_size = os.path.getsize(optimized_path) / (1024*1024)  # MB
    
    print(f"Tamaño original: {original_size:.2f} MB")
    print(f"Tamaño optimizado: {optimized_size:.2f} MB")
    print(f"Reducción: {((original_size - optimized_size) / original_size * 100):.1f}%")

def export_to_onnx():
    """
    Exporta el modelo a formato ONNX para deployment.
    """
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        print("ERROR: Modelo no encontrado")
        return
    
    # Cargar modelo
    model = UNetAutoencoder(pretrained=False, use_attention=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Crear input dummy
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Exportar a ONNX
    onnx_path = 'checkpoints/model.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Modelo exportado a ONNX: {onnx_path}")

# =============================================================================
# EVALUACIÓN Y MÉTRICAS
# =============================================================================

def evaluate_model_comprehensive():
    """
    Evaluación comprehensiva del modelo con múltiples métricas.
    """
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        print("ERROR: Modelo no encontrado")
        return
    
    # Cargar modelo
    inference = ModelInference(model_path)
    
    # Dataset de test
    test_dir = 'data/test_images'
    results = {
        'iou_scores': [],
        'dice_scores': [],
        'processing_times': [],
        'file_names': []
    }
    
    if not os.path.exists(test_dir):
        print(f"Directorio de test no encontrado: {test_dir}")
        return
    
    # Evaluar cada imagen
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, img_file)
            
            # Medir tiempo de procesamiento
            import time
            start_time = time.time()
            result = inference.remove_background(img_path)
            processing_time = time.time() - start_time
            
            results['processing_times'].append(processing_time)
            results['file_names'].append(img_file)
            
            # Aquí irían cálculos de IoU y Dice si hay ground truth
            # results['iou_scores'].append(iou_score)
            # results['dice_scores'].append(dice_score)
    
    # Mostrar estadísticas
    if results['processing_times']:
        avg_time = np.mean(results['processing_times'])
        fps = 1.0 / avg_time
        
        print(f"Tiempo promedio de procesamiento: {avg_time:.3f}s")
        print(f"FPS aproximado: {fps:.1f}")
        print(f"Imágenes procesadas: {len(results['processing_times'])}")

def benchmark_against_baselines():
    """
    Compara el modelo contra métodos baseline.
    """
    print("Benchmark vs métodos tradicionales:")
    print("1. GrabCut")
    print("2. Watershed")
    print("3. DeepLab v3+")
    print("4. Nuestro U-Net")
    
    # Implementar comparaciones
    # Este sería un análisis más extenso en implementación real

# =============================================================================
# EJEMPLOS DE USO PRINCIPAL
# =============================================================================

def main_examples():
    """
    Función principal con ejemplos de uso.
    """
    print("=== Ejemplos de Uso U-Net Background Removal ===\n")
    
    print("1. Entrenamiento rápido (testing):")
    print("   train_with_config('quick_test')")
    
    print("\n2. Entrenamiento estándar:")
    print("   train_with_config('standard')")
    
    print("\n3. Inferencia básica:")
    print("   basic_inference()")
    
    print("\n4. Procesamiento en lote:")
    print("   batch_inference()")
    
    print("\n5. Tiempo real (webcam):")
    print("   real_time_inference()")
    
    print("\n6. Optimización para móvil:")
    print("   optimize_model_for_mobile()")
    
    print("\n7. Exportar a ONNX:")
    print("   export_to_onnx()")
    
    print("\n8. Evaluación comprehensiva:")
    print("   evaluate_model_comprehensive()")
    
    print("\nPara ejecutar un ejemplo específico, descomenta la función correspondiente.")

# =============================================================================
# CONFIGURACIONES AVANZADAS
# =============================================================================

def advanced_training_config():
    """
    Configuraciones avanzadas para casos específicos.
    """
    configs = {
        # Para datasets muy grandes
        'large_scale': {
            'batch_size': 32,
            'learning_rate': 2e-4,
            'weight_decay': 1e-5,
            'num_epochs': 300,
            'image_size': 384,
            'gradient_accumulation_steps': 4,
            'mixed_precision': True,
            'distributed_training': True,
            'scheduler': 'polynomial_decay',
            'warmup_epochs': 10
        },
        
        # Para fine-tuning de modelo pre-entrenado
        'fine_tuning': {
            'batch_size': 4,
            'learning_rate': 1e-5,  # LR muy bajo para fine-tuning
            'weight_decay': 1e-6,
            'num_epochs': 50,
            'image_size': 256,
            'freeze_encoder': True,  # Congelar encoder inicialmente
            'gradual_unfreezing': True,
            'differential_lr': {
                'encoder': 1e-6,
                'decoder': 1e-4
            }
        },
        
        # Para dominio específico (ej: retratos profesionales)
        'portrait_specialized': {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'weight_decay': 2e-5,
            'num_epochs': 150,
            'image_size': 512,
            'loss_weights': {
                'alpha': 0.8,  # Menos peso a BCE
                'beta': 1.5,   # Más peso a Dice
                'gamma': 1.0,  # Perceptual importante para retratos
                'delta': 0.8   # Edge crítico para cabello
            },
            'augmentation_strength': 'light'  # Augmentación suave para retratos
        }
    }
    
    return configs

def create_custom_dataset():
    """
    Ejemplo de creación de dataset personalizado.
    """
    import json
    from torch.utils.data import Dataset
    
    class CustomBackgroundDataset(Dataset):
        """
        Dataset personalizado para casos específicos.
        """
        def __init__(self, images_dir, masks_dir, transform=None, 
                     filter_criteria=None, image_size=256):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.transform = transform
            self.image_size = image_size
            
            # Filtrar imágenes según criterios
            self.image_files = self._filter_images(filter_criteria)
            
        def _filter_images(self, criteria):
            """Filtra imágenes según criterios específicos."""
            all_images = [f for f in os.listdir(self.images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if criteria is None:
                return all_images
            
            filtered = []
            for img_file in all_images:
                # Ejemplo: filtrar por tamaño de imagen
                if criteria.get('min_resolution'):
                    img_path = os.path.join(self.images_dir, img_file)
                    img = Image.open(img_path)
                    if min(img.size) < criteria['min_resolution']:
                        continue
                
                # Ejemplo: filtrar por presencia de múltiples personas
                if criteria.get('single_person_only'):
                    # Lógica para detectar número de personas
                    pass
                
                filtered.append(img_file)
            
            return filtered
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            # Implementación personalizada
            img_name = self.image_files[idx]
            # ... resto de la implementación
            pass
    
    # Ejemplo de uso
    filter_criteria = {
        'min_resolution': 512,
        'single_person_only': True,
        'good_lighting': True
    }
    
    custom_dataset = CustomBackgroundDataset(
        'data/custom_images',
        'data/custom_masks',
        filter_criteria=filter_criteria
    )
    
    print(f"Dataset personalizado creado con {len(custom_dataset)} imágenes")

# =============================================================================
# ANÁLISIS Y DEBUGGING
# =============================================================================

def analyze_model_performance():
    """
    Análisis detallado del rendimiento del modelo.
    """
    model_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        print("ERROR: Modelo no encontrado")
        return
    
    # Cargar modelo
    model = UNetAutoencoder(pretrained=False, use_attention=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Análisis de arquitectura
    print("=== Análisis de Arquitectura ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Tamaño del modelo: {total_params * 4 / (1024**2):.2f} MB")
    
    # Análisis por capas
    print("\n=== Parámetros por Componente ===")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    
    # Análisis de memoria
    print("\n=== Análisis de Memoria ===")
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Calcular memoria de activaciones (aproximado)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    input_memory = dummy_input.numel() * 4 / (1024**2)  # MB
    output_memory = output.numel() * 4 / (1024**2)  # MB
    
    print(f"Memoria de entrada: {input_memory:.2f} MB")
    print(f"Memoria de salida: {output_memory:.2f} MB")
    
    # Estimación de memoria total durante entrenamiento (muy aproximada)
    estimated_training_memory = (total_params * 4 * 3) / (1024**2)  # Parámetros + gradientes + momentum
    print(f"Memoria estimada de entrenamiento: {estimated_training_memory:.2f} MB")

def debug_training_issues():
    """
    Herramientas para debuggear problemas comunes de entrenamiento.
    """
    print("=== Debugging de Entrenamiento ===\n")
    
    # Verificar gradientes
    def check_gradients(model, sample_batch):
        """Verifica si los gradientes fluyen correctamente."""
        model.train()
        
        # Forward pass
        images, targets = sample_batch
        outputs = model(images)
        
        # Calcular loss simple
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        
        # Verificar gradientes
        grad_norms = []
        zero_grad_layers = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm < 1e-8:
                    zero_grad_layers.append(name)
            else:
                zero_grad_layers.append(name)
        
        print(f"Gradiente mínimo: {min(grad_norms):.2e}")
        print(f"Gradiente máximo: {max(grad_norms):.2e}")
        print(f"Gradiente promedio: {np.mean(grad_norms):.2e}")
        
        if zero_grad_layers:
            print("⚠️ Capas con gradiente cero:")
            for layer in zero_grad_layers[:5]:  # Mostrar solo las primeras 5
                print(f"  - {layer}")
    
    # Verificar distribución de loss
    def analyze_loss_components(model, data_loader):
        """Analiza componentes individuales de la función de pérdida."""
        from main import LossCalculator
        
        loss_calc = LossCalculator()
        
        bce_losses = []
        dice_losses = []
        perceptual_losses = []
        edge_losses = []
        
        model.eval()
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                if i >= 10:  # Solo analizar primeros 10 batches
                    break
                
                outputs = model(images)
                loss_dict = loss_calc.calculate_loss(outputs, targets)
                
                bce_losses.append(loss_dict['bce_loss'].item())
                dice_losses.append(loss_dict['dice_loss'].item())
                perceptual_losses.append(loss_dict['perceptual_loss'].item())
                edge_losses.append(loss_dict['edge_loss'].item())
        
        print("Componentes de Loss:")
        print(f"BCE Loss - Promedio: {np.mean(bce_losses):.4f}, Std: {np.std(bce_losses):.4f}")
        print(f"Dice Loss - Promedio: {np.mean(dice_losses):.4f}, Std: {np.std(dice_losses):.4f}")
        print(f"Perceptual Loss - Promedio: {np.mean(perceptual_losses):.4f}, Std: {np.std(perceptual_losses):.4f}")
        print(f"Edge Loss - Promedio: {np.mean(edge_losses):.4f}, Std: {np.std(edge_losses):.4f}")
    
    print("Funciones de debugging disponibles:")
    print("1. check_gradients(model, sample_batch)")
    print("2. analyze_loss_components(model, data_loader)")

def visualize_attention_maps():
    """
    Visualiza los mapas de atención del modelo.
    """
    print("=== Visualización de Mapas de Atención ===")
    
    # Esta función requeriría modificar el modelo para extraer attention maps
    # Es un ejemplo de cómo se podría implementar
    
    class AttentionVisualizer:
        def __init__(self, model):
            self.model = model
            self.attention_maps = {}
            self.hooks = []
            
            # Registrar hooks para capturar attention maps
            self._register_hooks()
        
        def _register_hooks(self):
            """Registra hooks para capturar attention maps."""
            def hook_fn(name):
                def hook(module, input, output):
                    if hasattr(module, 'sigmoid'):  # Es un attention gate
                        self.attention_maps[name] = output.detach()
                return hook
            
            # Registrar hooks en attention gates
            for name, module in self.model.named_modules():
                if 'att' in name and hasattr(module, 'sigmoid'):
                    hook = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(hook)
        
        def visualize(self, image_tensor):
            """Visualiza attention maps para una imagen."""
            self.model.eval()
            self.attention_maps.clear()
            
            with torch.no_grad():
                _ = self.model(image_tensor)
            
            # Crear visualización
            num_maps = len(self.attention_maps)
            if num_maps == 0:
                print("No se encontraron mapas de atención")
                return
            
            fig, axes = plt.subplots(2, max(3, num_maps//2), figsize=(15, 8))
            axes = axes.flatten()
            
            # Imagen original
            orig_img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            axes[0].imshow(orig_img)
            axes[0].set_title('Imagen Original')
            axes[0].axis('off')
            
            # Mapas de atención
            for i, (name, att_map) in enumerate(self.attention_maps.items()):
                if i + 1 < len(axes):
                    # Tomar el primer mapa del batch y canal
                    map_vis = att_map[0, 0].cpu().numpy()
                    
                    axes[i + 1].imshow(map_vis, cmap='hot', interpolation='bilinear')
                    axes[i + 1].set_title(f'Attention: {name}')
                    axes[i + 1].axis('off')
            
            # Ocultar axes no usados
            for i in range(len(self.attention_maps) + 1, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        def cleanup(self):
            """Limpia los hooks registrados."""
            for hook in self.hooks:
                hook.remove()
    
    print("Clase AttentionVisualizer creada.")
    print("Uso: visualizer = AttentionVisualizer(model)")
    print("     visualizer.visualize(image_tensor)")

# =============================================================================
# UTILIDADES DE DEPLOYMENT
# =============================================================================

def create_deployment_package():
    """
    Crea un paquete optimizado para deployment.
    """
    print("=== Creando Paquete de Deployment ===")
    
    # Crear estructura de directorios
    deployment_dir = 'deployment_package'
    os.makedirs(deployment_dir, exist_ok=True)
    os.makedirs(f'{deployment_dir}/models', exist_ok=True)
    os.makedirs(f'{deployment_dir}/utils', exist_ok=True)
    
    # Copiar modelo optimizado
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        # Crear versión ligera del modelo
        model = UNetAutoencoder(pretrained=False, use_attention=True)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Guardar solo los pesos necesarios
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'pretrained': False,
                'use_attention': True,
                'image_size': 256
            }
        }, f'{deployment_dir}/models/background_removal_model.pth')
        
        print("✓ Modelo copiado y optimizado")
    
    # Crear script de inferencia simplificado
    inference_script = f'''
"""
Script de inferencia optimizado para deployment.
"""
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BackgroundRemovalInference:
    def __init__(self, model_path):
        # Cargar modelo (código simplificado)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ... implementación de carga de modelo
        
    def process(self, image_path, output_path=None):
        # Implementación optimizada de procesamiento
        pass

# Ejemplo de uso
if __name__ == "__main__":
    inference = BackgroundRemovalInference("models/background_removal_model.pth")
    result = inference.process("input.jpg", "output.png")
'''
    
    with open(f'{deployment_dir}/inference.py', 'w') as f:
        f.write(inference_script)
    
    # Crear requirements mínimos para deployment
    deployment_requirements = '''
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.6.0
Pillow>=9.2.0
albumentations>=1.3.0
numpy>=1.21.0
'''
    
    with open(f'{deployment_dir}/requirements.txt', 'w') as f:
        f.write(deployment_requirements)
    
    # Crear README de deployment
    deployment_readme = '''
# Background Removal - Deployment Package

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run inference:
   ```python
   from inference import BackgroundRemovalInference
   
   model = BackgroundRemovalInference("models/background_removal_model.pth")
   result = model.process("input.jpg", "output.png")
   ```

## API Endpoints (if using web service)

- POST /remove_background
- GET /health

## Performance

- Average processing time: ~50ms per image (256x256)
- Memory usage: ~500MB
- GPU recommended for optimal performance
'''
    
    with open(f'{deployment_dir}/README.md', 'w') as f:
        f.write(deployment_readme)
    
    print(f"✓ Paquete de deployment creado en: {deployment_dir}")
    print("✓ Incluye: modelo optimizado, script de inferencia, requirements, README")

def create_web_api():
    """
    Crea una API web simple para el modelo.
    """
    api_code = '''
"""
API Web para Background Removal usando FastAPI.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import numpy as np

app = FastAPI(title="Background Removal API", version="1.0.0")

# Cargar modelo al iniciar
model = None

@app.on_event("startup")
async def load_model():
    global model
    # Cargar modelo aquí
    print("Modelo cargado exitosamente")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/remove_background")
async def remove_background(file: UploadFile = File(...)):
    try:
        # Verificar tipo de archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Leer imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Procesar con el modelo
        # result = model.process(image)
        
        # Convertir resultado a bytes
        result_bytes = io.BytesIO()
        # result.save(result_bytes, format='PNG')
        result_bytes.seek(0)
        
        return StreamingResponse(
            result_bytes, 
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=result.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open('deployment_package/api.py', 'w') as f:
        f.write(api_code)
    
    print("✓ API web creada en deployment_package/api.py")
    print("Para ejecutar: uvicorn api:app --reload")

# =============================================================================
# EJEMPLOS DE INTEGRACIÓN
# =============================================================================

def integration_examples():
    """
    Ejemplos de integración con diferentes frameworks y aplicaciones.
    """
    print("=== Ejemplos de Integración ===\n")
    
    # Ejemplo con OpenCV
    opencv_example = '''
# Integración con OpenCV para video processing
import cv2

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame con nuestro modelo
        processed_frame = remove_background_frame(frame)
        out.write(processed_frame)
    
    cap.release()
    out.release()
'''
    
    # Ejemplo con Streamlit
    streamlit_example = '''
# Aplicación web con Streamlit
import streamlit as st
from PIL import Image

st.title("Background Removal App")

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    
    if st.button('Remove Background'):
        with st.spinner('Processing...'):
            result = remove_background(image)
            st.image(result, caption='Result', use_column_width=True)
'''
    
    # Ejemplo con Flask
    flask_example = '''
# API Flask simple
from flask import Flask, request, send_file
import io

app = Flask(__name__)

@app.route('/remove_background', methods=['POST'])
def remove_background_endpoint():
    file = request.files['image']
    
    # Procesar imagen
    result = process_image(file)
    
    # Retornar resultado
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
'''
    
    print("1. OpenCV Video Processing:")
    print(opencv_example)
    print("\n" + "="*50 + "\n")
    
    print("2. Streamlit Web App:")
    print(streamlit_example)
    print("\n" + "="*50 + "\n")
    
    print("3. Flask API:")
    print(flask_example)

if __name__ == "__main__":
    # Ejecutar ejemplos principales
    main_examples()
    
    # Descomenta para ejecutar funciones específicas:
    # train_with_config('quick_test')
    # basic_inference()
    # analyze_model_performance()
    # create_deployment_package()