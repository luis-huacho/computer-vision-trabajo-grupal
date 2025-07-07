import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import torch
from transformers import pipeline
import colorsys

class SmartAutoCompositor:
    def __init__(self, model_type="fast"):
        """
        Initialize the smart compositor
        model_type: 'fast' (MiDaS), 'quality' (ZoeDepth), or 'best' (Depth Anything V2)
        """
        print(f"🔧 Loading {model_type} depth estimation model...")
        
        if model_type == "fast":
            self.depth_estimator = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            self.depth_estimator.eval()
            self.estimate_method = self._estimate_depth_midas
        elif model_type == "quality":
            self.depth_estimator = pipeline("depth-estimation", model="Intel/zoedepth-nyu-kitti")
            self.estimate_method = self._estimate_depth_zoedepth
        else:  # best
            self.depth_estimator = pipeline("depth-estimation", model="LiheYoung/depth-anything-large-hf")
            self.estimate_method = self._estimate_depth_pipeline
            
        print("✅ Model loaded successfully!")
    
    def _numpy_to_pil(self, img_array):
        """Convert numpy array to PIL Image"""
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _pil_to_numpy(self, pil_img):
        """Convert PIL Image to numpy array"""
        return np.array(pil_img)
    
    def _estimate_depth_midas(self, image_array):
        """Fast depth estimation with MiDaS from numpy array"""
        # Convert RGB to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        input_batch = self.transform(img_rgb)
        
        with torch.no_grad():
            prediction = self.depth_estimator(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        return (depth - depth.min()) / (depth.max() - depth.min())
    
    def _estimate_depth_pipeline(self, image_array):
        """High-quality depth estimation with pipeline models from numpy array"""
        pil_image = self._numpy_to_pil(image_array)
        result = self.depth_estimator(pil_image)
        depth_map = np.array(result["depth"])
        return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    def _estimate_depth_zoedepth(self, image_array):
        """Metric depth estimation with ZoeDepth from numpy array"""
        return self._estimate_depth_pipeline(image_array)
    
    def analyze_background_lighting(self, bg_array):
        """Analyze lighting conditions in background from numpy array"""
        # Ensure correct data type
        if bg_array.dtype != np.uint8:
            bg_array = np.clip(bg_array, 0, 255).astype(np.uint8)
        
        # Analyze brightness distribution
        brightness = np.mean(cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY))
        
        # Analyze color temperature (warm vs cool)
        avg_red = np.mean(bg_array[:, :, 0])
        avg_blue = np.mean(bg_array[:, :, 2])
        color_temp = "warm" if avg_red > avg_blue else "cool"
        
        # Detect dominant light direction
        gray = cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients to find light direction
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        avg_grad_x = np.mean(grad_x)
        avg_grad_y = np.mean(grad_y)
        
        # Determine light direction with more nuanced detection
        if abs(avg_grad_x) > abs(avg_grad_y):
            if avg_grad_x > 0:
                light_direction = "left"
            else:
                light_direction = "right"
        else:
            if avg_grad_y > 0:
                light_direction = "top"
            else:
                light_direction = "bottom"
        
        # For mixed lighting, default to top-left (most common)
        if abs(avg_grad_x) > 0 and abs(avg_grad_y) > 0:
            if avg_grad_x > 0 and avg_grad_y > 0:
                light_direction = "top-left"
            elif avg_grad_x < 0 and avg_grad_y > 0:
                light_direction = "top-right"
        
        # Analyze contrast (helps determine shadow intensity)
        contrast = np.std(gray)
        
        return {
            'brightness': brightness / 255.0,
            'color_temp': color_temp,
            'light_direction': light_direction,
            'contrast': contrast / 255.0,
            'avg_red': avg_red,
            'avg_blue': avg_blue
        }
    
    def adjust_foreground_lighting(self, fg_array, bg_lighting):
        """Adjust foreground lighting to match background from numpy array"""
        # Convert to PIL for enhancement operations
        fg_image = self._numpy_to_pil(fg_array)
        
        # Brightness adjustment
        brightness_enhancer = ImageEnhance.Brightness(fg_image)
        target_brightness = 0.8 + (bg_lighting['brightness'] * 0.4)
        fg_adjusted = brightness_enhancer.enhance(target_brightness)
        
        # Contrast adjustment
        contrast_enhancer = ImageEnhance.Contrast(fg_adjusted)
        target_contrast = 0.9 + (bg_lighting['contrast'] * 0.2)
        fg_adjusted = contrast_enhancer.enhance(target_contrast)
        
        # Color temperature adjustment
        fg_adjusted_array = np.array(fg_adjusted).astype(np.float32)
        
        if bg_lighting['color_temp'] == 'warm':
            # Add warmth (more red/yellow)
            warmth_factor = 1.1
            fg_adjusted_array[:, :, 0] = np.clip(fg_adjusted_array[:, :, 0] * warmth_factor, 0, 255)
            fg_adjusted_array[:, :, 1] = np.clip(fg_adjusted_array[:, :, 1] * (warmth_factor * 0.95), 0, 255)
        else:
            # Add coolness (more blue)
            coolness_factor = 1.1
            fg_adjusted_array[:, :, 2] = np.clip(fg_adjusted_array[:, :, 2] * coolness_factor, 0, 255)
            fg_adjusted_array[:, :, 0] = np.clip(fg_adjusted_array[:, :, 0] * 0.95, 0, 255)
        
        return fg_adjusted_array.astype(np.uint8)
    
    def apply_directional_lighting(self, fg_array, fg_depth, light_direction, intensity=0.3):
        """Apply directional lighting based on background analysis"""
        fg_work = fg_array.astype(np.float32)
        h, w = fg_depth.shape
        
        # Create directional light gradient
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        if light_direction == "top":
            light_gradient = (h - y_coords) / h
        elif light_direction == "bottom":
            light_gradient = y_coords / h
        elif light_direction == "left":
            light_gradient = (w - x_coords) / w
        elif light_direction == "right":
            light_gradient = x_coords / w
        elif light_direction == "top-left":
            light_gradient = ((h - y_coords) / h + (w - x_coords) / w) / 2
        elif light_direction == "top-right":
            light_gradient = ((h - y_coords) / h + x_coords / w) / 2
        else:  # default to top
            light_gradient = (h - y_coords) / h
        
        # Combine with depth information
        lighting_effect = fg_depth * light_gradient * intensity + (1 - intensity)
        
        # Resize lighting effect to match image dimensions
        if fg_work.shape[:2] != lighting_effect.shape:
            lighting_effect = cv2.resize(lighting_effect, (fg_work.shape[1], fg_work.shape[0]))
        
        # Apply lighting to each channel
        for channel in range(min(3, fg_work.shape[2])):
            fg_work[:, :, channel] *= lighting_effect
        
        return np.clip(fg_work, 0, 255).astype(np.uint8)
    
    def _calculate_shadow_projection(self, light_direction, bg_h, bg_w):
        """Calculate shadow offset based on light direction"""
        # Shadow projection factors (how far shadows extend)
        projection_factor = 0.12  # Reduced for more natural shadows
        
        if light_direction == "top":
            # Light from top = shadows project downward
            return 0, int(bg_h * projection_factor)
        elif light_direction == "top-left":
            # Light from top-left = shadows project down-right
            return int(bg_w * projection_factor * 0.4), int(bg_h * projection_factor)
        elif light_direction == "top-right":
            # Light from top-right = shadows project down-left  
            return -int(bg_w * projection_factor * 0.4), int(bg_h * projection_factor)
        elif light_direction == "left":
            # Light from left = shadows project right
            return int(bg_w * projection_factor * 0.8), int(bg_h * projection_factor * 0.3)
        elif light_direction == "right":
            # Light from right = shadows project left
            return -int(bg_w * projection_factor * 0.8), int(bg_h * projection_factor * 0.3)
        else:
            # Default: top lighting
            return 0, int(bg_h * projection_factor)
    
    def _project_shadow_on_ground(self, fg_mask, offset_x, offset_y, bg_h, bg_w):
        """Project shadow mask onto ground with perspective"""
        shadow_projection = np.zeros((bg_h, bg_w), dtype=np.float32)
        
        # Get foreground object positions
        y_coords, x_coords = np.where(fg_mask)
        
        if len(y_coords) == 0:
            return shadow_projection
        
        # Project each foreground pixel onto ground
        for y, x in zip(y_coords, x_coords):
            # Calculate projected position
            shadow_y = min(bg_h - 1, y + offset_y)
            shadow_x = max(0, min(bg_w - 1, x + offset_x))
            
            # Apply perspective scaling (objects lower in image = larger shadows)
            perspective_scale = 0.3 + 0.4 * (y / bg_h)  # Bottom = larger shadows
            shadow_intensity = perspective_scale * 0.6  # Reduced intensity
            
            # Add shadow at projected position with falloff
            if 0 <= shadow_y < bg_h and 0 <= shadow_x < bg_w:
                # Add soft falloff around shadow point
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        sy, sx = int(shadow_y) + dy, int(shadow_x) + dx
                        if 0 <= sy < bg_h and 0 <= sx < bg_w:
                            distance = np.sqrt(dy*dy + dx*dx)
                            falloff = max(0, 1 - distance / 3)
                            shadow_projection[sy, sx] = max(shadow_projection[sy, sx], 
                                                           shadow_intensity * falloff)
        
        return shadow_projection
    
    def _create_ground_mask(self, bg_h, bg_w):
        """Create mask for ground area (bottom portion of image)"""
        ground_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
        
        # Ground is bottom 60% of image with gradient falloff
        ground_start = int(bg_h * 0.4)
        
        for y in range(ground_start, bg_h):
            # Gradient from 0 at top to 1 at bottom
            intensity = (y - ground_start) / (bg_h - ground_start)
            # Smooth gradient
            intensity = intensity * intensity  # Quadratic falloff
            ground_mask[y, :] = intensity
        
        return ground_mask
    
    def generate_realistic_shadows(self, bg_array, fg_depth, light_direction, shadow_intensity=0.2):
        """Generate realistic shadows projected on ground based on foreground depth"""
        bg_work = bg_array.astype(np.float32)
        bg_h, bg_w = bg_work.shape[:2]
        
        # Resize depth map to match background
        fg_depth_resized = cv2.resize(fg_depth, (bg_w, bg_h))
        
        # Create base shadow map
        shadow_map = np.ones((bg_h, bg_w))
        
        # Create foreground mask (areas with people/objects)
        fg_mask = fg_depth_resized > 0.3  # More selective threshold
        
        # Calculate shadow projection based on light direction
        shadow_offset_x, shadow_offset_y = self._calculate_shadow_projection(light_direction, bg_h, bg_w)
        
        # Project shadows onto ground
        projected_shadow = self._project_shadow_on_ground(fg_mask, shadow_offset_x, shadow_offset_y, bg_h, bg_w)
        
        # Create soft shadow with gaussian blur
        shadow_blur_size = max(15, min(bg_h, bg_w) // 30)  # Adaptive blur size
        soft_shadow = cv2.GaussianBlur(projected_shadow.astype(np.float32), 
                                       (shadow_blur_size, shadow_blur_size), 
                                       shadow_blur_size // 3)
        
        # Apply shadow intensity (reduced for more natural look)
        shadow_strength = np.where(soft_shadow > 0.05, 
                                   shadow_intensity * soft_shadow,
                                   0)
        
        # Create final shadow map
        shadow_map = 1 - shadow_strength
        
        # Apply shadows only to ground area
        ground_mask = self._create_ground_mask(bg_h, bg_w)
        shadow_map = shadow_map * ground_mask + (1 - ground_mask)
        
        # Apply shadows to background
        for channel in range(min(3, bg_work.shape[2])):
            bg_work[:, :, channel] *= shadow_map
        
        return np.clip(bg_work, 0, 255).astype(np.uint8)
    
    def calculate_optimal_scale_and_position(self, fg_shape, bg_shape, fg_depth, bg_depth):
        """Calculate optimal scale and position based on depth analysis"""
        bg_h, bg_w = bg_shape[:2]
        fg_h, fg_w = fg_shape[:2]
        
        # Analyze depth to determine apparent distance
        avg_fg_depth = np.mean(fg_depth)
        avg_bg_depth = np.mean(bg_depth)
        
        # Calculate scale based on relative depth
        depth_ratio = avg_fg_depth / (avg_bg_depth + 0.001)
        base_scale = 0.4
        
        # Adjust scale based on depth relationship
        if depth_ratio > 1.2:  # Foreground much closer
            scale_factor = base_scale + 0.25
        elif depth_ratio > 0.8:  # Similar depth
            scale_factor = base_scale + 0.05
        else:  # Foreground further
            scale_factor = base_scale - 0.05
        
        # Ensure reasonable bounds
        scale_factor = np.clip(scale_factor, 0.25, 0.7)
        
        # Calculate new dimensions
        new_fg_w = int(fg_w * scale_factor)
        new_fg_h = int(fg_h * scale_factor)
        
        # Calculate position (center horizontally, ground-aligned vertically)
        pos_x = (bg_w - new_fg_w) // 2
        pos_y = bg_h - new_fg_h - int(bg_h * 0.03)  # Small margin from bottom
        
        return (new_fg_w, new_fg_h), (pos_x, pos_y), scale_factor
    
    def apply_depth_of_field(self, fg_array, fg_depth, focus_distance=0.6, blur_intensity=2):
        """Apply depth of field blur to foreground"""
        # Convert to PIL for blur operations
        fg_image = self._numpy_to_pil(fg_array)
        
        # Calculate blur map
        blur_map = np.abs(fg_depth - focus_distance)
        blur_map = blur_map / blur_map.max() * blur_intensity
        
        # Create blurred version
        blurred_fg = fg_image.filter(ImageFilter.GaussianBlur(blur_intensity))
        
        # Blend based on depth
        fg_orig = np.array(fg_image)
        blurred_array = np.array(blurred_fg)
        result_array = fg_orig.copy()
        
        # Resize blur map to match image
        if blur_map.shape != fg_orig.shape[:2]:
            blur_map = cv2.resize(blur_map, (fg_orig.shape[1], fg_orig.shape[0]))
        
        # Apply variable blur (simplified for performance)
        avg_blur = np.mean(blur_map)
        if avg_blur > 0.5:
            # Apply moderate blur to entire image
            result_array = (0.7 * fg_orig + 0.3 * blurred_array).astype(np.uint8)
        
        return result_array.astype(np.uint8)
    
    def auto_composite(self, foreground_array, background_array, 
                      apply_dof=True, shadow_intensity=0.15, return_debug=False):
        """
        Main function: Automatically composite foreground and background numpy arrays
        
        Args:
            foreground_array: numpy array RGB/RGBA (H, W, 3/4) - foreground image
            background_array: numpy array RGB (H, W, 3) - background image  
            apply_dof: bool - apply depth of field blur
            shadow_intensity: float 0-1 - shadow strength (reduced default)
            return_debug: bool - return debug information
        
        Returns:
            result_array: numpy array RGB (H, W, 3) - final composition
            info: dict - composition information (if return_debug=True)
        """
        print("🚀 Starting smart auto-composition...")
        
        # Ensure input arrays are uint8
        if foreground_array.dtype != np.uint8:
            foreground_array = np.clip(foreground_array, 0, 255).astype(np.uint8)
        if background_array.dtype != np.uint8:
            background_array = np.clip(background_array, 0, 255).astype(np.uint8)
        
        # Handle RGBA foreground (extract alpha channel)
        has_alpha = False
        alpha_mask = None
        if len(foreground_array.shape) == 3 and foreground_array.shape[2] == 4:
            print("🔍 Detected RGBA foreground - extracting alpha channel...")
            alpha_mask = foreground_array[:, :, 3]
            foreground_array = foreground_array[:, :, :3]  # Keep only RGB
            has_alpha = True
        
        # Ensure background is RGB
        if len(background_array.shape) == 3 and background_array.shape[2] == 4:
            print("🔍 Converting RGBA background to RGB...")
            background_array = background_array[:, :, :3]
        
        # Estimate depth maps
        print("📏 Analyzing depth information...")
        fg_depth = self.estimate_method(foreground_array)
        bg_depth = self.estimate_method(background_array)
        
        # Analyze background lighting
        print("💡 Analyzing background lighting conditions...")
        bg_lighting = self.analyze_background_lighting(background_array)
        print(f"   └─ Detected: {bg_lighting['brightness']:.2f} brightness, {bg_lighting['color_temp']} temperature, {bg_lighting['light_direction']} lighting")
        
        # Adjust foreground lighting to match background
        print("🎨 Matching foreground lighting to background...")
        fg_lit = self.adjust_foreground_lighting(foreground_array, bg_lighting)
        
        # Apply directional lighting
        print("🔦 Applying directional lighting...")
        fg_directional = self.apply_directional_lighting(
            fg_lit, fg_depth, bg_lighting['light_direction']
        )
        
        # Apply depth of field if requested
        if apply_dof:
            print("🌫️ Applying depth of field...")
            fg_final = self.apply_depth_of_field(fg_directional, fg_depth)
        else:
            fg_final = fg_directional
        
        # Generate shadows on background
        print("👥 Generating realistic ground shadows...")
        bg_shadowed = self.generate_realistic_shadows(
            background_array, fg_depth, bg_lighting['light_direction'], shadow_intensity
        )
        
        # Calculate optimal scale and position
        print("📐 Calculating optimal positioning...")
        new_size, position, scale_factor = self.calculate_optimal_scale_and_position(
            foreground_array.shape, background_array.shape, fg_depth, bg_depth
        )
        
        # Scale foreground
        print("🔄 Final composition...")
        fg_scaled = cv2.resize(fg_final, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Scale alpha mask if present
        if has_alpha:
            alpha_scaled = cv2.resize(alpha_mask, new_size, interpolation=cv2.INTER_LANCZOS4)
            alpha_normalized = alpha_scaled.astype(np.float32) / 255.0
        
        # Final composite
        result = bg_shadowed.copy()
        pos_x, pos_y = position
        
        # Ensure we don't go out of bounds
        end_y = min(pos_y + fg_scaled.shape[0], result.shape[0])
        end_x = min(pos_x + fg_scaled.shape[1], result.shape[1])
        fg_h = end_y - pos_y
        fg_w = end_x - pos_x
        
        if fg_h > 0 and fg_w > 0:
            if has_alpha:
                # Alpha blending using transparency mask
                print("🎭 Applying alpha blending...")
                alpha_crop = alpha_normalized[:fg_h, :fg_w]
                fg_crop = fg_scaled[:fg_h, :fg_w].astype(np.float32)
                bg_crop = result[pos_y:end_y, pos_x:end_x].astype(np.float32)
                
                # Blend: result = fg * alpha + bg * (1 - alpha)
                for c in range(3):
                    result[pos_y:end_y, pos_x:end_x, c] = (
                        fg_crop[:, :, c] * alpha_crop + 
                        bg_crop[:, :, c] * (1 - alpha_crop)
                    ).astype(np.uint8)
            else:
                # Simple replacement (assuming foreground has no background)
                result[pos_y:end_y, pos_x:end_x] = fg_scaled[:fg_h, :fg_w]
        
        print("✅ Smart auto-composition complete!")
        print(f"   └─ Applied: lighting matching, {bg_lighting['light_direction']} directional lighting, ground shadows, depth-based positioning")
        
        if return_debug:
            debug_info = {
                'bg_lighting': bg_lighting,
                'fg_depth': fg_depth,
                'bg_depth': bg_depth,
                'scale_used': scale_factor,
                'position_used': position,
                'fg_lit': fg_lit,
                'bg_shadowed': bg_shadowed
            }
            return result, debug_info
        
        return result

# Función helper para manejar conversiones
def convert_rgba_to_rgb(rgba_array, background_color=(255, 255, 255)):
    """
    Convierte RGBA a RGB usando un color de fondo
    
    Args:
        rgba_array: numpy array (H, W, 4) RGBA
        background_color: tuple (R, G, B) - color de fondo para alpha blending
    
    Returns:
        rgb_array: numpy array (H, W, 3) RGB
    """
    if rgba_array.shape[2] != 4:
        return rgba_array  # Ya es RGB
    
    rgb = rgba_array[:, :, :3].astype(np.float32)
    alpha = rgba_array[:, :, 3:4].astype(np.float32) / 255.0
    
    bg_color = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
    
    # Alpha blending: result = rgb * alpha + background * (1 - alpha)
    result = rgb * alpha + bg_color * (1 - alpha)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def smart_composite_arrays(foreground_rgb, background_rgb, model_quality="fast", 
                          apply_dof=True, shadow_intensity=0.15):
    """
    Función simple para composición automática con arrays numpy
    
    Args:
        foreground_rgb: numpy array (H, W, 3) RGB o (H, W, 4) RGBA - imagen foreground
        background_rgb: numpy array (H, W, 3) RGB - imagen background
        model_quality: str - "fast", "quality", o "best"
        apply_dof: bool - aplicar depth of field
        shadow_intensity: float 0-1 - intensidad de sombras (DEFAULT REDUCIDO: 0.15)
    
    Returns:
        result_array: numpy array (H, W, 3) RGB - composición final
    """
    compositor = SmartAutoCompositor(model_type=model_quality)
    
    result = compositor.auto_composite(
        foreground_rgb, 
        background_rgb,
        apply_dof=apply_dof,
        shadow_intensity=shadow_intensity
    )
    
    return result

def smart_composite_arrays_debug(foreground_rgb, background_rgb, model_quality="fast", 
                                shadow_intensity=0.15):
    """
    Versión con información de debug
    
    Returns:
        result_array: numpy array (H, W, 3) RGB - composición final
        debug_info: dict - información detallada del proceso
    """
    compositor = SmartAutoCompositor(model_type=model_quality)
    
    result, debug_info = compositor.auto_composite(
        foreground_rgb, 
        background_rgb,
        shadow_intensity=shadow_intensity,
        return_debug=True
    )
    
    return result, debug_info

# Función para ajuste fino de sombras
def smart_composite_custom_shadows(foreground_rgb, background_rgb, 
                                 shadow_intensity=0.1, model_quality="fast"):
    """
    Versión con control total de sombras
    
    Args:
        shadow_intensity: 0.0 = sin sombras, 0.1 = muy sutiles, 0.2 = normales, 0.3+ = intensas
    """
    compositor = SmartAutoCompositor(model_type=model_quality)
    
    result = compositor.auto_composite(
        foreground_rgb, 
        background_rgb,
        shadow_intensity=shadow_intensity,
        apply_dof=True
    )
    
    return result

# Ejemplo de uso actualizado
if __name__ == "__main__":
    print("💡 Ejemplos de uso con sombras mejoradas:")
    print()
    print("# Uso básico (sombras sutiles por defecto):")
    print("result = smart_composite_arrays(foreground_rgba, background_rgb)")
    print()
    print("# Sombras muy sutiles:")
    print("result = smart_composite_arrays(foreground_rgba, background_rgb, shadow_intensity=0.1)")
    print()
    print("# Sin sombras:")
    print("result = smart_composite_arrays(foreground_rgba, background_rgb, shadow_intensity=0.0)")
    print()
    print("# Control total:")
    print("result = smart_composite_custom_shadows(foreground_rgba, background_rgb, shadow_intensity=0.15)")
    print()
    print("🎯 NOTA: shadow_intensity por defecto ahora es 0.15 (antes 0.4)")
    print("🎯 Sombras ahora se proyectan correctamente hacia el suelo")
    print("🎯 Soporte completo para RGBA con alpha blending")