import os
import json

def diagnose_dataset():
    data_dir = 'persons/project'
    dataset_dirs = [d for d in os.listdir(data_dir) if d.startswith('ds') and os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"=== DIAGNÓSTICO DEL DATASET ===")
    print(f"Directorio base: {data_dir}")
    print(f"Subdirectorios encontrados: {len(dataset_dirs)}")
    print(f"Lista: {dataset_dirs}")
    print()
    
    total_images = 0
    total_annotations = 0
    valid_pairs = 0
    
    for ds_dir in dataset_dirs[:3]:  # Solo los primeros 3 para no saturar
        print(f"--- Analizando {ds_dir} ---")
        ds_path = os.path.join(data_dir, ds_dir)
        img_dir = os.path.join(ds_path, 'img')
        ann_dir = os.path.join(ds_path, 'ann')
        
        print(f"Directorio imágenes: {img_dir}")
        print(f"Existe: {os.path.exists(img_dir)}")
        
        print(f"Directorio anotaciones: {ann_dir}")
        print(f"Existe: {os.path.exists(ann_dir)}")
        
        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Imágenes: {len(img_files)}")
            print(f"Ejemplos: {img_files[:3]}")
            total_images += len(img_files)
            
        if os.path.exists(ann_dir):
            ann_files = [f for f in os.listdir(ann_dir) if f.lower().endswith('.json')]
            print(f"Anotaciones: {len(ann_files)}")
            print(f"Ejemplos: {ann_files[:3]}")
            total_annotations += len(ann_files)
            
            # Verificar contenido de una anotación
            if ann_files:
                sample_ann = os.path.join(ann_dir, ann_files[0])
                try:
                    with open(sample_ann, 'r') as f:
                        data = json.load(f)
                    print(f"Estructura anotación: {list(data.keys())}")
                    if 'objects' in data:
                        print(f"Objetos: {len(data['objects'])}")
                        if data['objects']:
                            obj = data['objects'][0]
                            print(f"Primer objeto: {obj.get('classTitle', 'N/A')} - {obj.get('geometryType', 'N/A')}")
                except Exception as e:
                    print(f"Error leyendo anotación: {e}")
        
        print()
    
    print(f"=== RESUMEN ===")
    print(f"Total imágenes: {total_images}")
    print(f"Total anotaciones: {total_annotations}")

if __name__ == "__main__":
    diagnose_dataset()