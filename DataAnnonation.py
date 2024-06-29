import os
import json
from PIL import Image

def create_annotations(dataset_dir):
    annotations = []
    class_to_id = {}
    category_id = 0
    
    for equipment_class in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, equipment_class)
        if not os.path.isdir(class_dir):
            continue
        
        class_to_id[equipment_class] = category_id
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            
            annotations.append({
                'file_name': img_path,
                'category_id': category_id
            })
        
        category_id += 1
    
    return annotations, class_to_id

dataset_dir = 'C:/Users/user/Documents/Deep Learning Projects/Gym Equipment Detector/Datatsets/'

annotations, class_to_id = create_annotations(dataset_dir)

annotations_file = 'annotations.json'
with open(annotations_file, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations saved to {annotations_file}")
