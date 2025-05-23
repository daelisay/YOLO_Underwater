import os
import numpy as np

def list_files_in_folder(folder):
    files = os.listdir(folder)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return files

base_dir = '/kaggle/working/YOLO_Underwater/data/image_folder'

train_files = list_files_in_folder(os.path.join(base_dir, 'train'))
val_files = list_files_in_folder(os.path.join(base_dir, 'val'))
test_files = list_files_in_folder(os.path.join(base_dir, 'test'))

# Simpan ke masing-masing file txt (tanpa ekstensi)
with open('/kaggle/working/YOLO_Underwater/data/train.txt', 'w') as f:
    for name in train_files:
        f.write(name.rsplit('.', 1)[0] + '\n')

with open('/kaggle/working/YOLO_Underwater/data/val.txt', 'w') as f:
    for name in val_files:
        f.write(name.rsplit('.', 1)[0] + '\n')

with open('/kaggle/working/YOLO_Underwater/data/test.txt', 'w') as f:
    for name in test_files:
        f.write(name.rsplit('.', 1)[0] + '\n')