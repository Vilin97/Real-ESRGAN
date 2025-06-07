import os
import shutil
from PIL import Image
from tqdm import tqdm

SRC_BASE = '/gscratch/krishna/ayanab/fff/data'
DST_BASE = '/gscratch/krishna/vilin/Real-ESRGAN/data'
SPLITS = ['full_train', 'full_test', 'full_val']
TYPES = ['gt', 'lq']
SIZES = {'gt': (1024, 1024), 'lq': (256, 256)}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_images():
    for split in SPLITS:
        for t in TYPES:
            src_dir = os.path.join(SRC_BASE, split, t)
            dst_dir = os.path.join(DST_BASE, split, t)
            ensure_dir(dst_dir)
            fnames = os.listdir(src_dir)
            for fname in tqdm(fnames, desc=f"{split}/{t}"):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                if os.path.exists(dst_path):
                    continue
                try:
                    with Image.open(src_path) as img:
                        img = img.convert('RGB')
                        img = img.resize(SIZES[t], Image.LANCZOS)
                        img.save(dst_path)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")

if __name__ == '__main__':
    process_images()
