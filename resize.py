import os
import shutil
from PIL import Image
from tqdm import tqdm
import random

SRC_BASE = '/gscratch/krishna/ayanab/fff/data'
DST_BASE = '/gscratch/krishna/vilin/Real-ESRGAN/data'
SPLITS = ['full_train', 'full_test', 'full_val']
TYPES = ['gt', 'lq']
SIZES = {'gt': (1024, 1024), 'lq': (256, 256)}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def process_images():
#     for split in SPLITS:
#         for t in TYPES:
#             src_dir = os.path.join(SRC_BASE, split, t)
#             dst_dir = os.path.join(DST_BASE, split, t)
#             ensure_dir(dst_dir)
#             fnames = os.listdir(src_dir)
#             for fname in tqdm(fnames, desc=f"{split}/{t}"):
#                 src_path = os.path.join(src_dir, fname)
#                 dst_path = os.path.join(dst_dir, fname)
#                 if os.path.exists(dst_path):
#                     continue
#                 try:
#                     with Image.open(src_path) as img:
#                         img = img.convert('RGB')
#                         img = img.resize(SIZES[t], Image.LANCZOS)
#                         img.save(dst_path)
#                 except Exception as e:
#                     print(f"Error processing {src_path}: {e}")

def create_partial_val():
    split = 'full_val'
    partial_split = 'partial_val'
    gt_dir = os.path.join(DST_BASE, split, 'gt')
    lq_dir = os.path.join(DST_BASE, split, 'lq')
    partial_gt_dir = os.path.join(DST_BASE, partial_split, 'gt')
    partial_lq_dir = os.path.join(DST_BASE, partial_split, 'lq')
    ensure_dir(partial_gt_dir)
    ensure_dir(partial_lq_dir)
    gt_fnames = set(os.listdir(gt_dir))
    lq_fnames = set(os.listdir(lq_dir))
    common_fnames = sorted(gt_fnames & lq_fnames)
    if not common_fnames:
        return
    sample_size = max(1, int(0.1 * len(common_fnames)))
    sampled = random.sample(common_fnames, sample_size)
    for fname in tqdm(sampled, desc="partial_val"):
        src_gt = os.path.join(gt_dir, fname)
        dst_gt = os.path.join(partial_gt_dir, fname)
        src_lq = os.path.join(lq_dir, fname)
        dst_lq = os.path.join(partial_lq_dir, fname)
        if not os.path.exists(dst_gt):
            shutil.copy2(src_gt, dst_gt)
        if not os.path.exists(dst_lq):
            shutil.copy2(src_lq, dst_lq)

if __name__ == '__main__':
    # process_images()
    create_partial_val()
