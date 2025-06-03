#%%
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

lrs = [1e-3, 1e-4, 1e-5, 1e-6]
img_names = [
    "40yqys", "50hrvk", "8e3uix",
    "4ebiy5", "d85khg", "5a2rik", "gbre4r",
    "4w9stg", "72dznv", "7sipet", "9rv0jd",  "hucgfs",
]

def get_value_from_filename(filename, img_name):
    # filename: {img_name}_{value}.png
    base = os.path.basename(filename)
    value_str = base[len(img_name)+1:-4]
    try:
        return float(value_str)
    except ValueError:
        return None

for img_name in img_names:
    fig, axs = plt.subplots(2, len(lrs), figsize=(4*len(lrs), 8))  # 2 rows: min (top), max (bottom)
    fig.suptitle(img_name)  # Set the figure title
    for j, lr in enumerate(lrs):
        lr_str = f"{lr:.0e}".replace("-0", "-")
        pattern = f"experiments/finetune_64_lr_{lr_str}/visualization/{img_name}/{img_name}_*.png"
        files = glob.glob(pattern)
        values_files = [(get_value_from_filename(f, img_name), f) for f in files]
        values_files = [vf for vf in values_files if vf[0] is not None]
        if not values_files:
            continue
        min_val, min_file = min(values_files, key=lambda x: x[0])
        max_val, max_file = max(values_files, key=lambda x: x[0])

        # Min image (top row)
        ax_min = axs[0, j]
        img_min = Image.open(min_file)
        ax_min.imshow(img_min)
        ax_min.axis('off')
        ax_min.set_title(f"lr={lr_str} iter={int(min_val)}")

        # Max image (bottom row)
        ax_max = axs[1, j]
        img_max = Image.open(max_file)
        ax_max.imshow(img_max)
        ax_max.axis('off')
        ax_max.set_title(f"lr={lr_str} iter={int(max_val)}")
    plt.tight_layout()
    os.makedirs("comparisons", exist_ok=True)
    out_path = f"comparisons/{img_name}.png"
    if os.path.exists(out_path):
        print(f"Warning: {out_path} exists and will be overwritten.")
    fig.savefig(out_path)
    plt.show()