#%%
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from  tqdm import tqdm

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
    num_cols = 2 + len(lrs)  # GT, LQ, and one max image per lr
    fig, axs = plt.subplots(
        1, num_cols,
        figsize=(4 * num_cols, 4),
        dpi=500
    )
    fig.subplots_adjust(wspace=0, hspace=0)  # remove horizontal space
    fig.suptitle(img_name)

    # GT image (leftmost)
    gt_path_jpeg = f"train_64/gt/{img_name}.jpeg"
    gt_path_jpg = f"train_64/gt/{img_name}.jpg"
    gt_path_png = f"train_64/gt/{img_name}.png"
    if os.path.exists(gt_path_jpeg):
        gt_path = gt_path_jpeg
    elif os.path.exists(gt_path_jpg):
        gt_path = gt_path_jpg
    elif os.path.exists(gt_path_png):
        gt_path = gt_path_png
    else:
        raise FileNotFoundError(f"GT image not found for {img_name} at {gt_path_jpeg}, {gt_path_jpg}, or {gt_path_png}")
    gt_img = Image.open(gt_path)
    axs[0].imshow(gt_img)
    axs[0].axis('off')
    axs[0].set_title("GT")

    # LQ image (second)
    lq_path_jpeg = f"train_64/lq/{img_name}.jpeg"
    lq_path_jpg = f"train_64/lq/{img_name}.jpg"
    lq_path_png = f"train_64/lq/{img_name}.png"
    if os.path.exists(lq_path_jpeg):
        lq_path = lq_path_jpeg
    elif os.path.exists(lq_path_jpg):
        lq_path = lq_path_jpg
    elif os.path.exists(lq_path_png):
        lq_path = lq_path_png
    else:
        raise FileNotFoundError(f"LQ image not found for {img_name} at {lq_path_jpeg}, {lq_path_jpg}, or {lq_path_png}")
    lq_img = Image.open(lq_path)
    axs[1].imshow(lq_img)
    axs[1].axis('off')
    axs[1].set_title("LQ")

    for j, lr in tqdm(enumerate(lrs)):
        lr_str = f"{lr:.0e}".replace("-0", "-")
        pattern = f"experiments/finetune_64_lr_{lr_str}/visualization/{img_name}/{img_name}_*.png"
        files = glob.glob(pattern)
        values_files = [(get_value_from_filename(f, img_name), f) for f in files]
        values_files = [vf for vf in values_files if vf[0] is not None]
        if not values_files:
            continue
        max_val, max_file = max(values_files, key=lambda x: x[0])

        # Max image for this lr (columns 2, 3, ...)
        ax = axs[2 + j]
        img = Image.open(max_file)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"lr={lr_str} iter={int(max_val)}")

    plt.tight_layout(pad=0)  # No padding
    os.makedirs("comparisons", exist_ok=True)
    out_path = f"comparisons/{img_name}.png"
    if os.path.exists(out_path):
        print(f"Warning: {out_path} exists and will be overwritten.")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()

# %%
