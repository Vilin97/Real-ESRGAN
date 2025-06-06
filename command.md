# Inference
python inference_realesrgan.py -n RealESRGAN_x4plus -i train_pico/lq/vas.png --model_path experiments/finetune_RealESRGANx4plus_pico_L1_loss/models/net_g_350.pth --suffix finetuned_L1_only

# Training
CUDA_VISIBLE_DEVICES=0 python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml
python realesrgan/train.py -opt options/sweep/finetune_lr_1e-3.yaml

# Generate meta info file
python scripts/generate_meta_info_pairdata.py --input train_64/gt train_64/lq --meta_info train_64/meta_info.txt