# Inference
python inference_realesrgan.py -n RealESRGAN_x4plus -i train_pico/lq/vas.png --model_path experiments/finetune_RealESRGANx4plus_pico_L1_loss/models/net_g_350.pth --suffix finetuned_L1_only

# Training
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml