HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/hf_home accelerate launch -m axolotl.cli.train /workspace/train_config.yaml --deepspeed /workspace/axolotl/deepspeed_configs/zero2.json