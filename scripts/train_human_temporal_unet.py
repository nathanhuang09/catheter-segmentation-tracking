"""Train the shared-encoder temporal U-Net using t-4, t-2, and t."""
from train_human_advanced import main


if __name__ == "__main__":
    main(default_model="temporal_unet")
