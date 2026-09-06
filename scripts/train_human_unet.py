"""Convenient entry point for the CathAction human U-Net baseline."""
from train_phantom_unet import main


if __name__ == "__main__":
    main(default_dataset="human")
