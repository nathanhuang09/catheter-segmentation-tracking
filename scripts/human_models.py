"""Models used by the human segmentation comparisons."""
import torch
import torch.nn.functional as F
from torch import nn

from train_phantom_unet import DoubleConv, UNet


class TemporalFusion(nn.Module):
    """Fuse ordered features plus explicit changes relative to the current frame."""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(5 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        old, middle, current = features
        return self.layers(torch.cat((old, middle, current, current - old, current - middle), dim=1))


class SharedEncoderTemporalUNet(nn.Module):
    """Three-frame U-Net with one shared encoder and multi-scale temporal fusion.

    Input channels are ordered grayscale frames t-4, t-2, and t. Each frame is
    encoded separately with identical weights, then fused at every resolution.
    """
    def __init__(self, out_channels=1, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1, self.enc2 = DoubleConv(1, b), DoubleConv(b, b * 2)
        self.enc3, self.enc4 = DoubleConv(b * 2, b * 4), DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.fusions = nn.ModuleList([TemporalFusion(c) for c in (b, b * 2, b * 4, b * 8, b * 16)])
        self.up4, self.dec4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2), DoubleConv(b * 16, b * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2), DoubleConv(b * 8, b * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2), DoubleConv(b * 4, b * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(b * 2, b, 2, 2), DoubleConv(b * 2, b)
        self.output = nn.Conv2d(b, out_channels, 1)

    def encode(self, frame):
        e1 = self.enc1(frame)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, bottleneck

    def forward(self, frames):
        if frames.shape[1] != 3:
            raise ValueError(f"Expected t-4,t-2,t as 3 channels, got {frames.shape[1]}")
        encoded = [self.encode(frames[:, index:index + 1]) for index in range(3)]
        e1, e2, e3, e4, x = [self.fusions[level]([features[level] for features in encoded])
                              for level in range(5)]
        x = self.dec4(torch.cat((self.up4(x), e4), 1))
        x = self.dec3(torch.cat((self.up3(x), e3), 1))
        x = self.dec2(torch.cat((self.up2(x), e2), 1))
        x = self.dec1(torch.cat((self.up1(x), e1), 1))
        return self.output(x)


class SegFormerBinary(nn.Module):
    """Pretrained MiT SegFormer with one binary foreground logit."""
    def __init__(self, model_name="nvidia/mit-b0"):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise ImportError("Install SegFormer support with: pip install transformers") from error
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = 1
        config.id2label = {0: "instrument"}
        config.label2id = {"instrument": 0}
        self.network = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, images):
        normalized = (images - self.mean) / self.std
        logits = self.network(pixel_values=normalized).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def build_model(kind: str, model_name="nvidia/mit-b0"):
    if kind in {"unet", "unet3"}:
        return UNet(3, 1, 16)
    if kind == "segformer":
        return SegFormerBinary(model_name)
    if kind == "temporal_unet":
        return SharedEncoderTemporalUNet(1, 16)
    raise ValueError(f"Unknown model kind: {kind}")
