"""Models used by the human segmentation comparisons."""
import torch
import torch.nn.functional as F
from torch import nn

from train_phantom_unet import UNet


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
    raise ValueError(f"Unknown model kind: {kind}")
