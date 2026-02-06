import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm

# Model definition

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x.view(x.size(0), -1)


class GramPool(nn.Module):
    def __init__(self, in_channels, out_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.proj = nn.Linear(in_channels * in_channels, out_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        feat = x.view(b, c, n)
        gram = torch.bmm(feat, feat.transpose(1, 2))  # (B, C, C)
        gram = gram / n
        gram = gram.view(b, c * c)
        gram = self.proj(gram)
        return gram


class RockNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # backbone: get C3 (384) and C4 (768)
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False, # weights are in .pt file
            features_only=True,
            out_indices=(2, 3)
        )

        c3_channels = 384
        c4_channels = 768
        fused_channels = 256

        # reduce to 256
        self.c3_reduce = nn.Conv2d(c3_channels, fused_channels, kernel_size=1)
        self.c4_reduce = nn.Conv2d(c4_channels, fused_channels, kernel_size=1)

        # fusion: conv -> ln -> gelu
        self.fuse_conv = nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1)
        self.fuse_ln = nn.LayerNorm(fused_channels, elementwise_affine=True)
        self.fuse_act = nn.GELU()

        # heads
        self.gem = GeM()
        self.gram = GramPool(in_channels=fused_channels, out_dim=256)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)     # list of len 2
        c3, c4 = feats               # c3: (B, 384, h3, w3), c4: (B, 768, h4, w4)

        c3 = self.c3_reduce(c3)      # (B, 256, h3, w3)
        c4 = self.c4_reduce(c4)      # (B, 256, h4, w4)

        # upsample c4 to c3 size
        c4_up = F.interpolate(c4, size=c3.shape[2:], mode="bilinear", align_corners=False)

        fused = c3 + c4_up           # (B, 256, h3, w3)

        # conv
        fused = self.fuse_conv(fused)  # (B, 256, h3, w3)

        # LayerNorm over channels: (B, C, H, W) -> (B, H, W, C)
        b, c, h, w = fused.shape
        fused = fused.permute(0, 2, 3, 1)         # (B, H, W, C)
        fused = self.fuse_ln(fused)               # LN over C
        fused = fused.permute(0, 3, 1, 2).contiguous()  # back to (B, C, H, W)

        fused = self.fuse_act(fused)

        # heads
        gem_feat = self.gem(fused)      # (B, 256)
        gram_feat = self.gram(fused)    # (B, 256)

        feat = torch.cat([gem_feat, gram_feat], dim=1)  # (B, 512)
        logits = self.classifier(feat)
        return logits, feat

# Transforms (val_tfms)

img_size = 640
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_tfms = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Class names

CLASS_NAMES = [
    'basalt ilmenite', 'basalt olivine', 'basalt pigeonite', 'breccia impact'
]

# Inference functions

def load_model(weights_path: str, device: torch.device, temperature: float = 1.0):
    num_classes = len(CLASS_NAMES)
    if num_classes == 0:
        raise ValueError("CLASS_NAMES is empty. Please fill in your class labels.")

    model = RockNet(num_classes=num_classes).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.temperature = temperature
    return model


def preprocess_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    tensor = val_tfms(img)            # (3, H, W)
    tensor = tensor.unsqueeze(0)      # (1, 3, H, W)
    return tensor


@torch.no_grad()
def predict_one(model: RockNet, img_tensor: torch.Tensor, device: torch.device):
    img_tensor = img_tensor.to(device)

    logits, _ = model(img_tensor)       # (1, C)

    # Apply temperature scaling if set
    T_val = getattr(model, "temperature", 1.0)
    if T_val != 1.0:
        logits = logits / T_val

    probs = F.softmax(logits, dim=1)    # (1, C)

    # Top-3 predictions
    k = min(3, probs.size(1))
    top_conf, top_idx = torch.topk(probs, k=k, dim=1)  # (1, k), (1, k)
    top_conf = top_conf.squeeze(0).tolist()
    top_idx = top_idx.squeeze(0).tolist()
    topk = [
        {
            "label": CLASS_NAMES[int(i)],
            "confidence": float(c),
        }
        for c, i in zip(top_conf, top_idx)
    ]

    # Keep top-1 fields for backward compatibility
    return {
        "label": topk[0]["label"],
        "confidence": topk[0]["confidence"],
        "top3": topk,
    }

def write_json(result_dict, json_path: str):
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RockNet inference on Jetson Orin")
    parser.add_argument(
        "--weights",
        type=str,
        default="best_rocknet.pt",
        help="Path to trained weights file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.462,
        help="Temperature for calibration (fitted T* = 1.462)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="default: <image_basename>_prediction.json"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.weights, device=device, temperature=args.temperature)

    # Preprocess image
    img_tensor = preprocess_image(args.image)

    # Predict
    result = predict_one(model, img_tensor, device)

    # Prepare output dict
    output_dict = result["top3"]

    # Print to console
    print(output_dict)

    # Write JSON
    if args.output_json is not None:
        json_path = args.output_json
    else:
        base, _ = os.path.splitext(os.path.basename(args.image))
        json_path = f"{base}_prediction.json"

    write_json(output_dict, json_path)


if __name__ == "__main__":
    main()
