"""
based off of rocknet_infer.py:
Backbone  : ConvNeXt-Tiny (ImageNet pretrained) → C3 (384ch) + C4 (768ch)
Fusion    : 1x1 reduce → bilinear upsample C4 → add → 3x3 conv → LayerNorm → GELU → 256ch
Pooling   : GeM (global avg) + GramPool (texture) → concat → 512-dim feature vector
Stage 1   : Primary head → 4-way classifier (basalt, anorthosite, breccia, other)
Stage 2   : Conditional FamilyHeads — activate only for the predicted rock family
              Each FamilyHead: shared 512→256 projection + independent Linear per feature
Stage 3   : Deterministic explanation layer — GEOLOGY_NOTES key lookup, no generation


To Be Implemented (placeholder in place): 
- Hard-negative "other" class: non-rock inputs, poor-quality images, and out-of-distribution rocks...
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm


# Four-class primary output (includes hard-negative "other")
FAMILY_NAMES  = ["basalt", "anorthosite", "breccia", "other"]
ROCK_FAMILIES = ["basalt", "anorthosite", "breccia"]   # secondary heads only
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}
IDX_TO_FAMILY = {i: f for f, i in FAMILY_TO_IDX.items()}


# Feature schemas
FEATURE_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "basalt": {
        "groundmass_texture": ["aphanitic", "medium_grained", "phaneritic"],
        "vesicularity":       ["low", "moderate", "high"],
        "luster":             ["dull", "moderate", "metallic"],
        "phenocryst_hint":    ["n/a", "olivine_like", "pyroxene_like"],  # "n/a" = no phenocrysts
    },
    "anorthosite": {
        "crystal_fabric":     ["equigranular", "polygonal"],
        "brightness":         ["moderate", "high"],
        "surface_character":  ["weathered", "vitreous"],
        "mafic_content_hint": ["low", "moderate"],
    },
    "breccia": {
        "clast_angularity": ["rounded", "angular"],
        "sorting":          ["poor", "well"],
        "support_fabric":   ["matrix_supported", "clast_supported"],
    },
}

# Labeling schemas
LABELING_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "basalt": {
        "groundmass_texture": ["aphanitic", "medium_grained", "phaneritic", "n/a"],
        "vesicularity":       ["low", "moderate", "high", "n/a"],
        "luster":             ["dull", "moderate", "metallic", "n/a"],
        "phenocryst_hint":    ["n/a", "olivine_like", "pyroxene_like"],   # n/a is class 0
    },
    "anorthosite": {
        "crystal_fabric":     ["equigranular", "polygonal", "n/a"],
        "brightness":         ["moderate", "high", "n/a"],
        "surface_character":  ["weathered", "vitreous", "n/a"],
        "mafic_content_hint": ["low", "moderate", "n/a"],
    },
    "breccia": {
        "clast_angularity": ["rounded", "angular", "n/a"],
        "sorting":          ["poor", "well", "n/a"],
        "support_fabric":   ["matrix_supported", "clast_supported", "n/a"],
    },
}

# Sentinel for PyTorch cross_entropy — feature labels mapped to this are not trained on.
IGNORE_INDEX = -100

# Features where "n/a" is a valid output class (not a missing-label sentinel).
VALID_NA_FEATURES = {"phenocryst_hint"}

# Per-label confidence weights. Labelers assign 1/2/3; these scale the secondary-head loss.
# Mirrors label_tool.CONF_LEVELS — keep in sync.
CONF_LEVELS: dict[int, float] = {1: 0.5, 2: 0.75, 3: 1.0}
DEFAULT_CONF_WEIGHT: float = 1.0   # backward compat: old string-format manifest entries


def get_feature_value(entry) -> str:
    """Accept both old string format and new dict format from manifest features."""
    return entry["value"] if isinstance(entry, dict) else entry

def get_feature_weight(entry) -> float:
    """Return the confidence weight for a feature entry. Defaults to 1.0 for old format."""
    if isinstance(entry, dict):
        return CONF_LEVELS.get(entry.get("conf", 3), 1.0)
    return DEFAULT_CONF_WEIGHT

# Convert a manifest feature entry (string or confidence dict) to a training index.
def label_to_idx(feat: str, value_or_entry, labeling_classes: list[str]) -> int:
    value = get_feature_value(value_or_entry)
    if value == "n/a" and feat not in VALID_NA_FEATURES:
        return IGNORE_INDEX
    try:
        return labeling_classes.index(value)
    except ValueError:
        return IGNORE_INDEX


GEOLOGY_NOTES: dict[str, str] = {
    # BASALT ─────────────────────────────────────────────────────────────────
    "basalt_groundmass_texture_aphanitic":
        "Fine-grained groundmass indicates rapid cooling with crystals too small to resolve.",
    "basalt_groundmass_texture_medium_grained":
        "Visible grains suggest slower cooling than aphanitic basalt, allowing partial crystal growth.",
    "basalt_groundmass_texture_phaneritic":
        "Coarser grain size indicates slower cooling or crystal accumulation prior to eruption.",

    "basalt_vesicularity_low":
        "Sparse vesicles indicate limited volatile exsolution during solidification.",
    "basalt_vesicularity_moderate":
        "Moderate vesicles reflect partial gas escape with some bubbles preserved during cooling.",
    "basalt_vesicularity_high":
        "Abundant vesicles indicate volatile-rich magma and rapid degassing during eruption.",

    "basalt_luster_dull":
        "Dull surface is typical of fine-grained or weathered basalt with minimal crystal faces.",
    "basalt_luster_moderate":
        "Moderate reflectivity indicates partial crystalline faces or minor glass content.",
    "basalt_luster_metallic":
        "Metallic luster suggests glassy phases or reflective ilmenite-rich mineral surfaces.",

    "basalt_phenocryst_hint_n/a":
        "No visible phenocrysts suggests a uniformly fine-grained or rapidly cooled lava.",
    "basalt_phenocryst_hint_olivine_like":
        "Rounded greenish grains are consistent with olivine phenocrysts crystallized prior to eruption.",
    "basalt_phenocryst_hint_pyroxene_like":
        "Dark blocky crystals are consistent with pyroxene phenocrysts formed during magma evolution.",

    # ANORTHOSITE ─────────────────────────────────────────────────────────────
    "anorthosite_crystal_fabric_equigranular":
        "Uniform crystal sizes indicate stable cooling conditions and a well-equilibrated plutonic texture.",
    "anorthosite_crystal_fabric_polygonal":
        "Polygonal interlocking grains reflect recrystallization and textural equilibration under solid-state conditions.",

    "anorthosite_brightness_moderate":
        "Moderate brightness reflects a feldspar-dominated composition with minor mafic phases.",
    "anorthosite_brightness_high":
        "High brightness indicates a plagioclase-rich composition with minimal dark mineral content.",

    "anorthosite_surface_character_weathered":
        "Weathered surface suggests alteration, microfracturing, or fine-grained secondary products.",
    "anorthosite_surface_character_vitreous":
        "Vitreous surfaces indicate fresh crystal faces or minimal weathering of plagioclase grains.",

    "anorthosite_mafic_content_hint_low":
        "Minor dark minerals suggest limited mafic inclusions within a predominantly plagioclase matrix.",
    "anorthosite_mafic_content_hint_moderate":
        "Noticeable mafic content indicates deviation from pure anorthosite toward more mixed lithology.",

    # BRECCIA ──────────────────────────────────────────────────────────────────
    "breccia_clast_angularity_rounded":
        "Rounded clasts indicate transport abrasion or prolonged reworking of original fragments.",
    "breccia_clast_angularity_angular":
        "Angular clasts indicate minimal transport and mechanical fragmentation at or near the source.",

    "breccia_sorting_poor":
        "Wide range of clast sizes indicates rapid deposition with minimal sorting processes.",
    "breccia_sorting_well":
        "Relatively uniform clast sizes indicate some degree of transport-driven sorting.",

    "breccia_support_fabric_matrix_supported":
        "Matrix-supported fabric indicates fine material dominates and separates larger clasts.",
    "breccia_support_fabric_clast_supported":
        "Clast-supported fabric indicates large fragments are in contact, suggesting collapse or rockfall origin.",
}


OUTPUT_JSON_SCHEMA = """
RockNet v2.0 — Inference Output Schema
═══════════════════════════════════════════════
{
  "schema_version": "rocknet_v2.0",

  "image_path":   string,          // absolute path to input image
  "inference_ms": number,          // wall-clock time for this call

  "primary": {
    "family":     "basalt" | "anorthosite" | "breccia" | "other",
    "confidence": float [0,1],     // calibrated softmax probability of top family
    "tier":       "high" | "medium" | "low" | "uncertain",
    "scores": {                    // all four calibrated probabilities
      "basalt": float,
      "anorthosite": float,
      "breccia": float,
      "other": float
    }
  },

  // null when primary.family == "other" OR primary.tier == "uncertain"
  "features": {
    "<feature_name>": {
      "value":      string | "uncertain",  // top class or "uncertain" if conf < threshold
      "confidence": float [0,1],
      "tier":       "high" | "medium" | "low" | "uncertain",
      "display":    bool,                  // false when tier == "uncertain"
      "scores":     { "<class>": float, ... }
    }
  },

  // only for features where display == true
  "geology_notes": [
    { "feature": string, "value": string, "note": string }
  ],

  "ui": {
    "show_primary":        bool,           // always true
    "show_features":       bool,           // false if primary tier is uncertain
    "suppressed_features": [string]        // feature names with display=false
  }
}

Confidence tiers (identical thresholds for primary and all feature heads):
  high      : confidence >= 0.80
  medium    : 0.60 <= confidence < 0.80
  low       : 0.45 <= confidence < 0.60
  uncertain : confidence < 0.45   → value set to "uncertain", display=false
"""

CONF_TIERS = [
    (0.80, "high"),
    (0.60, "medium"),
    (0.45, "low"),
    (0.00, "uncertain"),
]
FEATURE_DISPLAY_THRESHOLD = 0.45


def confidence_tier(conf: float) -> str:
    for threshold, label in CONF_TIERS:
        if conf >= threshold:
            return label
    return "uncertain"


IMG_SIZE = 448
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomRotation(45),
    T.RandomGrayscale(p=0.05),   # simulate dusty / monochrome field images
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).view(x.size(0), -1)


class GramPool(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_channels * in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feat = x.view(b, c, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2)) / (h * w)
        return self.proj(gram.view(b, c * c))


class FamilyHead(nn.Module):
    def __init__(self, in_dim: int, feature_schema: dict[str, list[str]]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({
            feat: nn.Linear(256, len(classes))
            for feat, classes in feature_schema.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x)
        return {feat: head(shared) for feat, head in self.heads.items()}


# Multi head rock classification model 
class RockNetV2(nn.Module):
    FEAT_DIM    = 512
    NUM_PRIMARY = len(FAMILY_NAMES)   # 4

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3),   # C3: 384ch, C4: 768ch
        )

        self.c3_reduce = nn.Conv2d(384, 256, kernel_size=1)
        self.c4_reduce = nn.Conv2d(768, 256, kernel_size=1)
        self.fuse_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fuse_ln   = nn.LayerNorm(256, elementwise_affine=True)
        self.fuse_act  = nn.GELU()

        self.gem  = GeM()
        self.gram = GramPool(in_channels=256, out_dim=256)

        # Stage 1 — primary head
        self.primary_head = nn.Sequential(
            nn.Linear(self.FEAT_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.NUM_PRIMARY),
        )

        # Stage 2 — one FamilyHead per rock family (not "other")
        self.family_heads = nn.ModuleDict({
            family: FamilyHead(self.FEAT_DIM, FEATURE_SCHEMAS[family])
            for family in ROCK_FAMILIES
        })

        # Temperature scalars for post-hoc calibration.
        # requires_grad=False until calibration phase.
        self.primary_temp = nn.Parameter(torch.ones(1), requires_grad=False)
        self.feature_temps = nn.ParameterDict({
            family: nn.ParameterDict({
                feat: nn.Parameter(torch.ones(1), requires_grad=False)
                for feat in FEATURE_SCHEMAS[family]
            })
            for family in ROCK_FAMILIES
        })

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4 = self.backbone(x)
        c3 = self.c3_reduce(c3)
        c4 = self.c4_reduce(c4)
        c4_up = F.interpolate(c4, size=c3.shape[2:], mode="bilinear", align_corners=False)

        fused = c3 + c4_up
        fused = self.fuse_conv(fused)
        b, c, h, w = fused.shape
        fused = self.fuse_ln(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        fused = self.fuse_act(fused)

        return torch.cat([self.gem(fused), self.gram(fused)], dim=1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> dict:
        feat          = self.extract_features(x)
        family_logits = self.primary_head(feat)
        feature_logits = {
            family: head(feat) for family, head in self.family_heads.items()
        }
        return {
            "family_logits":  family_logits,
            "feature_logits": feature_logits,
            "feat":           feat,
        }


LAMBDA_SECONDARY = 0.5       # secondary head contribution to total loss
SECONDARY_WARMUP_EPOCHS = 5  # epochs before secondary losses activate


def compute_loss(
    outputs: dict,
    family_targets: torch.Tensor,
    feature_targets: dict[str, dict[str, torch.Tensor]],
    epoch: int = 999,
    label_smoothing: float = 0.1,
    feat_weights: "dict[str, dict[str, torch.Tensor]] | None" = None,
) -> tuple[torch.Tensor, dict]:
    primary_loss = F.cross_entropy(
        outputs["family_logits"],
        family_targets,
        label_smoothing=label_smoothing,
    )

    secondary_loss  = torch.zeros(1, device=primary_loss.device).squeeze()
    secondary_terms = 0

    if epoch >= SECONDARY_WARMUP_EPOCHS:
        for family in ROCK_FAMILIES:
            family_idx  = FAMILY_TO_IDX[family]
            family_mask = family_targets == family_idx

            for feat, tgt in feature_targets[family].items():
                valid = family_mask & (tgt != IGNORE_INDEX)
                if not valid.any():
                    continue
                logits   = outputs["feature_logits"][family][feat][valid]
                per_loss = F.cross_entropy(logits, tgt[valid], reduction="none")  # (N_valid,)

                if feat_weights is not None:
                    w = feat_weights[family][feat][valid].to(per_loss.device)
                    feat_loss = (per_loss * w).sum() / w.sum().clamp(min=1e-8)
                else:
                    feat_loss = per_loss.mean()

                secondary_loss  = secondary_loss + feat_loss
                secondary_terms += 1

    if secondary_terms > 0:
        secondary_loss = secondary_loss / secondary_terms

    total = primary_loss + LAMBDA_SECONDARY * secondary_loss

    return total, {
        "total":           total.item(),
        "primary":         primary_loss.item(),
        "secondary":       secondary_loss.item() if secondary_terms else 0.0,
        "secondary_terms": secondary_terms,
    }


# two stage inference formatter 
@torch.no_grad()
def run_inference(
    model: RockNetV2,
    image_path: str,
    device: torch.device,
) -> dict:
    t0 = time.perf_counter()

    img    = Image.open(image_path).convert("RGB")
    tensor = val_tfms(img).unsqueeze(0).to(device)

    model.eval()
    outputs = model(tensor)

    # ── Primary ───────────────────────────────────────────────────────────
    fam_logits = outputs["family_logits"] / model.primary_temp.to(device)
    fam_probs  = F.softmax(fam_logits, dim=1).squeeze(0).cpu().tolist()
    fam_idx    = int(torch.argmax(torch.tensor(fam_probs)))
    fam_name   = IDX_TO_FAMILY[fam_idx]
    fam_conf   = fam_probs[fam_idx]
    fam_tier   = confidence_tier(fam_conf)

    primary = {
        "family":     fam_name,
        "confidence": round(fam_conf, 4),
        "tier":       fam_tier,
        "scores":     {f: round(p, 4) for f, p in zip(FAMILY_NAMES, fam_probs)},
    }

    # ── Secondary ─────────────────────────────────────────────────────────
    features   = None
    geo_notes  = []
    suppressed = []
    show_feats = (fam_name in ROCK_FAMILIES) and (fam_tier != "uncertain")

    if show_feats:
        features = {}
        schema   = FEATURE_SCHEMAS[fam_name]
        head_out = outputs["feature_logits"][fam_name]

        for feat_name, classes in schema.items():
            temp   = model.feature_temps[fam_name][feat_name].to(device)
            logits = head_out[feat_name] / temp
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            top_i  = int(torch.argmax(torch.tensor(probs)))
            conf   = probs[top_i]
            tier   = confidence_tier(conf)
            display = tier != "uncertain"
            value   = classes[top_i] if display else "uncertain"

            features[feat_name] = {
                "value":      value,
                "confidence": round(conf, 4),
                "tier":       tier,
                "display":    display,
                "scores":     {c: round(p, 4) for c, p in zip(classes, probs)},
            }

            if display:
                note = GEOLOGY_NOTES.get(f"{fam_name}_{feat_name}_{value}", "")
                if note:
                    geo_notes.append({"feature": feat_name, "value": value, "note": note})
            else:
                suppressed.append(feat_name)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "schema_version": "rocknet_v2.0",
        "image_path":     str(image_path),
        "inference_ms":   elapsed_ms,
        "primary":        primary,
        "features":       features,
        "geology_notes":  geo_notes,
        "ui": {
            "show_primary":        True,
            "show_features":       show_feats,
            "suppressed_features": suppressed,
        },
    }


def save_checkpoint(
    model: RockNetV2,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> tuple[RockNetV2, dict]:
    payload    = torch.load(str(path), map_location=device, weights_only=False)
    model      = RockNetV2().to(device)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=strict)
    return model, payload.get("metadata", {})
