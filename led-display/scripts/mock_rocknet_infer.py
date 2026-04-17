import argparse
import json
import random
import sys
import time
from pathlib import Path

FAMILIES = ["basalt", "anorthosite", "breccia"]
ALL_CLASSES = ["basalt", "anorthosite", "breccia", "other"]

FEATURE_CLASSES = {
    "basalt": {
        "groundmass_texture": ["aphanitic", "porphyritic", "glassy", "vesicular"],
        "vesicularity":       ["none", "low", "moderate", "high"],
        "luster":             ["vitreous", "dull", "waxy"],
        "phenocryst_hint":    ["none", "feldspar", "olivine", "pyroxene"],
    },
    "anorthosite": {
        "crystal_fabric":      ["foliated", "massive", "granular"],
        "brightness":          ["bright", "intermediate", "dark"],
        "surface_character":   ["smooth", "rough", "fractured"],
        "mafic_content_hint":  ["low", "moderate", "high"],
    },
    "breccia": {
        "clast_angularity": ["angular", "subangular", "rounded"],
        "sorting":          ["well_sorted", "poorly_sorted", "unsorted"],
        "support_fabric":   ["clast_supported", "matrix_supported"],
    },
}

GEOLOGY_NOTES = {
    "groundmass_texture": {
        "aphanitic":    "Fine-grained groundmass indicates rapid cooling at or near the surface.",
        "porphyritic":  "Large crystals set in fine groundmass reflect two-stage cooling history.",
        "glassy":       "Glassy texture results from quenching of lava, often in water or air.",
        "vesicular":    "Vesicles formed from gas bubbles trapped during rapid solidification.",
    },
    "vesicularity": {
        "none":     "Absence of vesicles suggests degassed or deep intrusive origin.",
        "low":      "Sparse vesicles indicate minor outgassing during solidification.",
        "moderate": "Moderate vesicularity typical of pahoehoe or aa lava flows.",
        "high":     "Abundant vesicles suggest vigorous outgassing near the vent.",
    },
    "luster": {
        "vitreous": "Glassy luster from smooth fracture surfaces in fine-grained silicate.",
        "dull":     "Dull luster often indicates alteration or weathering of primary minerals.",
        "waxy":     "Waxy luster associated with fine crystalline or crypto-crystalline silica.",
    },
    "phenocryst_hint": {
        "feldspar":  "Feldspar phenocrysts indicate relatively slow early crystallization.",
        "olivine":   "Olivine phenocrysts point to a mafic, mantle-derived magma source.",
        "pyroxene":  "Pyroxene phenocrysts reflect high-temperature crystallization from mafic melt.",
    },
    "crystal_fabric": {
        "foliated":  "Foliated fabric formed under differential stress during or after crystallization.",
        "massive":   "Massive fabric indicates crystallization without significant shear stress.",
        "granular":  "Granular texture reflects coarse interlocking crystals of roughly equal size.",
    },
    "brightness": {
        "bright":        "High albedo consistent with plagioclase-dominated mineralogy.",
        "intermediate":  "Intermediate brightness suggests mixed plagioclase and mafic mineral content.",
        "dark":          "Darker tone may indicate elevated mafic mineral or alteration-product content.",
    },
    "surface_character": {
        "smooth":    "Smooth surface may reflect polishing by abrasion or fine-grained mineralogy.",
        "rough":     "Rough surface typical of coarse crystalline or altered exterior.",
        "fractured": "Fractures may indicate impact-related stress or thermal cycling.",
    },
    "mafic_content_hint": {
        "low":      "Low mafic content consistent with near-pure anorthosite.",
        "moderate": "Moderate mafic content suggests norite or troctolite component.",
        "high":     "High mafic content indicates significant pyroxene or olivine admixture.",
    },
    "clast_angularity": {
        "angular":    "Angular clasts indicate minimal transport; likely proximal to source.",
        "subangular": "Subangular clasts suggest moderate transport distance.",
        "rounded":    "Rounded clasts indicate significant abrasion during transport.",
    },
    "sorting": {
        "well_sorted":   "Well-sorted clasts reflect high-energy, sustained transport.",
        "poorly_sorted": "Poor sorting typical of mass-wasting or impact ejecta deposits.",
        "unsorted":      "Unsorted matrix characteristic of debris flows or impact melt breccias.",
    },
    "support_fabric": {
        "clast_supported":  "Clast-supported fabric with clasts in direct contact.",
        "matrix_supported": "Matrix-supported fabric; clasts float within finer-grained material.",
    },
}

DISPLAY_FEATURES = {
    "groundmass_texture", "vesicularity", "luster",
    "crystal_fabric", "brightness", "surface_character",
    "clast_angularity", "sorting",
}


def _tier(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    if conf >= 0.60:
        return "medium"
    if conf >= 0.45:
        return "low"
    return "uncertain"


def _random_scores(classes: list[str], winner_idx: int) -> dict:
    raw = [random.uniform(0.01, 0.15) for _ in classes]
    raw[winner_idx] = random.uniform(0.50, 0.95)
    total = sum(raw)
    return {c: round(raw[i] / total, 4) for i, c in enumerate(classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    time.sleep(1.5)

    family = random.choice(FAMILIES)
    winner_idx = ALL_CLASSES.index(family)
    primary_scores = _random_scores(ALL_CLASSES, winner_idx)
    primary_conf = primary_scores[family]
    primary_tier = _tier(primary_conf)

    # Build features (null when family is "other" or tier is "uncertain")
    features = None
    geology_notes = []
    suppressed = []

    if primary_tier != "uncertain":
        features = {}
        feat_map = FEATURE_CLASSES[family]
        for feat_name, classes in feat_map.items():
            winner = random.randint(0, len(classes) - 1)
            feat_scores = _random_scores(classes, winner)
            top_class = classes[winner]
            feat_conf = feat_scores[top_class]
            feat_tier = _tier(feat_conf)
            display = feat_name in DISPLAY_FEATURES
            features[feat_name] = {
                "value":      top_class if feat_tier != "uncertain" else "uncertain",
                "confidence": round(feat_conf, 4),
                "tier":       feat_tier,
                "display":    display,
                "scores":     feat_scores,
            }
            if display and feat_tier in ("high", "medium") and feat_tier != "uncertain":
                note_text = GEOLOGY_NOTES.get(feat_name, {}).get(top_class, "")
                if note_text:
                    geology_notes.append({
                        "feature": feat_name,
                        "value":   top_class,
                        "note":    note_text,
                    })
            elif feat_tier == "uncertain":
                suppressed.append(feat_name)

    import time as _t
    output = {
        "schema_version": "rocknet_v2.0",
        "image_path":     args.image,
        "inference_ms":   round(random.uniform(80, 250), 1),
        "primary": {
            "family":     family,
            "confidence": round(primary_conf, 4),
            "tier":       primary_tier,
            "scores":     primary_scores,
        },
        "features":      features,
        "geology_notes": geology_notes,
        "ui": {
            "show_primary":        True,
            "show_features":       features is not None,
            "suppressed_features": suppressed,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
