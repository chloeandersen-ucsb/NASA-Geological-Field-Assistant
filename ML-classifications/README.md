# ML-Classifications

This is the rock classification pipeline: takes a photo of a rock and figures out if it's **basalt**, **anorthosite**, or **breccia** (or none of the above). These 3 rock types were chosen for their prevalence on the lunar surface.

## how are images collected + pre-processed?

1. Images come from the sources listed in the training data table below
2. Raw images are organized by file and cropped to 448×448 px (OpenCV GrabCut) and added to `dataset_manifest.json`
3. Processed images are ran through Claude Vision (sonnet API) to classify geological _features_ for each rock image, writing 
predictions back to `dataset_manifest.json` with confidence scores (1-3)
4. Image labels with low confidence scores (<2) are manually reviewed  

## what's the model doing?

The model is called **RockNet** and it does classification in stages:

1. **Primary classification**: is this rock basalt, anorthosite, breccia, or "other"?
2. **Feature extraction**: depending on which rock type it is, it then looks for more specific visual traits (e.g. for basalt: vesicularity, texture, luster; for breccia: clast angularity and sorting, etc.)
3. **Geology notes**: a lookup-based explanation layer that translates those features into plain-text geological context

The backbone is a ConvNeXt-Tiny pretrained on ImageNet, with multi-scale feature fusion and a texture-aware pooling stage (GeM + Gram pooling). See `scripts/rocknet_v2.py` for the full architecture.

## classification outputs a JSON blob: 
```json
{
  "schema_version": "rocknet_v2.0",
  "image_path": "/abs/path/to/image.jpg",
  "inference_ms": 142.3,

  "primary": {
    "family":     "basalt" | "anorthosite" | "breccia" | "other",
    "confidence": 0.0–1.0,
    "tier":       "high" | "medium" | "low" | "uncertain",
    "scores": {
      "basalt":      0.0–1.0,
      "anorthosite": 0.0–1.0,
      "breccia":     0.0–1.0,
      "other":       0.0–1.0
    }
  },

  "features": null | {
    "<feature_name>": {
      "value":      "<class_string>" | "uncertain",
      "confidence": 0.0–1.0,
      "tier":       "high" | "medium" | "low" | "uncertain",
      "display":    true | false,
      "scores":     { "<class>": 0.0–1.0, ... }
    }
  },

  "geology_notes": [
    { "feature": "<name>", "value": "<class>", "note": "<explanation>" }
  ],

  "ui": {
    "show_primary":        true,
    "show_features":       true | false,
    "suppressed_features": ["<feature_name>", ...]
  }
}
```

## training data

Images were pulled from several sources with varying quality and origin (lunar vs. Earth-based). Here's the breakdown:

| Source | Basalt | Breccia | Anorthosite | Notes |
|---|---|---|---|---|
| [Lunar Sample Atlas (LPI)](https://www.lpi.usra.edu/lunar/samples/atlas/) | 600 | 600 | 90 | Lunar. High quality — weighted more heavily |
| Kaggle (4 datasets)* | 1000 | 15 | 61 | Mostly Earth-based. Weighted less — image quality is inconsistent |
| [Smithsonian](https://collections.nmnh.si.edu/search/ms/) | 200 | 17 | 2 | Earth-based |
| Flickr (James St. John, Kevin Gill, Mike Fitz) | 164 | 74 | 374 | Mostly Earth-based |
| [Astromaterials](https://ares.jsc.nasa.gov/astromaterials3d/apollo-lunar.htm) | 14 | 27 | 3 | Lunar |
| Manual (hand-collected) | 26 | 0 | 10 | Lunar + Earth |

*Kaggle datasets used: `salmaneunus/rock-classification`, `stealthtechnologies/rock-classification`, `ifeoyelakin/nasa-space-rocks`, `potesanket/rockdataset`
