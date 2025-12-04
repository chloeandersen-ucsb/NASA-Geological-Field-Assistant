#!/usr/bin/env python3
"""
Configuration module for 3D generation pipelines.

Provides centralized configuration management with support for:
- Default presets (fast, balanced, high_quality)
- YAML config files
- Command-line override

Usage:
    from config import Config
    
    # Load defaults
    cfg = Config()
    
    # Load from file
    cfg = Config.from_yaml('my_config.yaml')
    
    # Access settings
    print(cfg.video.fps)
    print(cfg.reconstruction.backend)
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class VideoExtractionConfig:
    """Video frame extraction parameters."""
    fps: Optional[float] = None
    every_n_frames: Optional[int] = None
    max_frames: int = 20
    
    # Quality filtering
    filter_quality: bool = True
    filter_similarity: bool = True
    blur_threshold: float = 100.0
    exposure_min: float = 30
    exposure_max: float = 220
    overexposure_threshold: float = 0.1
    underexposure_threshold: float = 0.1
    motion_threshold: Optional[float] = None
    similarity_threshold: float = 0.85
    
    # Output
    save_format: str = 'jpg'
    save_quality: int = 95


@dataclass
class ReconstructionConfig:
    """3D reconstruction parameters."""
    backend: str = 'mast3r'  # 'colmap', 'dust3r', 'mast3r'
    device: str = 'cpu'  # 'cpu' or 'cuda'
    voxel_size: Optional[float] = 0.05
    
    # COLMAP-specific
    colmap_image_size: Optional[int] = None
    
    # DUSt3R/MASt3R-specific
    dust3r_image_size: int = 512
    dust3r_batch_size: int = 1


@dataclass
class Config:
    """Main configuration class."""
    video: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    
    # General
    output_base: str = 'outputs'
    keep_frames: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            video=VideoExtractionConfig(**data.get('video', {})),
            reconstruction=ReconstructionConfig(**data.get('reconstruction', {})),
            output_base=data.get('output_base', 'outputs'),
            keep_frames=data.get('keep_frames', False)
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            'video': asdict(self.video),
            'reconstruction': asdict(self.reconstruction),
            'output_base': self.output_base,
            'keep_frames': self.keep_frames
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def preset_fast(cls) -> 'Config':
        """Fast preview preset - minimal frames, DUSt3R."""
        cfg = cls()
        cfg.video.max_frames = 10
        cfg.video.blur_threshold = 80.0
        cfg.video.filter_similarity = True
        cfg.reconstruction.backend = 'dust3r'
        cfg.reconstruction.voxel_size = 0.1
        return cfg
    
    @classmethod
    def preset_balanced(cls) -> 'Config':
        """Balanced preset - moderate quality, MASt3R."""
        cfg = cls()
        cfg.video.max_frames = 20
        cfg.video.blur_threshold = 100.0
        cfg.video.filter_quality = True
        cfg.video.filter_similarity = True
        cfg.reconstruction.backend = 'mast3r'
        cfg.reconstruction.voxel_size = 0.05
        return cfg
    
    @classmethod
    def preset_high_quality(cls) -> 'Config':
        """High quality preset - many frames, COLMAP."""
        cfg = cls()
        cfg.video.max_frames = 50
        cfg.video.blur_threshold = 150.0
        cfg.video.filter_quality = True
        cfg.video.filter_similarity = True
        cfg.video.similarity_threshold = 0.90
        cfg.reconstruction.backend = 'colmap'
        cfg.reconstruction.voxel_size = 0.02
        cfg.keep_frames = True
        return cfg
    
    def apply_overrides(self, **kwargs):
        """Apply command-line overrides to config."""
        for key, value in kwargs.items():
            if value is None:
                continue
            
            # Navigate nested attributes
            if '.' in key:
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)


# Preset configurations
PRESETS = {
    'fast': Config.preset_fast,
    'balanced': Config.preset_balanced,
    'high_quality': Config.preset_high_quality
}


def load_config(
    preset: Optional[str] = None,
    config_file: Optional[str] = None,
    **overrides
) -> Config:
    """
    Load configuration with optional preset, file, and overrides.
    
    Priority: overrides > config_file > preset > defaults
    
    Args:
        preset: preset name ('fast', 'balanced', 'high_quality')
        config_file: path to YAML config file
        **overrides: command-line overrides
    
    Returns:
        Config object
    """
    # Start with preset or defaults
    if preset and preset in PRESETS:
        cfg = PRESETS[preset]()
    else:
        cfg = Config()
    
    # Load from file if provided
    if config_file and Path(config_file).exists():
        file_cfg = Config.from_yaml(config_file)
        # Merge file config into preset
        cfg.video = file_cfg.video
        cfg.reconstruction = file_cfg.reconstruction
        cfg.output_base = file_cfg.output_base
        cfg.keep_frames = file_cfg.keep_frames
    
    # Apply overrides
    cfg.apply_overrides(**overrides)
    
    return cfg


if __name__ == '__main__':
    # Example: Generate sample config files
    import sys
    
    print("=" * 60)
    print("Configuration File Generator")
    print("=" * 60)
    print()
    
    # Generate preset configs
    print("Generating sample configuration files...\n")
    
    for preset_name, preset_func in PRESETS.items():
        cfg = preset_func()
        output_file = f"config_{preset_name}.yaml"
        cfg.to_yaml(output_file)
        print(f"✓ Created: {output_file}")
        
        # Print summary
        print(f"  Preset: {preset_name}")
        print(f"    Frames: {cfg.video.max_frames}")
        print(f"    Backend: {cfg.reconstruction.backend}")
        print(f"    Blur threshold: {cfg.video.blur_threshold}")
        print(f"    Quality filter: {cfg.video.filter_quality}")
        print()
    
    print("=" * 60)
    print("Usage Examples:")
    print("=" * 60)
    print()
    print("1. Use a preset config:")
    print("   python scripts/generate_scene_from_video.py video.mp4 \\")
    print("       --config config_fast.yaml")
    print()
    print("2. Override specific settings:")
    print("   python scripts/generate_scene_from_video.py video.mp4 \\")
    print("       --config config_balanced.yaml \\")
    print("       --max-frames 30 \\")
    print("       --device cuda")
    print()
    print("3. Create custom config:")
    print("   - Edit any generated YAML file")
    print("   - Adjust parameters to your needs")
    print("   - Save with new name")
    print()
