#!/usr/bin/env python3
"""
STEP 3: Fine-tune stt_en_fastconformer_ctc_large on your geology corpus.

IMPORTANT: Run this on Google Colab (GPU), NOT on your Mac or Jetson.
  - Go to colab.research.google.com
  - Runtime → Change runtime type → T4 GPU (free) or A100 (Colab Pro)
  - Upload this script + your manifests, then run it

Strategy:
  Phase 1 (epochs 1-3):  Freeze encoder, train decoder+output only
                         → learns your vocabulary without forgetting English
  Phase 2 (epochs 4-10): Unfreeze encoder, fine-tune everything at low LR
                         → adapts acoustics to your speakers/recording style

Usage:
    python 3_finetune_nemo.py \
        --train_manifest ./data/train_manifest_train.json \
        --val_manifest   ./data/train_manifest_val.json \
        --output_dir     ./checkpoints \
        --experiment_name geology_fastconformer_v1
"""

import argparse
import os
import sys
from pathlib import Path


def check_environment():
    """Verify GPU and NeMo are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: No CUDA GPU detected. Fine-tuning on CPU will be "
                  "extremely slow. Are you running on Colab with GPU enabled?")
            print("  Runtime → Change runtime type → T4 GPU")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB)")
    except ImportError:
        print("ERROR: torch not installed")
        sys.exit(1)

    try:
        import nemo
        print(f"NeMo version: {nemo.__version__}")
    except ImportError:
        print("ERROR: NeMo not installed. On Colab run:")
        print("  !pip install nemo_toolkit['asr']")
        sys.exit(1)


def load_model(freeze_encoder: bool = True):
    """Download and prepare the base FastConformer model."""
    import nemo.collections.asr as nemo_asr
    import torch

    print("\nLoading stt_en_fastconformer_ctc_large...")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_fastconformer_ctc_large"
    )

    if freeze_encoder:
        print("Freezing encoder (Phase 1 training)")
        model.encoder.freeze()
        model.encoder.eval()
        # Only decoder + output projection stay trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")
    else:
        print("All parameters trainable (Phase 2 training)")
        model.encoder.unfreeze()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

    return model


def make_trainer_config(
    train_manifest: str,
    val_manifest:   str,
    output_dir:     str,
    experiment_name: str,
    max_epochs:     int,
    learning_rate:  float,
    batch_size:     int = 8,
):
    """Build the NeMo trainer + data config dicts."""
    import torch

    # Detect GPU memory to set batch size automatically
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem_gb < 12:      # T4 (15GB) but being conservative
            batch_size = 8
        elif gpu_mem_gb < 40:    # A100 40GB
            batch_size = 16
        else:                    # A100 80GB
            batch_size = 24
        print(f"Auto batch_size: {batch_size} (GPU: {gpu_mem_gb:.0f}GB)")

    train_config = {
        "manifest_filepath": train_manifest,
        "sample_rate":       16000,
        "batch_size":        batch_size,
        "shuffle":           True,
        "num_workers":       4,
        "pin_memory":        True,
        # Augmentation — helps generalise across recording conditions
        "augmentor": {
            "speed":   {"prob": 0.3, "sr": 16000,
                        "resample_type": "kaiser_fast",
                        "min_speed_rate": 0.9, "max_speed_rate": 1.1}
        },
    }

    val_config = {
        "manifest_filepath": val_manifest,
        "sample_rate":       16000,
        "batch_size":        batch_size,
        "shuffle":           False,
        "num_workers":       4,
    }

    optim_config = {
        "name":          "adamw",
        "lr":            learning_rate,
        "weight_decay":  1e-3,
        "betas":         [0.9, 0.98],
        "sched": {
            "name":             "CosineAnnealing",
            "warmup_steps":     500,
            "warmup_ratio":     None,
            "min_lr":           learning_rate * 0.05,
            "last_epoch":       -1,
        },
    }

    return train_config, val_config, optim_config


def run_training_phase(
    model,
    train_manifest: str,
    val_manifest:   str,
    output_dir:     str,
    experiment_name: str,
    max_epochs:     int,
    learning_rate:  float,
    phase:          int,
):
    """Run one training phase (frozen or unfrozen encoder)."""
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from lightning.pytorch.loggers import TensorBoardLogger
    except ImportError:
        print("ERROR: lightning not installed")
        sys.exit(1)

    from omegaconf import OmegaConf, DictConfig

    train_cfg, val_cfg, optim_cfg = make_trainer_config(
        train_manifest, val_manifest, output_dir, experiment_name,
        max_epochs, learning_rate,
    )

    # Inject configs into the model
    model.setup_training_data(DictConfig(train_cfg))
    model.setup_validation_data(DictConfig(val_cfg))
    model.setup_optimization(DictConfig(optim_cfg))

    phase_dir = os.path.join(output_dir, f"phase{phase}")
    os.makedirs(phase_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=phase_dir,
        filename=f"{experiment_name}_phase{phase}_{{epoch:02d}}_{{val_wer:.3f}}",
        monitor="val_wer",
        mode="min",
        save_top_k=2,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_wer",
        patience=3,
        mode="min",
    )

    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=f"{experiment_name}_phase{phase}",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision=16,           # mixed precision — faster + fits in GPU memory
        gradient_clip_val=1.0,  # prevent exploding gradients
        accumulate_grad_batches=2,  # effective batch = batch_size * 2
        log_every_n_steps=10,
        val_check_interval=0.5, # validate twice per epoch
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        enable_progress_bar=True,
    )

    print(f"\n{'='*60}")
    print(f"Phase {phase} training: {max_epochs} epochs, lr={learning_rate}")
    print(f"Checkpoints → {phase_dir}")
    print(f"{'='*60}\n")

    trainer.fit(model)

    # Return path to best checkpoint
    best_ckpt = checkpoint_callback.best_model_path
    print(f"\nPhase {phase} complete. Best checkpoint: {best_ckpt}")
    return best_ckpt, model


def export_nemo(model, output_path: str):
    """Save the final model as a .nemo file."""
    model.save_to(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n✓ Model saved: {output_path}  ({size_mb:.0f} MB)")
    print(f"\nTo use in your app, copy this file to:")
    print(f"  led-display/voiceNotes/newest_model.nemo")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune FastConformer on geology corpus (run on Colab GPU)")
    parser.add_argument("--train_manifest",   required=True)
    parser.add_argument("--val_manifest",     required=True)
    parser.add_argument("--output_dir",       default="./checkpoints")
    parser.add_argument("--experiment_name",  default="geology_fastconformer_v1")
    parser.add_argument("--phase1_epochs",    type=int, default=3,
                        help="Epochs with frozen encoder")
    parser.add_argument("--phase2_epochs",    type=int, default=7,
                        help="Epochs with full model unfrozen")
    parser.add_argument("--phase1_lr",        type=float, default=5e-4)
    parser.add_argument("--phase2_lr",        type=float, default=5e-5,
                        help="Should be ~10x lower than phase1")
    parser.add_argument("--skip_phase1",      action="store_true",
                        help="Skip to phase 2 (if you already ran phase 1)")
    parser.add_argument("--phase1_checkpoint", default=None,
                        help="Path to phase 1 .ckpt to resume from for phase 2")
    args = parser.parse_args()

    check_environment()

    os.makedirs(args.output_dir, exist_ok=True)
    final_nemo = os.path.join(args.output_dir,
                              f"{args.experiment_name}_final.nemo")

    # ── Phase 1: Frozen encoder ────────────────────────────────────────────
    if not args.skip_phase1:
        model = load_model(freeze_encoder=True)
        best_p1_ckpt, model = run_training_phase(
            model,
            train_manifest   = args.train_manifest,
            val_manifest     = args.val_manifest,
            output_dir       = args.output_dir,
            experiment_name  = args.experiment_name,
            max_epochs       = args.phase1_epochs,
            learning_rate    = args.phase1_lr,
            phase            = 1,
        )
        print(f"\nPhase 1 complete. Best: {best_p1_ckpt}")
        # Save intermediate .nemo
        mid_nemo = os.path.join(args.output_dir,
                                f"{args.experiment_name}_phase1.nemo")
        export_nemo(model, mid_nemo)
    else:
        # Load from a phase 1 checkpoint to continue
        import nemo.collections.asr as nemo_asr
        if args.phase1_checkpoint:
            print(f"Loading phase 1 checkpoint: {args.phase1_checkpoint}")
            model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(
                args.phase1_checkpoint
            )
        else:
            model = load_model(freeze_encoder=False)

    # ── Phase 2: Full model unfrozen ───────────────────────────────────────
    print("\nUnfreezing encoder for Phase 2...")
    model.encoder.unfreeze()
    model.train()

    _, model = run_training_phase(
        model,
        train_manifest   = args.train_manifest,
        val_manifest     = args.val_manifest,
        output_dir       = args.output_dir,
        experiment_name  = args.experiment_name,
        max_epochs       = args.phase2_epochs,
        learning_rate    = args.phase2_lr,
        phase            = 2,
    )

    # ── Export final model ─────────────────────────────────────────────────
    export_nemo(model, final_nemo)

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    print(f"Final model: {final_nemo}")
    print("\nEvaluate on a held-out test set before deploying:")
    print("  python -c \"")
    print("  import nemo.collections.asr as nemo_asr")
    print(f"  m = nemo_asr.models.EncDecCTCModelBPE.restore_from('{final_nemo}')")
    print("  wer = m.transcribe(['test.wav'])")
    print("  print(wer)\"")


if __name__ == "__main__":
    main()
