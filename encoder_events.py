"""Encode candidate coarse-grid event sequences using a trained FloodVAE.

Use the normalization statistics and domain mask used during VAE training.
Input arrays contain nonnegative physical values with shape [T, H, W].
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from optimal_selection_VAE_SA import read_manifest, write_manifest
from vae_network import FloodVAE


def load_mask(path):
    mask = np.load(path, allow_pickle=False)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("Mask must be a nonempty binary array [H,W].")
    return mask.astype(bool)


def validate_event(array, mask):
    if array.ndim != 3 or array.shape[1:] != mask.shape or array.shape[0] < 1:
        raise ValueError(f"Expected nonempty [T,{mask.shape[0]},{mask.shape[1]}], got {array.shape}.")
    # Domain-exterior NoData values are ignored and replaced with zero.
    for frame in array:
        values = frame[mask]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Valid-domain physical values must be finite and nonnegative.")


def fit_norm_stats(rows, mask):
    """Fit log1p min-max statistics on candidate training snapshots only.

    Supply the same snapshot sampling used for VAE training. Do not include
    independent test events or fine-grid labels. This routine streams arrays.
    """
    low, high = np.inf, -np.inf
    for row in rows:
        array = np.load(row["path"], mmap_mode="r", allow_pickle=False)
        validate_event(array, mask)
        for frame in array:
            values = np.log1p(np.asarray(frame[mask], dtype=np.float64))
            low, high = min(low, float(values.min())), max(high, float(values.max()))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Candidate snapshots must have a nonzero finite log1p range.")
    return {"log1p_min": low, "log1p_scale": high-low}


def load_norm_stats(path):
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    values = [float(stats[k]) for k in ("log1p_min", "log1p_scale")]
    if not np.isfinite(values).all() or values[1] <= 0:
        raise ValueError("Normalization statistics must be finite with a positive scale.")
    return dict(zip(("log1p_min", "log1p_scale"), values))


def normalize_log1p_maxmin(raw, stats, mask):
    """Apply training-time scaling and clipping, then zero the domain exterior."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-2:] != mask.shape:
        raise ValueError("Data and mask shapes differ.")
    valid = raw[..., mask]
    if not np.isfinite(valid).all() or np.any(valid < 0):
        raise ValueError("Valid-domain values must be finite and nonnegative.")
    out = np.zeros_like(raw, dtype=np.float32)
    out[..., mask] = np.clip((np.log1p(valid) - stats["log1p_min"]) / stats["log1p_scale"], 0, 1)
    return out


def inverse_normalize(normalized, stats):
    """Restore physical units before reconstruction-error evaluation."""
    return np.expm1(np.asarray(normalized) * stats["log1p_scale"] + stats["log1p_min"])


def build_model(checkpoint, mask, z_dim, base_ch, device):
    model = FloodVAE(base_ch=base_ch, z_dim=z_dim, input_hw=mask.shape, basin_mask=mask).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    state = saved.get("model_state_dict", saved)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "basin_mask" in state:
        supplied = torch.as_tensor(mask, device=device).bool()
        stored = state["basin_mask"].squeeze().bool()
        if stored.shape != supplied.shape or not torch.equal(stored, supplied):
            raise ValueError("Checkpoint mask differs from the supplied mask.")
    else:
        # The original network optionally omitted a mask buffer from its state.
        state["basin_mask"] = model.basin_mask
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def encode_file_mu(model, array, stats, mask, device, batch_size=128):
    validate_event(array, mask)
    chunks = []
    for start in range(0, len(array), batch_size):
        x = normalize_log1p_maxmin(array[start:start+batch_size], stats, mask)
        x = torch.from_numpy(x[:, None]).to(device)
        chunks.append(model.encode_mu(x).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-json", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--z-dim", type=int, default=64, help="Must match the trained checkpoint.")
    parser.add_argument("--base-ch", type=int, default=32, help="Must match the trained checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty. Use a separate directory for each model.")
    rows = read_manifest(args.manifest)
    mask, stats = load_mask(args.mask), load_norm_stats(args.norm_json)
    model = build_model(args.checkpoint, mask, args.z_dim, args.base_ch, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded_rows = []
    for i, row in enumerate(rows):
        array = np.load(row["path"], mmap_mode="r", allow_pickle=False)
        mu = encode_file_mu(model, array, stats, mask, args.device, args.batch_size)
        name = f"event_{i:06d}.npy"
        np.save(args.output_dir / name, mu)
        encoded_rows.append({**row, "path": name})
        print(f"Encoded {row['event_id']}: {mu.shape}")
    write_manifest(args.output_dir / "latent_manifest.csv", encoded_rows)
    meta = {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "z_dim": args.z_dim, "base_ch": args.base_ch, "input_hw": list(mask.shape),
            "batch_size": args.batch_size, "device": args.device, "normalization": stats,
            "candidate_count": len(rows), "encoded_all_frames": True,
            "file_sha256": {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
                            for key in ("manifest", "checkpoint", "norm_json", "mask")}}
    (args.output_dir / "encoding_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
