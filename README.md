# VAE-SA: coarse-grid-informed training-event selection

Core Python modules accompanying *A latent space informed framework for data-efficient spatiotemporal flood surrogate modeling* (Under review).

This release provides the FloodVAE architecture and loss, candidate-event encoding, and multi-start simulated-annealing selection. The interfaces are designed for application-specific hydrodynamic datasets rather than a fixed study area or server directory. The example below includes a small executable training recipe using synthetic data.

## Data availability and scope

This repository provides the core VAE-SA modules and a runnable synthetic example. Study-area datasets are not distributed with the code.

For new applications, users should prepare coarse-grid hydrodynamic simulations, domain masks, event attributes, and normalization statistics for their own study areas. Candidate events should cover the intended prediction conditions, including relevant extremes.

The synthetic example demonstrates the workflow. Reproducing the manuscript experiments requires the corresponding datasets and experimental settings.

## Files and dependencies

| File | Purpose |
| --- | --- |
| `vae_network.py` | FloodVAE network, posterior-mean encoder, masked MAE plus KL loss |
| `encoder_events.py` | Candidate normalization utilities and deterministic posterior-mean encoding |
| `optimal_selection_VAE_SA.py` | Temporal descriptors, standardization, cosine distances, attribute-balanced selection and run logs |

Use Python 3.10 or later, NumPy, and PyTorch. Only NumPy is needed for selection from precomputed latent sequences. Install a PyTorch build appropriate for your CPU/GPU environment. For a CPU example in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch
python vae_network.py
```

The synthetic workflow was checked on CPU with Python 3.12.14, NumPy 2.3.5, and PyTorch 2.15.0. These are validation-environment versions, not a requirement to use the same hardware or the manuscript training environment.

Record dependency versions and the repository revision for your experiments. Encoding metadata records NumPy and PyTorch versions, and selection summaries record NumPy versions and all selection arguments. Identical seeds do not guarantee bitwise-identical GPU results across environments.

## Input contract

Run each pool separately for a study area, hydraulic variable (water depth or scalar flow speed), and coarse-grid scale. Do not mix latent spaces from independently trained FloodVAE models.

Supply a UTF-8 CSV with these columns:

```csv
event_id,path,duration,magnitude,pattern,is_extreme
E001,maps/event_001.npy,4h,M1,P1,0
E002,maps/event_002.npy,4h,M3,P2,1
```

This two-row fragment illustrates the schema, not a complete selection pool.

- `event_id`: unique, stable identifier. Arbitrary original filenames can be used without parsing them for attributes.
- `path`: NumPy file path, relative to the CSV location or absolute. For encoding it points to a physical coarse-grid sequence `[T,H,W]`. For selection it points to a posterior-mean sequence `[T,Z]` produced by the encoder.
- `duration`: duration category. Use consistent strings, e.g. do not mix `4`, `4.0`, and `4h` for one category.
- `magnitude`: category used by the magnitude-balance penalty. Labels are not automatically sorted by physical severity or merged. Define them explicitly and consistently for your application.
- `pattern`: temporal-pattern category.
- `is_extreme`: `1` for events in the required highest-magnitude category, otherwise `0`. Every selected subset must contain at least one marked event. The code does not infer extremes from label spelling.

The manifest explicitly lists the candidate pool. Independent test events and unselected fine-grid outputs must not enter this pool or its normalization. No event is silently excluded based on a filename.

Coarse-grid arrays must contain finite, nonnegative physical values within the valid domain. A binary mask `[H,W]` identifies valid cells. Exterior NoData is replaced with zero before encoding. All events for one model must use the same grid shape and alignment. The convolutional architecture requires both dimensions to be multiples of 16. If padding is necessary, prepare it consistently in training and encoding and mark padded cells invalid. Temporal lengths may differ. Frames must be ordered chronologically and sampled consistently within each event.

### Normalization and training

FloodVAE uses `log1p` followed by min-max scaling. Estimate statistics from the coarse-grid candidate snapshots used for VAE training, within the valid domain. Save them as:

```json
{"log1p_min": 0.0, "log1p_scale": 2.0}
```

The numeric values above are illustrative. `log1p_scale` is **log-space maximum minus minimum**, not the physical maximum. `fit_norm_stats` computes these quantities from a supplied manifest. Use the same snapshot sampling as VAE training when fitting them. Scaling is fitted before the internal sample split in the manuscript implementation. Reuse the saved statistics for encoding and testing rather than estimating test-specific statistics.

The supplied normalization function clips transformed inputs to `[0,1]`, matching the supplied original encoder. Retain the same convention in training. Values outside the training range may therefore saturate. For physical-unit evaluation use `inverse_normalize`, apply the domain mask, compute per-frame metrics, and then aggregate using the stated event-level convention.

Train a separate VAE for each hydraulic variable and scale. Use `FloodVAE`, `beta_vae_loss`, and the training pattern shown below with your own dataset loader. Production training should stream large datasets rather than loading the entire pool into RAM. Save either `model.state_dict()` or a dictionary containing `model_state_dict`. The encoder loads state-dictionary checkpoints with strict architectural matching. Supply the same `base_ch`, `z_dim`, input shape, normalization, and mask used in training.

The core architecture and state-dictionary layer names are retained from the supplied network. Historical checkpoints are usable only when those settings match. This release does not distribute historical weights.

## Method and configurable defaults

Each event is represented by the temporal mean and standard deviation of its posterior-mean vectors. The initial frame is excluded by default, matching the supplied selection script. If the data already omit the initial dry frame, use `--keep-initial-frame` to avoid removing another frame. At least two frames must remain. Standard deviations use `ddof=0`.

The resulting `2Z`-dimensional descriptors are standardized feature by feature over the whole candidate pool. The current implementation stores a dense candidate-to-candidate distance matrix, requiring memory proportional to the square of the candidate count. Cosine distances are calculated in that space. Selection minimizes:

```text
J(S) = mean_i min_{j in S} cosine_distance(f_i, f_j)
       + magnitude_weight * sum_g (n_g(S)/k - 1/G_magnitude)^2
       + pattern_weight   * sum_g (n_g(S)/k - 1/G_pattern)^2
```

The two balance terms are soft attribute constraints, not hydraulic conservation equations. Equal event counts across duration categories and inclusion of at least one extreme event are hard constraints. The subset size must be divisible by the number of duration categories. Every category must have enough candidates to satisfy its quota. Simulated annealing preserves these constraints through within-duration exchanges and retains the best subset encountered across starts. It is a heuristic search and does not certify a global optimum.

| Setting | Manuscript setting / release default |
| --- | --- |
| Latent dimension | 64 by default, 48 and 32 in the manuscript's SF=16 and SF=32 analyses |
| Network base channels | 32 |
| VAE loss | Valid-domain MAE plus beta-weighted KL, using the supplied loss reduction |
| VAE maximum KL weight | `5e-4` loss default; increase gradually over the first 100 training epochs |
| Reference VAE training | AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 128, maximum 500 epochs, internal split 9:1 |
| Magnitude / pattern weights | 0.5 / 0.3 |
| Duration quota | Equal counts across manifest duration categories |
| Extreme inclusion | At least one event marked `is_extreme=1` |
| SA temperature | 1.0 to `1e-6`, geometric cooling |
| SA steps / starts | 10,000 / 20 |
| SA base seed | 42; successive starts use `seed + 10007 * start_index` |

Defaults are starting points, not universally optimal settings. Adapt sampling, architecture dimensions, batch size, optimization settings, category definitions, and selection weights using candidate/training data and internal validation appropriate to your application. Keep independent test events separate from these decisions. The obsolete redundancy term (`gamma`, `tau`), unused duration penalty, alternate descriptor modes, and legacy parameter aliases have been removed from this release.

Magnitude categories are used **exactly as supplied in the manifest**, following the manuscript's category-based objective. There is no automatic merging of Carlisle-specific labels into low/middle/high groups. A user-defined grouping must be reflected explicitly in the manifest and documented. This generalized release is not a claim of bitwise reproduction of historical filename-based runs.

## Encode and select with your own data

```bash
python encoder_events.py \
  --manifest data/candidates.csv \
  --checkpoint models/floodvae.pth \
  --norm-json models/normalization.json \
  --mask data/domain_mask.npy \
  --output-dir runs/encoded \
  --z-dim 64 --base-ch 32 --device cpu

python optimal_selection_VAE_SA.py \
  --manifest runs/encoded/latent_manifest.csv \
  --output-dir runs/selection_k3 \
  --subset-k 3
```

Use new or empty output directories to avoid mixing pools, seeds, variables, or checkpoints. Encoding saves all time steps. Initial-frame removal occurs only during descriptor construction in the selection script.

Selection outputs:

- `selected_events.csv`: selected stable event IDs, attributes, and latent-array paths. Match IDs to your own forcing inputs to generate fine-grid labels. These paths are not fine-grid labels.
- `selection_summary.json`: configuration, manifest hash, selected category counts, best objective and all start summaries.
- `best_start_history.json`: current and best objective values along the winning start.
- `descriptor_scaling.npz`: candidate-pool descriptor means and scales.

Depth and speed are separate selection tasks and may yield different subsets. If developing both surrogates jointly, account for unique hydrodynamic runs because one run can produce both outputs. No fine-grid simulation saving or target-accuracy guarantee follows from running the selector alone.

## Small synthetic execution example

Run this block from the directory containing the three modules. It creates nine artificial events, performs two demonstration epochs, and saves a small checkpoint. These reduced settings are for a CPU smoke test, not the manuscript configuration. It uses no real terrain or forcing data. The example creates a new `demo` directory and refuses to overwrite an existing one.

```python
# SYNTHETIC_EXAMPLE_BEGIN
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from vae_network import FloodVAE, beta_vae_loss
from encoder_events import fit_norm_stats, normalize_log1p_maxmin
from optimal_selection_VAE_SA import read_manifest, write_manifest

root = Path("demo")
root.mkdir(exist_ok=False)
(root / "maps").mkdir()
rng = np.random.default_rng(42)
torch.manual_seed(42)
torch.set_num_threads(1)
mask = np.ones((32, 32), dtype=bool)
mask[:2, :] = False
np.save(root / "mask.npy", mask)
y, x = np.mgrid[0:32, 0:32]
rows = []
for d in range(3):
    for m in range(3):
        i = 3*d + m
        spatial = np.exp(-((x-(7+6*m))**2 + (y-(8+5*d))**2) / 65)
        temporal = np.array([0, .4, 1, .6, .15], dtype=np.float32)
        raw = temporal[:, None, None] * spatial * (m+1)
        raw += rng.uniform(0, .01, raw.shape) * temporal[:, None, None]
        raw[:, ~mask] = 0
        name = f"maps/event_{i}.npy"
        np.save(root / name, raw.astype(np.float32))
        rows.append(dict(event_id=f"E{i:03d}", path=name, duration=f"D{d}",
                         magnitude=f"M{m}", pattern=f"P{(d+m)%2}", is_extreme=str(int(m == 2))))
write_manifest(root / "candidates.csv", rows)
rows = read_manifest(root / "candidates.csv")
stats = fit_norm_stats(rows, mask)
(root / "norm.json").write_text(json.dumps(stats), encoding="utf-8")
frames = np.concatenate([normalize_log1p_maxmin(np.load(r["path"]), stats, mask) for r in rows])
dataset = TensorDataset(torch.from_numpy(frames[:, None]))
n_train = int(.9 * len(dataset))
train, val = random_split(dataset, [n_train, len(dataset)-n_train],
                          generator=torch.Generator().manual_seed(42))
loader = DataLoader(train, batch_size=8, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
model = FloodVAE(base_ch=4, z_dim=8, input_hw=mask.shape, basin_mask=mask)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
best_validation = float("inf")
for epoch in range(2):
    beta = 5e-4 * min((epoch+1)/100, 1.0)
    model.train()
    for (batch,) in loader:
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(batch)
        loss, _, _ = beta_vae_loss(batch, reconstruction, mu, logvar,
                                  beta=beta, basin_mask=mask)
        loss.backward()
        optimizer.step()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for (batch,) in DataLoader(val, batch_size=8):
            reconstruction, mu, logvar = model(batch)
            loss, _, _ = beta_vae_loss(batch, reconstruction, mu, logvar,
                                      beta=beta, basin_mask=mask)
            total += loss.item() * len(batch)
    score = total / len(val)
    if score < best_validation:
        best_validation = score
        torch.save({"model_state_dict": model.state_dict()}, root / "model.pth")
print("Synthetic training finished. Next run the encoder and selector below.")
# SYNTHETIC_EXAMPLE_END
```

Then run:

```bash
python encoder_events.py --manifest demo/candidates.csv \
  --checkpoint demo/model.pth --norm-json demo/norm.json --mask demo/mask.npy \
  --output-dir demo/encoded --base-ch 4 --z-dim 8 --device cpu

python optimal_selection_VAE_SA.py --manifest demo/encoded/latent_manifest.csv \
  --output-dir demo/selection --subset-k 3 --starts 2 --steps 200
```

The selected subset must contain three different event IDs, one from each duration category, with at least one extreme event. Increase the search settings and use properly trained representations for scientific applications. Keep a record of preprocessing, sampling, category definitions, configurations, seeds, and code revisions for any reported results.

## Downstream flood surrogate and citation

RSwinUNet is the downstream surrogate used to evaluate selected event sets in the manuscript. Its architecture code is available separately in the repository associated with:

Song, W., & Guan, M. (2026). Enhancing cross-regional transferability of super-resolution-based flood surrogate models for data-scarce catchments. *Water Research, 298*, 125799. https://doi.org/10.1016/j.watres.2026.125799

[RSwinUNet source repository](https://github.com/songwk0924/Residual-SwinUNet-for-Flood-Surrogate-Modeling)

This citation identifies the downstream model. The RSwinUNet repository is not a substitute for the VAE-SA selection modules released here. Prepare selected-event coarse-grid inputs, static features, and fine-grid labels for your own surrogate workflow. Estimate surrogate dynamic-input and fine-label normalization from the selected subset, and reuse those statistics for independent testing.

Please also cite the VAE-SA manuscript when using this method, using its final publication details once available. A license should be chosen by the rights holder before the repository is presented as a licensed open-source release.
