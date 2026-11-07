# VAE-SA: coarse-grid-informed training-event selection

Core Python modules accompanying *A latent space informed framework for data-efficient spatiotemporal flood surrogate modeling* (Under Review).

This release provides the FloodVAE architecture and loss, candidate-event encoding, and multi-start simulated-annealing selection. Users can apply the modules to hydrodynamic datasets from their own study areas. 

## Data and application requirements

This repository provides the core VAE-SA modules and basic usage instructions. Study-area datasets are not distributed with the code.

For new applications, users should prepare coarse-grid hydrodynamic simulations, domain masks, event attributes, and normalization statistics for their own study areas. Candidate events should cover the intended prediction conditions, including relevant extremes.

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

The core modules use Python, NumPy, and PyTorch. Users should record the dependency versions, random seeds, and configuration settings used in their experiments to support reproducibility.

## Input contract

Run each pool separately for a study area, hydraulic variable (water depth or scalar flow speed), and coarse-grid scale. Do not mix latent spaces from independently trained FloodVAE models.

Supply a UTF-8 CSV with these columns:

```csv
event_id,path,duration,magnitude,pattern,is_extreme
E001,maps/event_001.npy,4h,M1,P1,0
E002,maps/event_002.npy,4h,M3,P2,1
```

This two-row fragment illustrates the schema, not a complete selection pool.

- `event_id`: unique, stable identifier. Filenames can be chosen freely because attributes are specified in the CSV.
- `path`: NumPy file path, relative to the CSV location or absolute. For encoding it points to a physical coarse-grid sequence `[T,H,W]`. For selection it points to a posterior-mean sequence `[T,Z]` produced by the encoder.
- `duration`: duration category. Use consistent strings, e.g. do not mix `4`, `4.0`, and `4h` for one category.
- `magnitude`: category used by the magnitude-balance penalty. Define category labels explicitly and consistently for your application.
- `pattern`: temporal-pattern category.
- `is_extreme`: `1` for events in the required highest-magnitude category, otherwise `0`. Every selected subset must contain at least one marked event. The code does not infer extremes from label spelling.

The manifest explicitly lists the candidate pool. Independent test events and unselected fine-grid outputs must not enter this pool or its normalization.

Coarse-grid arrays must contain finite, nonnegative physical values within the valid domain. A binary mask `[H,W]` identifies valid cells. Exterior NoData is replaced with zero before encoding. All events for one model must use the same grid shape and alignment. The convolutional architecture requires both dimensions to be multiples of 16. If padding is necessary, prepare it consistently in training and encoding and mark padded cells invalid. Temporal lengths may differ. Frames must be ordered chronologically and sampled consistently within each event.

### Normalization and training

FloodVAE uses `log1p` followed by min-max scaling. Estimate statistics from the coarse-grid candidate snapshots used for VAE training, within the valid domain. Save them as:

```json
{"log1p_min": 0.0, "log1p_scale": 2.0}
```

The values above are illustrative. `log1p_scale` is the maximum minus the minimum after log1p transformation. Use `fit_norm_stats` to estimate these statistics from the candidate snapshots used for VAE training, and reuse them for encoding and testing. Normalized inputs are clipped to [0, 1]. Restore physical units with `inverse_normalize` before evaluating reconstruction errors.

Train a separate FloodVAE for each study area, hydraulic variable, and coarse-grid scale using the provided network and loss function. Adapt the data loader and training settings to your application. Save the trained weights as `model.state_dict()` or under the key `model_state_dict`, and use matching network settings, normalization statistics, and domain masks when encoding events.

## Method and configurable defaults

Each event is represented by the temporal mean and standard deviation of its posterior-mean vectors. The initial frame is excluded by default. If the data already omit the initial dry frame, use `--keep-initial-frame` to avoid removing another frame. At least two frames must remain. Standard deviations use `ddof=0`.

The resulting `2Z`-dimensional descriptors are standardized feature by feature over the whole candidate pool. The current implementation stores a dense candidate-to-candidate distance matrix, requiring memory proportional to the square of the candidate count. Cosine distances are calculated in that space. Selection minimizes:

```text
J(S) = mean_i min_{j in S} cosine_distance(f_i, f_j)
       + magnitude_weight * sum_g (n_g(S)/k - 1/G_magnitude)^2
       + pattern_weight   * sum_g (n_g(S)/k - 1/G_pattern)^2
```

The balance terms encourage representation across magnitude and temporal-pattern categories. Duration allocation should consider the event durations and the resulting training-data volumes in each application. In this study, events were selected equally across duration categories to maintain comparable training-data volumes across selection strategies. Simulated annealing searches for event combinations with lower objective values and retains the best subset found across multiple starts.

| Setting | Manuscript setting / release default |
| --- | --- |
| Latent dimension | 64 by default, 48 and 32 in the manuscript's SF=16 and SF=32 analyses |
| Network base channels | 32 |
| VAE loss | Valid-domain MAE plus beta-weighted KL |
| VAE maximum KL weight | `5e-4` loss default; increase gradually over the first 100 training epochs |
| Reference VAE training | AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 128, maximum 500 epochs, internal split 9:1 |
| Magnitude / pattern weights | 0.5 / 0.3 |
| Allocation across duration categories | Equal numbers of selected events per duration category |
| Extreme inclusion | At least one event marked `is_extreme=1` |
| SA temperature | 1.0 to `1e-6`, geometric cooling |
| SA steps / starts | 10,000 / 20 |
| SA base seed | 42; successive starts use `seed + 10007 * start_index` |

Defaults are starting points, not universally optimal settings. Adapt sampling, architecture dimensions, batch size, optimization settings, category definitions, and selection weights using candidate/training data and internal validation appropriate to your application. Keep independent test events separate from these decisions.


## Basic usage

Train FloodVAE on your own coarse-grid candidate data, then use the saved weights and normalization statistics to encode events and select a training subset:

The current implementation uses equal duration quotas, so `--subset-k` must be divisible by the number of duration categories. Each category must contain enough candidates for its quota.

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

Depth and speed are separate selection tasks and may yield different subsets. Selected event IDs can be used to prepare the corresponding fine-grid training data.

## Downstream flood surrogate and citation

RSwinUNet is the downstream surrogate used to evaluate selected event sets in the manuscript. Its architecture code is available separately in the repository associated with:

Song, W., & Guan, M. (2026). Enhancing cross-regional transferability of super-resolution-based flood surrogate models for data-scarce catchments. *Water Research, 298*, 125799. https://doi.org/10.1016/j.watres.2026.125799

[RSwinUNet source repository](https://github.com/songwk0924/Residual-SwinUNet-for-Flood-Surrogate-Modeling)
