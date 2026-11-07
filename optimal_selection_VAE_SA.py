"""Select candidate events using latent coverage and attribute balance.

Inputs are an explicit CSV manifest and posterior-mean arrays [T, Z].
No study-area names, filename conventions, or fine-grid labels are required.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FIELDS = ("event_id", "path", "duration", "magnitude", "pattern", "is_extreme")


def read_manifest(path):
    """Read one pool for one area, hydraulic variable, and coarse-grid scale.

    Attribute values are categorical strings. Paths are relative to the CSV.
    is_extreme marks membership in the user-defined highest-magnitude category.
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns: {', '.join(FIELDS)}")
        rows = [{key: (row.get(key) or "").strip() for key in FIELDS} for row in reader]
    if not rows:
        raise ValueError("The candidate manifest is empty.")
    seen = set()
    for row in rows:
        if not all(row.values()):
            raise ValueError("Every required manifest field must have a value.")
        if row["event_id"] in seen:
            raise ValueError(f"Duplicate event_id: {row['event_id']}")
        seen.add(row["event_id"])
        if row["is_extreme"] not in ("0", "1"):
            raise ValueError("is_extreme must be 0 or 1.")
        row["path"] = str((path.parent / row["path"]).resolve())
    return rows


def write_manifest(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)


def build_event_feature(mu, drop_initial=True):
    """Concatenate temporal mean and population standard deviation (ddof=0)."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] == 0 or not np.isfinite(mu).all():
        raise ValueError("Each latent array must be finite with shape [T, Z].")
    if drop_initial:
        mu = mu[1:]
    if len(mu) < 2:
        raise ValueError("At least two encoded frames must remain after initial-frame removal.")
    return np.concatenate((mu.mean(axis=0), mu.std(axis=0, ddof=0)))


def standardized_distances(features):
    """Fit feature-wise z-scores on the candidate pool and compute cosine distances."""
    features = np.asarray(features, dtype=np.float64)
    mean, scale = features.mean(axis=0), features.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = (features - mean) / scale
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A standardized descriptor has zero norm. Cosine distance is undefined.")
    x /= norms
    distance = np.clip(1.0 - x @ x.T, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    return distance, mean, scale


def category_index(rows, field):
    labels = sorted({r[field] for r in rows})
    lookup = {label: i for i, label in enumerate(labels)}
    return labels, np.array([lookup[r[field]] for r in rows], dtype=int)


def objective(selected, distances, magnitude, pattern, magnitude_weight=0.5,
              pattern_weight=0.3):
    """Coverage + weighted squared deviations from uniform category proportions."""
    selected = np.asarray(selected, dtype=int)
    coverage = float(distances[:, selected].min(axis=1).mean())
    penalties = []
    for groups in (magnitude, pattern):
        count = int(groups.max()) + 1
        proportions = np.bincount(groups[selected], minlength=count) / len(selected)
        penalties.append(float(np.sum((proportions - 1.0 / count) ** 2)))
    return {"score": coverage + magnitude_weight * penalties[0] + pattern_weight * penalties[1],
            "coverage": coverage, "magnitude_imbalance": penalties[0],
            "pattern_imbalance": penalties[1]}


def anneal(distances, magnitude, pattern, duration, extreme, k, rng,
           magnitude_weight=0.5, pattern_weight=0.3, steps=10000,
           temperature_start=1.0, temperature_end=1e-6):
    """Perform one SA start with equal duration quotas and one extreme event.

    A proposal exchanges events only within one duration category. Groups
    with no remaining unselected candidate are skipped rather than ending SA.
    """
    groups = [np.flatnonzero(duration == g) for g in np.unique(duration)]
    if k < 1 or k > len(duration) or k % len(groups):
        raise ValueError("k must not exceed the pool and must be divisible by the duration-category count.")
    quota = k // len(groups)
    if any(len(group) < quota for group in groups):
        raise ValueError("A duration category has too few candidates for its quota.")
    if not np.any(extreme):
        raise ValueError("At least one candidate must be marked is_extreme=1.")
    if steps < 1 or not 0 < temperature_end <= temperature_start:
        raise ValueError("Require steps >= 1 and 0 < final temperature <= initial temperature.")
    if not np.isfinite([temperature_start, temperature_end, magnitude_weight, pattern_weight]).all():
        raise ValueError("Temperatures and weights must be finite.")
    if min(magnitude_weight, pattern_weight) < 0:
        raise ValueError("Balance weights must be nonnegative.")
    selected = np.concatenate([rng.choice(group, quota, replace=False) for group in groups])
    if not extreme[selected].any():
        incoming = int(rng.choice(np.flatnonzero(extreme)))
        pos = int(rng.choice(np.flatnonzero(duration[selected] == duration[incoming])))
        selected[pos] = incoming
    evaluate = lambda s: objective(s, distances, magnitude, pattern, magnitude_weight, pattern_weight)
    terms = evaluate(selected)
    best, best_terms = selected.copy(), terms.copy()
    history = [{"step": 0, "temperature": temperature_start, "accepted": True,
                **terms, "best_score": best_terms["score"]}]
    movable = np.array([len(groups[g]) > quota for g in duration])
    for step in range(1, steps + 1):
        positions = np.flatnonzero(movable[selected])
        if not len(positions):
            break  # All candidates are already selected.
        temperature = temperature_start * (temperature_end / temperature_start) ** (step / steps)
        pos = int(rng.choice(positions))
        candidates = np.setdiff1d(groups[duration[selected[pos]]], selected)
        proposal = selected.copy()
        proposal[pos] = int(rng.choice(candidates))
        accepted = False
        if extreme[proposal].any():
            proposed_terms = evaluate(proposal)
            change = proposed_terms["score"] - terms["score"]
            accepted = change <= 0 or rng.random() < np.exp(-change / temperature)
            if accepted:
                selected, terms = proposal, proposed_terms
                if terms["score"] < best_terms["score"]:
                    best, best_terms = selected.copy(), terms.copy()
        history.append({"step": step, "temperature": temperature, "accepted": bool(accepted),
                        **terms, "best_score": best_terms["score"]})
    return best, best_terms, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV pointing to encoded [T,Z] arrays.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--subset-k", type=int, required=True)
    parser.add_argument("--magnitude-weight", type=float, default=0.5)
    parser.add_argument("--pattern-weight", type=float, default=0.3)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-initial-frame", action="store_true",
                        help="Use when the supplied sequences already exclude the initial dry frame.")
    args = parser.parse_args()
    if args.starts < 1 or args.seed < 0:
        parser.error("starts must be positive and seed must be nonnegative.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty. Use a separate directory for each experiment.")
    rows = read_manifest(args.manifest)
    features = np.stack([build_event_feature(np.load(r["path"], allow_pickle=False),
                                            not args.keep_initial_frame) for r in rows])
    distances, mean, scale = standardized_distances(features)
    mag_labels, magnitude = category_index(rows, "magnitude")
    pat_labels, pattern = category_index(rows, "pattern")
    dur_labels, duration = category_index(rows, "duration")
    extreme = np.array([r["is_extreme"] == "1" for r in rows])
    best, best_terms, best_history = None, None, None
    starts = []
    for i in range(args.starts):
        seed = args.seed + i * 10007
        selected, terms, history = anneal(
            distances, magnitude, pattern, duration, extreme, args.subset_k,
            np.random.default_rng(seed), args.magnitude_weight, args.pattern_weight,
            args.steps, args.temperature_start, args.temperature_end)
        starts.append({"start": i, "seed": seed, **terms})
        if best_terms is None or terms["score"] < best_terms["score"]:
            best, best_terms, best_history, best_seed = selected, terms, history, seed
        print(f"Start {i+1}/{args.starts}: score={terms['score']:.6f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = sorted(best.tolist())
    # Retain explicit event IDs; CSV row numbers are not portable event IDs.
    selected_rows = [{**rows[i], "path": os.path.relpath(rows[i]["path"], args.output_dir.resolve())}
                     for i in chosen]
    write_manifest(args.output_dir / "selected_events.csv", selected_rows)
    np.savez(args.output_dir / "descriptor_scaling.npz", mean=mean, scale=scale)
    counts = {field: {label: sum(rows[i][field] == label for i in chosen) for label in labels}
              for field, labels in [("magnitude", mag_labels), ("pattern", pat_labels), ("duration", dur_labels)]}
    summary = {"configuration": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "numpy_version": np.__version__, "candidate_count": len(rows),
               "descriptor_dimension": int(features.shape[1]), "best_seed": best_seed,
               "selected_event_ids": [rows[i]["event_id"] for i in chosen],
               "selected_category_counts": counts, "best_objective": best_terms, "starts": starts}
    (args.output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "best_start_history.json").write_text(json.dumps(best_history, indent=2), encoding="utf-8")
    print(f"Selected {len(chosen)} events. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
