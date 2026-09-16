import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logging_utils import get_logger

import numpy as np

BANDS = ["g", "r", "i"]

# Very rough, illustrative shape parameters per class (NOT physically
# calibrated — for pipeline smoke-testing only).
CLASS_PARAMS = {
    "Ia":  {"rise": 18, "decline": 0.03, "noise": 0.05},
    "Ibc": {"rise": 12, "decline": 0.05, "noise": 0.08},
    "II":  {"rise": 8,  "decline": 0.015, "noise": 0.10},
}


def simulate_light_curve(rise, decline, noise, n_days=80, cadence_mean=3.0, rng=None):
    rng = rng or np.random.default_rng()
    detections = []
    t = 0.0
    while t < n_days:
        t += rng.exponential(cadence_mean)
        if t >= n_days:
            break
        # simple rise/decline flux template
        if t < rise:
            flux = (t / rise) ** 2
        else:
            flux = np.exp(-decline * (t - rise))
        for band in BANDS:
            if rng.random() < 0.7:  # not every band observed every night
                band_noise = rng.normal(0, noise)
                detections.append({
                    "mjd": 59000.0 + t,
                    "band": band,
                    "flux": float(max(flux + band_noise, 0.0)),
                    "flux_err": float(noise),
                    "detected": True,
                })
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_per_class", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/synthetic_objects.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger("generate_synthetic_data")
    logger.info(f"Arguments: {vars(args)}")

    try:
        rng = np.random.default_rng(args.seed)
        objects = []
        obj_counter = 0
        for label, params in CLASS_PARAMS.items():
            n_kept = 0
            for _ in range(args.n_per_class):
                dets = simulate_light_curve(
                    rise=params["rise"] * rng.uniform(0.8, 1.2),
                    decline=params["decline"] * rng.uniform(0.8, 1.2),
                    noise=params["noise"],
                    rng=rng,
                )
                if len(dets) < 3:
                    continue
                objects.append({
                    "object_id": f"SYN{obj_counter:05d}",
                    "label": label,
                    "detections": dets,
                })
                obj_counter += 1
                n_kept += 1
            logger.info(f"Class {label}: generated {n_kept}/{args.n_per_class} objects")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(objects, f)
        logger.info(f"Wrote {len(objects)} synthetic objects to {args.out}")
        logger.info(f"Full log written to: {log_path}")
    except Exception as e:
        logger.error(f"generate_synthetic_data.py failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
