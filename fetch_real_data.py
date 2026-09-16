from __future__ import annotations
import argparse
import json
import os
import sys
import time

from src.logging_utils import get_logger, log_exceptions

logger, LOG_PATH = get_logger("fetch_real_data")

# ALeRCE light-curve classifier class names -> this project's 3-class schema
ALERCE_CLASS_MAP = {
    "SNIa": "Ia",
    "SNIbc": "Ibc",
    "SNII": "II",
}

# Transient network errors worth retrying (DNS blips, connection resets,
# timeouts). Anything else (bad request, auth, etc.) is not retried --
# retrying a malformed request just wastes time and delays the real error.
RETRYABLE_EXCEPTIONS = (
    ConnectionError, TimeoutError, OSError,
)


def retry_with_backoff(fn, *args, max_retries=4, base_delay=2.0, on_retry_log=None, **kwargs):
    """Call fn(*args, **kwargs), retrying on transient network errors with
    exponential backoff (2s, 4s, 8s, 16s by default). This is specifically
    for the failure mode of a single DNS/connection hiccup killing an
    otherwise-successful multi-hour fetch -- most such failures resolve
    themselves within a few seconds if retried.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # requests/urllib3 wrap the real network error in their own
            # exception classes, so check the exception chain, not just
            # isinstance() on the top-level exception.
            is_retryable = isinstance(e, RETRYABLE_EXCEPTIONS)
            cause = e.__cause__
            while cause is not None and not is_retryable:
                is_retryable = isinstance(cause, RETRYABLE_EXCEPTIONS)
                cause = cause.__cause__
            # Also treat requests.exceptions.ConnectionError / Timeout by
            # name, without a hard dependency on importing requests here.
            if not is_retryable:
                is_retryable = type(e).__name__ in ("ConnectionError", "Timeout", "ReadTimeout")

            last_exc = e
            if not is_retryable or attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            msg = f"Transient error ({type(e).__name__}: {e}) -- retry {attempt}/{max_retries} in {delay:.0f}s"
            if on_retry_log:
                on_retry_log(msg)
            else:
                logger.warning(msg)
            time.sleep(delay)
    raise last_exc


def _import_alerce_client():
    try:
        from alerce.core import Alerce
        return Alerce
    except ImportError:
        logger.error(
            "The 'alerce' package is not installed. Install it with: "
            "pip install alerce   (also listed in environment.yaml)"
        )
        raise


@log_exceptions(logger)
def query_objects(alerce_class, page_size, max_pages=10):
    """Query ALeRCE for objects classified into a given light-curve class,
    paginating until we run out of results or hit max_pages.
    """
    Alerce = _import_alerce_client()
    client = Alerce()

    def _query_page(page):
        return client.query_objects(
            classifier="lc_classifier",
            class_name=alerce_class,
            probability=0.5,
            page_size=page_size,
            page=page,
            format="pandas",
        )

    objects = []
    for page in range(1, max_pages + 1):
        logger.info(f"Querying ALeRCE for class={alerce_class}, page={page}")
        result = retry_with_backoff(_query_page, page)
        if result is None or len(result) == 0:
            logger.info(f"No more results for {alerce_class} at page {page}")
            break
        objects.append(result)
        if len(result) < page_size:
            break
        time.sleep(0.5)  # be polite to the public API

    if not objects:
        return []
    import pandas as pd
    df = pd.concat(objects, ignore_index=True)
    return df["oid"].tolist()


BAND_MAP = {1: "g", 2: "r", 3: "i"}


def convert_detections_df(detections_df):
    """Pure, network-free conversion of an ALeRCE detections DataFrame
    (columns: fid, magpsf, sigmapsf, mjd) into this project's detection
    schema: {mjd, band, flux, flux_err, detected}.

    Split out from fetch_light_curve() so this logic can be unit-tested
    offline with a mock DataFrame (see tests/test_fetch_real_data.py) --
    the actual network call is the only untestable part in a sandboxed
    environment; the data-shape logic is fully covered.
    """
    if detections_df is None or len(detections_df) == 0:
        return []

    detections = []
    for _, row in detections_df.iterrows():
        band = BAND_MAP.get(int(row.get("fid", 0)), None)
        if band is None:
            continue
        # ALeRCE reports magnitudes (magpsf); convert to a linear flux
        # proxy for consistency with this project's flux-based schema.
        # This is a standard mag->flux conversion for relative-shape
        # purposes -- absolute flux calibration is not required here
        # since preprocessing.py peak-normalizes per object anyway.
        mag = row.get("magpsf", None)
        if mag is None:
            continue
        flux = 10 ** (-0.4 * float(mag))
        detections.append({
            "mjd": float(row["mjd"]),
            "band": band,
            "flux": flux,
            "flux_err": float(row.get("sigmapsf", 0.1)),
            "detected": True,
        })
    return detections


@log_exceptions(logger)
def fetch_light_curve(oid):
    """Fetch detections for a single object ID from the real ALeRCE API
    and convert them via convert_detections_df(). Transient network
    errors are retried with backoff before giving up on this object.
    """
    Alerce = _import_alerce_client()
    client = Alerce()
    detections_df = retry_with_backoff(client.query_detections, oid, format="pandas")
    return convert_detections_df(detections_df)


def _atomic_write_json(data, path):
    """Write JSON to `path` via a temp file + rename, so a crash or
    interrupt mid-write never corrupts or truncates the existing file --
    the old file is only replaced once the new one is fully written.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def _load_existing_dataset(out_path):
    """For --resume: load a partially-completed dataset from a previous
    (possibly crashed) run, so already-fetched objects are never
    re-fetched. Returns (dataset_list, {label: set(object_ids_done)}).
    """
    if not os.path.exists(out_path):
        return [], {}
    try:
        with open(out_path) as f:
            dataset = json.load(f)
        done_by_label = {}
        for obj in dataset:
            done_by_label.setdefault(obj["label"], set()).add(obj["object_id"])
        logger.info(f"Resuming from existing {out_path}: "
                    f"{len(dataset)} objects already fetched "
                    f"({ {k: len(v) for k, v in done_by_label.items()} })")
        return dataset, done_by_label
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not parse existing {out_path} for resume ({e}) -- starting fresh.")
        return [], {}


@log_exceptions(logger)
def build_real_dataset(n_per_class: int, min_detections: int, out_path: str,
                        resume: bool = True, save_every: int = 10,
                        balance_mode: str = "equal", total_budget: int | None = None,
                        candidate_page_size: int = 200, candidate_max_pages: int = 5):
    """Builds the dataset one class at a time, saving progress to
    `out_path` after every `save_every` newly-fetched objects and at the
    end of each class -- so a crash (network blip, killed process, etc.)
    at any point loses at most `save_every` objects of partial work, not
    the entire run. Individual object fetch failures (after retries are
    exhausted) are logged and skipped rather than aborting the run.

    balance_mode:
      "equal"   (default) -- fetch exactly n_per_class objects per class,
                the same as always. Produces an artificially balanced
                sample -- good for a clean classifier comparison, but
                NOT representative of true SN sub-type rates in nature
                (see report Limitations).
      "natural" -- fetch a NON-curated, naturally-imbalanced sample
                instead: candidate counts are queried for all three
                classes first, then `total_budget` objects (default
                n_per_class * 3, i.e. same total fetch effort as equal
                mode) are allocated across classes IN PROPORTION to how
                many candidates ALeRCE reports for each -- e.g. if Ia
                candidates outnumber Ibc candidates 2:1, roughly twice
                as many Ia objects are fetched as Ibc. This is a proxy
                for natural class imbalance (it reflects what ALeRCE's
                classifier has actually found and labeled, not a true
                volume-limited survey rate), not a claim of true
                population statistics -- document this distinction
                wherever these numbers are reported.
    """
    dataset, done_by_label = _load_existing_dataset(out_path) if resume else ([], {})

    # Candidate lists are needed either way, and needed BEFORE the fetch
    # loop in "natural" mode so per-class targets can be computed from
    # relative candidate counts up front.
    candidate_lists = {}
    for alerce_class, our_label in ALERCE_CLASS_MAP.items():
        logger.info(f"Querying candidate list for {alerce_class} -> {our_label}")
        oids = query_objects(alerce_class, page_size=candidate_page_size, max_pages=candidate_max_pages)
        candidate_lists[our_label] = oids
        logger.info(f"Got {len(oids)} candidate object IDs for {alerce_class}")

    if balance_mode == "natural":
        total_budget = total_budget or (n_per_class * len(ALERCE_CLASS_MAP))
        total_candidates = sum(len(v) for v in candidate_lists.values())
        if total_candidates == 0:
            raise RuntimeError("No candidates found for any class -- check ALeRCE API connectivity.")
        class_targets = {
            label: max(1, round(total_budget * len(oids) / total_candidates))
            for label, oids in candidate_lists.items()
        }
        logger.info(f"NATURAL (non-curated) balance mode: total_budget={total_budget}, "
                    f"candidate counts={ {k: len(v) for k, v in candidate_lists.items()} }, "
                    f"-> per-class targets={class_targets}")
    elif balance_mode == "equal":
        class_targets = {label: n_per_class for label in ALERCE_CLASS_MAP.values()}
        logger.info(f"EQUAL balance mode: per-class targets={class_targets}")
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode!r} (use 'equal' or 'natural')")

    for alerce_class, our_label in ALERCE_CLASS_MAP.items():
        target = class_targets[our_label]
        already_done = done_by_label.get(our_label, set())
        kept = len(already_done)
        if kept >= target:
            logger.info(f"{our_label}: already have {kept}/{target} from a previous run -- skipping.")
            continue

        oids = candidate_lists[our_label]
        logger.info(f"Fetching {our_label} detections: target={target}, "
                    f"{kept} already done, {len(oids)} candidates available")

        n_fetched_this_session = 0
        n_failed = 0
        for oid in oids:
            if kept >= target:
                break
            if oid in already_done:
                continue  # already fetched in a previous (interrupted) run

            try:
                dets = fetch_light_curve(oid)
            except Exception as e:
                n_failed += 1
                logger.warning(f"Skipping object {oid} after retries exhausted: "
                                f"{type(e).__name__}: {e}")
                continue

            if len(dets) < min_detections:
                continue
            dataset.append({"object_id": oid, "label": our_label, "detections": dets})
            already_done.add(oid)
            kept += 1
            n_fetched_this_session += 1

            if kept % 20 == 0:
                logger.info(f"{our_label}: {kept}/{target} objects fetched")
            if n_fetched_this_session % save_every == 0:
                _atomic_write_json(dataset, out_path)
                logger.info(f"Checkpoint saved: {len(dataset)} total objects so far -> {out_path}")

            time.sleep(0.2)  # be polite to the public API

        _atomic_write_json(dataset, out_path)  # always checkpoint at end of each class
        logger.info(f"Finished {our_label}: {kept}/{target} objects with >= {min_detections} detections "
                    f"({n_failed} objects skipped due to fetch failures). Checkpoint saved.")

    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_per_class", type=int, default=150)
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--out", type=str, default="data/ztf_real_objects.json")
    parser.add_argument("--no_resume", action="store_true",
                         help="Ignore any existing --out file and start fresh instead of "
                              "resuming/skipping already-fetched objects.")
    parser.add_argument("--save_every", type=int, default=10,
                         help="Checkpoint (write partial results) to --out every N newly-fetched objects.")
    parser.add_argument("--balance_mode", type=str, choices=["equal", "natural"], default="equal",
                         help="'equal' (default): fetch --n_per_class objects for every class -- an "
                              "artificially balanced sample, good for a clean model comparison. "
                              "'natural': fetch a NON-curated sample instead, allocated across classes "
                              "in proportion to each class's relative candidate count in ALeRCE -- a "
                              "proxy for real-world class imbalance. See report Limitations re: this "
                              "is not a claim of true population statistics.")
    parser.add_argument("--total_budget", type=int, default=None,
                         help="Only used with --balance_mode natural: total objects to fetch across "
                              "all classes combined. Defaults to --n_per_class * 3 (same total fetch "
                              "effort as equal mode, just redistributed).")
    args = parser.parse_args()

    logger.info(f"Starting real ZTF data fetch via ALeRCE: "
                f"n_per_class={args.n_per_class}, min_detections={args.min_detections}, "
                f"resume={not args.no_resume}, balance_mode={args.balance_mode}")
    try:
        dataset = build_real_dataset(
            args.n_per_class, args.min_detections, out_path=args.out,
            resume=not args.no_resume, save_every=args.save_every,
            balance_mode=args.balance_mode, total_budget=args.total_budget,
        )
    except Exception:
        # Note: even on a fatal failure here, build_real_dataset has
        # already checkpointed everything successfully fetched so far to
        # --out (every save_every objects and at the end of each class),
        # so re-running this exact command will resume rather than
        # start over from zero.
        logger.error(
            "Fetch failed before completing. Progress made so far has "
            f"already been saved to {args.out} -- simply re-run the same "
            "command to resume from where it stopped. Common causes if "
            "this keeps failing: (1) 'alerce' package not installed -- "
            "pip install alerce; (2) no/unstable internet access to "
            "api.alerce.online from this machine/network; (3) ALeRCE "
            "API schema has changed since this script was written -- "
            "check https://alerce.readthedocs.io/ for the current "
            "client API and adjust query_objects()/fetch_light_curve() "
            "accordingly."
        )
        sys.exit(1)

    logger.info(f"Wrote {len(dataset)} real ZTF objects to {args.out}")
    logger.info(f"Full log written to: {LOG_PATH}")
    logger.info("Done. You can now run: python train.py --data "
                f"{args.out} --outdir figures_real --run_name real")


if __name__ == "__main__":
    main()
