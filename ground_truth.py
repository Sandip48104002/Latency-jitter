"""
ground_truth.py
───────────────
Reads the latency CSV produced by queuing_simulation.py and computes a
ground-truth jitter label for every packet using a per-flow EWMA.

Algorithm (per flow, in arrival order)
───────────────────────────────────────
  p1  → seed EWMA with its latency; no jitter decision yet
  p2  → update EWMA; no jitter decision yet (need at least 2 samples)
  p3+ → BEFORE updating EWMA with pkt n:
          if latency(n) > K * EWMA(n-1)  → HIGH jitter
          if latency(n) < (1/K) * EWMA(n-1) → LOW jitter  (optional)
          else → no jitter
        THEN update EWMA:
          EWMA(n) = alpha * latency(n) + (1-alpha) * EWMA(n-1)

The result CSV has all original columns plus two new ones:
  ewma_before   – EWMA value used for the jitter test on this packet
  jitter        – 1 if jitter detected, 0 otherwise

Tunable hyperparameters (see defaults below):
  ALPHA  – EWMA smoothing factor (higher = more reactive)
  K      – jitter detection multiplier
  MIN_PKTS_BEFORE_DECISION – packets needed per flow before we start judging
"""

import csv
import os
import argparse
from collections import defaultdict
from pathlib import Path

# ── hyperparameters ───────────────────────────────────────────────────────────
ALPHA   = 0.125        # EWMA smoothing factor (RFC 6298 inspired)
K       = 2.0          # threshold multiplier: jitter if latency > K*EWMA
                       # or latency < (1/K)*EWMA
MIN_PKTS_BEFORE_DECISION = 2   # need at least this many prior packets per flow


class FlowEWMAState:
    """Tracks EWMA state and packet count for a single flow."""

    def __init__(self, alpha: float):
        self.alpha   = alpha
        self.ewma    = None    # None until first packet seen
        self.count   = 0       # number of packets processed so far

    def update(self, latency: float) -> None:
        """Update EWMA with new latency sample."""
        if self.ewma is None:
            self.ewma = latency          # seed with first observation
        else:
            self.ewma = self.alpha * latency + (1 - self.alpha) * self.ewma
        self.count += 1

    def ewma_before(self) -> float | None:
        """EWMA value BEFORE the current packet is folded in."""
        return self.ewma   # ewma has NOT been updated yet when we call this


def detect_jitter(latency: float, ewma: float, k: float) -> int:
    """
    Return 1 if the latency deviates from ewma by more than factor k,
    return 0 otherwise.
    Both high spikes (latency > k*ewma) and low dips (latency < ewma/k) count.
    """
    if latency > k * ewma:
        return 1          # high jitter – sudden latency spike
    if latency < ewma / k:
        return 1          # low jitter  – sudden latency drop
    return 0


def run(in_csv: str, out_csv: str,
        alpha: float = ALPHA, k: float = K,
        min_pkts: int = MIN_PKTS_BEFORE_DECISION) -> list[dict]:
    """
    Parameters
    ----------
    in_csv   : latency CSV from queuing_simulation
    out_csv  : output path for ground-truth CSV
    alpha    : EWMA smoothing factor
    k        : jitter multiplier threshold
    min_pkts : minimum prior packets in flow before making a jitter decision

    Returns
    -------
    List of result dicts (same rows written to out_csv).
    """
    print(f"[ground_truth] Reading latency dataset: {in_csv}")

    # Load all rows; we need to process per-flow in arrival order.
    # The CSV is already sorted by ingress_ts (queuing_simulation sorts),
    # so we simply stream through and maintain per-flow state.
    rows = []
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[ground_truth] Processing {len(rows):,} packets …")

    flow_state: dict[str, FlowEWMAState] = defaultdict(lambda: FlowEWMAState(alpha))

    results = []
    jitter_count = 0

    for row in rows:
        flow_id    = row["flow_id"]
        latency    = float(row["latency_us"])
        state      = flow_state[flow_id]

        # ── decision phase (before EWMA update) ──────────────────────────────
        ewma_val   = state.ewma_before()   # None if no prior packets

        if ewma_val is not None and state.count >= min_pkts:
            # Enough history: make a jitter decision
            is_jitter  = detect_jitter(latency, ewma_val, k)
        else:
            # Not enough history yet; never flag as jitter
            is_jitter  = 0
            ewma_val   = latency           # report current latency as placeholder

        # ── EWMA update phase ─────────────────────────────────────────────────
        state.update(latency)

        if is_jitter:
            jitter_count += 1

        result = dict(row)                  # copy original columns
        result["ewma_before"] = round(ewma_val, 4)
        result["jitter"]      = is_jitter
        results.append(result)

    # ── write output CSV ──────────────────────────────────────────────────────
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    jitter_rate = 100 * jitter_count / max(len(results), 1)
    print(f"[ground_truth] Jitter packets  : {jitter_count:,} / {len(results):,} "
          f"({jitter_rate:.2f} %)")
    print(f"[ground_truth] Ground truth written → {out_csv}")

    return results


# ── standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground-truth jitter detection")
    parser.add_argument("in_csv",   help="Latency CSV from queuing_simulation.py")
    parser.add_argument("--out",    default="data/ground_truth.csv",
                        help="Output CSV (default: data/ground_truth.csv)")
    parser.add_argument("--alpha",  type=float, default=ALPHA,
                        help=f"EWMA alpha (default {ALPHA})")
    parser.add_argument("--k",      type=float, default=K,
                        help=f"Jitter multiplier (default {K})")
    parser.add_argument("--min-pkts", type=int, default=MIN_PKTS_BEFORE_DECISION,
                        help=f"Min prior pkts before decision (default {MIN_PKTS_BEFORE_DECISION})")
    args = parser.parse_args()
    run(args.in_csv, args.out, args.alpha, args.k, args.min_pkts)