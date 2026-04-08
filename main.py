"""
main.py
───────
Orchestrates the full jitter-detection experiment pipeline:

  Step 1 → queuing_simulation  : parse CAIDA .dat, simulate FIFO queue,
                                  produce latency dataset CSV
  Step 2 → ground_truth        : apply per-flow EWMA jitter detection,
                                  produce ground-truth labels CSV
  Step 3 → sketch              : replay the latency dataset through the
                                  JitterSketch and collect sketch predictions
  Step 4 → evaluate            : compute Precision, Recall, F1-score and
                                  Throughput (packets/second)

Usage
─────
  # With a real CAIDA .dat file:
  python main.py --dat path/to/trace.dat

  # Quick smoke-test with synthetic data (no .dat file needed):
  python main.py --synthetic --num-pkts 200000

Output
──────
  data/latency_dataset.csv    – per-packet latency from queue simulation
  data/ground_truth.csv       – per-packet jitter ground truth labels
  data/sketch_results.csv     – per-packet sketch predictions
  Evaluation metrics printed to stdout.
"""

import argparse
import csv
import os
import random
import struct
import tempfile
import time
from pathlib import Path

# ── project modules ───────────────────────────────────────────────────────────
import queuing_simulation as qs
import ground_truth       as gt
from sketch import JitterSketch, ALPHA, K, BUCKETS_PER_ROW, NUM_ROWS

# ── default paths ─────────────────────────────────────────────────────────────
DATA_DIR          = "data"
LATENCY_CSV       = os.path.join(DATA_DIR, "latency_dataset.csv")
GROUND_TRUTH_CSV  = os.path.join(DATA_DIR, "ground_truth.csv")
SKETCH_RESULTS_CSV= os.path.join(DATA_DIR, "sketch_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# Used when no real .dat file is available (for testing/demo).
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dat(path: str, num_packets: int = 100_000,
                            num_flows: int = 500, jitter_prob: float = 0.05):
    """
    Write a synthetic CAIDA-format .dat file.

    Most packets in a flow have a baseline pkt_size (→ similar latency).
    A fraction jitter_prob of packets get a much larger size to simulate
    latency spikes in the queue.

    24-byte layout (big-endian): src_ip(4) dst_ip(4) src_port(2) dst_port(2)
                                  proto(1) flags(1) pkt_size(2) ts_us(8)
    """
    record_fmt = struct.Struct(">IIHH BBH Q")
    os.makedirs(Path(path).parent, exist_ok=True)

    # Pre-generate flow endpoints
    flows = []
    for _ in range(num_flows):
        src = random.randint(0x0A000001, 0x0AFFFFFF)   # 10.x.x.x
        dst = random.randint(0xC0A80001, 0xC0A8FFFF)   # 192.168.x.x
        sp  = random.randint(1024, 65535)
        dp  = random.randint(1, 1023)
        pr  = random.choice([6, 17])                    # TCP or UDP
        flows.append((src, dst, sp, dp, pr))

    ts = 1_000_000_000   # start timestamp (µs) – arbitrary epoch offset
    INTER_ARRIVAL_US = 10  # average inter-arrival gap (µs)

    with open(path, "wb") as f:
        for i in range(num_packets):
            src_ip, dst_ip, src_port, dst_port, proto = random.choice(flows)
            flags    = 0x02 if proto == 6 else 0x00
            # Most packets 64-1500 bytes; occasional spike to 9000 (jitter probe)
            if random.random() < jitter_prob:
                pkt_size = random.randint(6000, 9000)
            else:
                pkt_size = random.randint(64, 1500)
            ts += random.randint(1, INTER_ARRIVAL_US * 2)
            f.write(record_fmt.pack(src_ip, dst_ip, src_port, dst_port,
                                     proto, flags, pkt_size, ts))

    print(f"[main] Synthetic trace written → {path}  "
          f"({num_packets:,} pkts, {num_flows} flows)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SKETCH REPLAY
# ─────────────────────────────────────────────────────────────────────────────

def run_sketch(ground_truth_csv: str,
               sketch_results_csv: str,
               alpha: float = ALPHA,
               k: float     = K) -> tuple[list[int], list[int], float]:
    """
    Replay every packet in the ground-truth CSV through the JitterSketch.

    Returns
    -------
    (true_labels, sketch_labels, throughput_pps)
        true_labels   : ground-truth jitter flags (0/1) in packet order
        sketch_labels : sketch prediction flags    (0/1) in packet order
        throughput_pps: packets processed per second
    """
    print(f"[sketch] Initialising JitterSketch "
          f"({NUM_ROWS} rows × {BUCKETS_PER_ROW:,} buckets "
          f"≈ {BUCKETS_PER_ROW * NUM_ROWS * 8 / 1024:.0f} KB)")

    sketch = JitterSketch(alpha=alpha, k=k)

    true_labels   = []
    sketch_labels = []
    result_rows   = []

    # Read the ground-truth CSV (already sorted by ingress_ts)
    with open(ground_truth_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    print(f"[sketch] Processing {len(rows):,} packets …")
    t_start = time.perf_counter()

    for row in rows:
        flow_id    = row["flow_id"]
        latency    = float(row["latency_us"])
        gt_label   = int(row["jitter"])

        sk_label   = sketch.process_packet(flow_id, latency)

        true_labels.append(gt_label)
        sketch_labels.append(sk_label)

        result_rows.append({
            "flow_id":      flow_id,
            "ingress_ts":   row["ingress_ts"],
            "latency_us":   row["latency_us"],
            "gt_jitter":    gt_label,
            "sk_jitter":    sk_label,
        })

    elapsed   = time.perf_counter() - t_start
    throughput = len(rows) / elapsed if elapsed > 0 else float("inf")

    # ── write sketch result CSV ───────────────────────────────────────────────
    os.makedirs(Path(sketch_results_csv).parent, exist_ok=True)
    with open(sketch_results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"[sketch] Done in {elapsed:.2f}s  "
          f"| throughput: {throughput:,.0f} pkt/s")
    print(f"[sketch] Sketch stats: {sketch.stats()}")
    print(f"[sketch] Results written → {sketch_results_csv}")

    return true_labels, sketch_labels, throughput


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(true_labels: list[int],
                    pred_labels: list[int],
                    throughput_pps: float) -> dict:
    """
    Compute binary classification metrics.

    Definitions (positive class = jitter detected)
    ────────────────────────────────────────────────
    TP : sketch says jitter, ground truth says jitter
    FP : sketch says jitter, ground truth says NO jitter  (false alarm)
    FN : sketch says NO jitter, ground truth says jitter  (missed jitter)
    TN : both say NO jitter

    Precision = TP / (TP + FP)   – "when sketch cries wolf, is it right?"
    Recall    = TP / (TP + FN)   – "how many real jitter events did we catch?"
    F1        = harmonic mean of Precision and Recall
    """
    TP = FP = FN = TN = 0
    for gt, sk in zip(true_labels, pred_labels):
        if gt == 1 and sk == 1:
            TP += 1
        elif gt == 0 and sk == 1:
            FP += 1
        elif gt == 1 and sk == 0:
            FN += 1
        else:
            TN += 1

    precision  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall     = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

    return {
        "TP":             TP,
        "FP":             FP,
        "FN":             FN,
        "TN":             TN,
        "precision":      round(precision,  4),
        "recall":         round(recall,     4),
        "f1_score":       round(f1,         4),
        "throughput_pps": round(throughput_pps, 0),
    }


def print_metrics(metrics: dict):
    """Pretty-print the evaluation metrics table."""
    sep = "─" * 52
    print()
    print("╔" + "═" * 50 + "╗")
    print("║{:^50}║".format("  Jitter Detection – Evaluation Results  "))
    print("╠" + "═" * 50 + "╣")
    print(f"║  {'True Positives  (TP)':<28} {metrics['TP']:>18,} ║")
    print(f"║  {'False Positives (FP)':<28} {metrics['FP']:>18,} ║")
    print(f"║  {'False Negatives (FN)':<28} {metrics['FN']:>18,} ║")
    print(f"║  {'True Negatives  (TN)':<28} {metrics['TN']:>18,} ║")
    print("╠" + "═" * 50 + "╣")
    print(f"║  {'Precision':<28} {metrics['precision']:>18.4f} ║")
    print(f"║  {'Recall':<28} {metrics['recall']:>18.4f} ║")
    print(f"║  {'F1 Score':<28} {metrics['f1_score']:>18.4f} ║")
    print("╠" + "═" * 50 + "╣")
    print(f"║  {'Throughput (pkt/s)':<28} {metrics['throughput_pps']:>18,.0f} ║")
    print("╚" + "═" * 50 + "╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sketch-based jitter detection experiment (CAIDA trace)"
    )
    # Input source
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--dat", metavar="FILE",
                           help="Path to CAIDA .dat trace file")
    src_group.add_argument("--synthetic", action="store_true",
                           help="Generate synthetic trace for testing")

    # Synthetic options
    parser.add_argument("--num-pkts",  type=int, default=500_000,
                        help="Packets in synthetic trace (default 500 000)")
    parser.add_argument("--num-flows", type=int, default=1_000,
                        help="Distinct flows in synthetic trace (default 1 000)")
    parser.add_argument("--jitter-prob", type=float, default=0.05,
                        help="Fraction of synthetic jitter packets (default 0.05)")

    # Algorithm knobs
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"EWMA alpha (default {ALPHA})")
    parser.add_argument("--k",     type=float, default=K,
                        help=f"Jitter multiplier (default {K})")
    parser.add_argument("--max-pkts", type=int, default=0,
                        help="Cap packets read from .dat (0 = all)")

    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 0: obtain a .dat file ────────────────────────────────────────────
    if args.synthetic:
        dat_file = os.path.join(DATA_DIR, "synthetic_trace.dat")
        generate_synthetic_dat(dat_file,
                               num_packets=args.num_pkts,
                               num_flows=args.num_flows,
                               jitter_prob=args.jitter_prob)
    else:
        dat_file = args.dat
        if not os.path.exists(dat_file):
            print(f"[main] ERROR: file not found: {dat_file}")
            return

    # ── Step 1: queuing simulation ────────────────────────────────────────────
    print("\n── Step 1: FIFO Queue Simulation ──────────────────────────────")
    qs.run(dat_file, LATENCY_CSV, max_packets=args.max_pkts)

    # ── Step 2: ground truth ──────────────────────────────────────────────────
    print("\n── Step 2: Ground Truth Generation ────────────────────────────")
    gt.run(LATENCY_CSV, GROUND_TRUTH_CSV, alpha=args.alpha, k=args.k)

    # ── Step 3: sketch replay ─────────────────────────────────────────────────
    print("\n── Step 3: Sketch Jitter Detection ────────────────────────────")
    true_labels, sketch_labels, throughput = run_sketch(
        GROUND_TRUTH_CSV, SKETCH_RESULTS_CSV,
        alpha=args.alpha, k=args.k
    )

    # ── Step 4: evaluation ────────────────────────────────────────────────────
    print("\n── Step 4: Evaluation Metrics ──────────────────────────────────")
    metrics = compute_metrics(true_labels, sketch_labels, throughput)
    print_metrics(metrics)

    # Save metrics to CSV for reproducibility
    metrics_csv = os.path.join(DATA_DIR, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[main] Metrics saved → {metrics_csv}")


if __name__ == "__main__":
    main()