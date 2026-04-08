"""
main.py
───────
Orchestrates the full jitter-detection experiment pipeline and runs
BOTH sketch variants back-to-back for a direct comparison:

  Step 1 → queuing_simulation  : parse CAIDA .dat, simulate FIFO queue,
                                  produce latency dataset CSV
  Step 2 → ground_truth        : apply per-flow EWMA, produce GT labels CSV
  Step 3a → JitterSketch       : sketch WITH fingerprint  (original)
  Step 3b → JitterSketchNoFP   : sketch WITHOUT fingerprint (new)
  Step 4  → evaluate & compare : Precision / Recall / F1 / Throughput
                                  printed side-by-side for both variants

Usage
─────
  python main.py --dat path/to/trace.dat
  python main.py --synthetic --num-pkts 500000

Output files
────────────
  data/latency_dataset.csv          – per-packet latency
  data/ground_truth.csv             – per-packet GT jitter labels
  data/sketch_fp_results.csv        – predictions: with fingerprint
  data/sketch_nofp_results.csv      – predictions: no fingerprint
  data/metrics_comparison.csv       – both metric rows in one CSV
"""

import argparse
import csv
import os
import random
import struct
import time
from pathlib import Path

import queuing_simulation as qs
import ground_truth       as gt
from sketch import JitterSketch, JitterSketchNoFP, ALPHA, K

# ── output paths ──────────────────────────────────────────────────────────────
DATA_DIR             = "data"
LATENCY_CSV          = os.path.join(DATA_DIR, "latency_dataset.csv")
GROUND_TRUTH_CSV     = os.path.join(DATA_DIR, "ground_truth.csv")
SKETCH_FP_CSV        = os.path.join(DATA_DIR, "sketch_fp_results.csv")
SKETCH_NOFP_CSV      = os.path.join(DATA_DIR, "sketch_nofp_results.csv")
METRICS_CSV          = os.path.join(DATA_DIR, "metrics_comparison.csv")


# ── synthetic data generator (no .dat file needed for testing) ────────────────

def generate_synthetic_dat(path, num_packets=500_000, num_flows=1_000,
                            jitter_prob=0.05):
    """Write a synthetic CAIDA-format .dat file (correct real format)."""
    # Format: src_ip(4,BE) dst_ip(4,BE) sp(2,BE) dp(2,BE) proto(1)
    #         ts_f64(8,LE) pkt_size(2,LE) flags(1) = 24 bytes
    prefix_fmt = struct.Struct('>IIHH B')   # first 13 bytes (big-endian prefix)

    os.makedirs(Path(path).parent, exist_ok=True)

    flows = []
    for _ in range(num_flows):
        flows.append((
            random.randint(0x0A000001, 0x0AFFFFFF),
            random.randint(0xC0A80001, 0xC0A8FFFF),
            random.randint(1024, 65535),
            random.randint(1, 1023),
            random.choice([6, 17]),
        ))

    ts      = 1_453_381_151.0   # same epoch as real CAIDA trace (2016-01-21)
    GAP_US  = 2.0 / num_packets  # spread across ~2 seconds

    with open(path, "wb") as f:
        for i in range(num_packets):
            src_ip, dst_ip, sp, dp, proto = random.choice(flows)
            pkt_size = random.randint(6000, 9000) if random.random() < jitter_prob \
                       else random.randint(64, 1500)
            ts += GAP_US
            flags = 0x10 if proto == 6 else 0x00

            record  = prefix_fmt.pack(src_ip, dst_ip, sp, dp, proto)
            record += struct.pack('<d', ts)          # float64 LE
            record += struct.pack('<H', pkt_size)    # uint16 LE
            record += bytes([flags])
            f.write(record)

    print(f"[main] Synthetic trace → {path}  ({num_packets:,} pkts, {num_flows} flows)")
    return path


# ── sketch replay ─────────────────────────────────────────────────────────────

def run_sketch(ground_truth_csv: str, out_csv: str, sketch_obj, label: str):
    """
    Replay the latency CSV through sketch_obj.
    Returns (true_labels, pred_labels, throughput_pps).
    """
    with open(ground_truth_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"[{label}] Processing {len(rows):,} packets …")
    t0 = time.perf_counter()

    true_labels = []
    pred_labels = []
    result_rows = []

    for row in rows:
        flow_id  = row["flow_id"]
        latency  = float(row["latency_us"])
        gt_label = int(row["jitter"])
        sk_label = sketch_obj.process_packet(flow_id, latency)

        true_labels.append(gt_label)
        pred_labels.append(sk_label)
        result_rows.append({
            "flow_id":    flow_id,
            "ingress_ts": row["ingress_ts"],
            "latency_us": row["latency_us"],
            "gt_jitter":  gt_label,
            "sk_jitter":  sk_label,
        })

    elapsed    = time.perf_counter() - t0
    throughput = len(rows) / elapsed if elapsed > 0 else float("inf")

    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    sk_stats = sketch_obj.stats()
    print(f"[{label}] Done in {elapsed:.2f}s  |  throughput: {throughput:,.0f} pkt/s")
    print(f"[{label}] Sketch stats: {sk_stats}")
    print(f"[{label}] Results → {out_csv}")

    return true_labels, pred_labels, throughput


# ── evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(true_labels, pred_labels, throughput) -> dict:
    """
    Precision / Recall / F1 for the positive class (jitter = 1).
    TP: sketch=1, truth=1  |  FP: sketch=1, truth=0
    FN: sketch=0, truth=1  |  TN: sketch=0, truth=0
    """
    TP = FP = FN = TN = 0
    for gt_v, sk_v in zip(true_labels, pred_labels):
        if   gt_v == 1 and sk_v == 1: TP += 1
        elif gt_v == 0 and sk_v == 1: FP += 1
        elif gt_v == 1 and sk_v == 0: FN += 1
        else:                         TN += 1

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision":      round(prec, 4),
        "recall":         round(rec,  4),
        "f1_score":       round(f1,   4),
        "throughput_pps": round(throughput, 0),
    }


def print_comparison(m_fp: dict, m_nofp: dict,
                     bpr_fp: int, bpr_nofp: int):
    """Print a side-by-side comparison table of both variants."""

    def row(label, key, fmt="{}", unit=""):
        v_fp   = fmt.format(m_fp[key])   + unit
        v_nofp = fmt.format(m_nofp[key]) + unit
        # Arrow shows which is better (higher = better for most metrics)
        if isinstance(m_fp[key], (int, float)):
            arrow = "▲" if m_nofp[key] > m_fp[key] else ("▼" if m_nofp[key] < m_fp[key] else "=")
        else:
            arrow = ""
        print(f"  {label:<26} {v_fp:>14}   {v_nofp:>14}  {arrow}")

    sep = "─" * 66
    print()
    print("╔" + "═" * 64 + "╗")
    print("║{:^64}║".format("  Jitter Detection — Variant Comparison  "))
    print("╠" + "═" * 64 + "╣")
    print(f"║  {'':26} {'With FP':>14}   {'No FP':>14}  ║")
    print(f"║  {'Buckets / row':26} {bpr_fp:>14,}   {bpr_nofp:>14,}   ║")
    print(f"║  {'Memory (KB)':26} {'400':>14}   {'400':>14}   ║")
    print(f"║  {'Bucket size (bytes)':26} {'12':>14}   {'10':>14}   ║")
    print("╠" + "═" * 64 + "╣")
    row("True Positives  (TP)", "TP", "{:,}")
    row("False Positives (FP)", "FP", "{:,}")
    row("False Negatives (FN)", "FN", "{:,}")
    row("True Negatives  (TN)", "TN", "{:,}")
    print("╠" + "═" * 64 + "╣")
    row("Precision",            "precision",      "{:.4f}")
    row("Recall",               "recall",         "{:.4f}")
    row("F1 Score",             "f1_score",       "{:.4f}")
    print("╠" + "═" * 64 + "╣")
    row("Throughput (pkt/s)",   "throughput_pps", "{:,.0f}")
    print("╚" + "═" * 64 + "╝")
    print()
    print("  ▲ = No-FP variant is better for this metric")
    print("  ▼ = No-FP variant is worse  for this metric")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sketch jitter detection — with vs without fingerprint"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dat",       metavar="FILE", help="CAIDA .dat trace file")
    src.add_argument("--synthetic", action="store_true",
                     help="Generate synthetic trace for testing")

    parser.add_argument("--num-pkts",    type=int,   default=500_000)
    parser.add_argument("--num-flows",   type=int,   default=1_000)
    parser.add_argument("--jitter-prob", type=float, default=0.05)
    parser.add_argument("--alpha",       type=float, default=ALPHA)
    parser.add_argument("--k",           type=float, default=K)
    parser.add_argument("--max-pkts",    type=int,   default=0)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 0: source data ───────────────────────────────────────────────────
    if args.synthetic:
        dat_file = os.path.join(DATA_DIR, "synthetic_trace.dat")
        generate_synthetic_dat(dat_file, args.num_pkts, args.num_flows,
                               args.jitter_prob)
    else:
        dat_file = args.dat
        if not os.path.exists(dat_file):
            print(f"[main] ERROR: file not found: {dat_file}")
            return

    # ── Step 1: queue simulation ──────────────────────────────────────────────
    print("\n── Step 1: FIFO Queue Simulation ──────────────────────────────")
    qs.run(dat_file, LATENCY_CSV, max_packets=args.max_pkts)

    # ── Step 2: ground truth ──────────────────────────────────────────────────
    print("\n── Step 2: Ground Truth Generation ────────────────────────────")
    gt.run(LATENCY_CSV, GROUND_TRUTH_CSV, alpha=args.alpha, k=args.k)

    # ── Step 3a: sketch WITH fingerprint ─────────────────────────────────────
    print("\n── Step 3a: Sketch WITH Fingerprint ────────────────────────────")
    sk_fp    = JitterSketch(alpha=args.alpha, k=args.k)
    bpr_fp   = sk_fp.bpr
    print(f"[sketch_fp] {sk_fp.num_rows} rows × "
          f"{sk_fp.bpr:,} buckets × 12 B = "
          f"{sk_fp.memory_bytes()/1024:.0f} KB")
    tl, pl_fp, tp_fp = run_sketch(GROUND_TRUTH_CSV, SKETCH_FP_CSV,
                                   sk_fp, "sketch_fp")

    # ── Step 3b: sketch WITHOUT fingerprint ──────────────────────────────────
    print("\n── Step 3b: Sketch WITHOUT Fingerprint ─────────────────────────")
    sk_nofp  = JitterSketchNoFP(alpha=args.alpha, k=args.k)
    bpr_nofp = sk_nofp.bpr
    print(f"[sketch_nofp] {sk_nofp.num_rows} rows × "
          f"{sk_nofp.bpr:,} buckets × 10 B = "
          f"{sk_nofp.memory_bytes()/1024:.0f} KB")
    tl2, pl_nofp, tp_nofp = run_sketch(GROUND_TRUTH_CSV, SKETCH_NOFP_CSV,
                                        sk_nofp, "sketch_nofp")

    # ── Step 4: evaluate & compare ────────────────────────────────────────────
    print("\n── Step 4: Evaluation & Comparison ─────────────────────────────")
    m_fp   = compute_metrics(tl,  pl_fp,   tp_fp)
    m_nofp = compute_metrics(tl2, pl_nofp, tp_nofp)

    m_fp["variant"]   = "with_fingerprint"
    m_nofp["variant"] = "no_fingerprint"

    print_comparison(m_fp, m_nofp, bpr_fp, bpr_nofp)

    # Save both rows to one CSV
    fieldnames = ["variant"] + [k for k in m_fp if k != "variant"]
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({**{"variant": m_fp.pop("variant")}, **m_fp})
        writer.writerow({**{"variant": m_nofp.pop("variant")}, **m_nofp})
    print(f"[main] Comparison metrics → {METRICS_CSV}")


if __name__ == "__main__":
    main()