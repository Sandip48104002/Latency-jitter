"""
queuing_simulation.py
─────────────────────
Reads a raw CAIDA .dat trace (24 bytes per packet), simulates a FIFO
output queue on a single switch port, and writes a CSV file containing
one row per packet with:
    flow_id, ingress_ts, egress_ts, latency_us, pkt_size

ACTUAL CAIDA 24-byte record layout (verified by byte-level inspection)
─────────────────────────────────────────────────────────────────────────
  Offset  Size  Endian  Field
   0       4    big     src_ip
   4       4    big     dst_ip
   8       2    big     src_port
  10       2    big     dst_port
  12       1    –       protocol   (6=TCP, 17=UDP)
  13       8    little  timestamp  (float64 seconds since Unix epoch,
                                    microsecond precision)
  21       2    little  pkt_size   (bytes)
  23       1    –       tcp_flags

NOTE: The timestamp is a 64-bit double (float64) stored little-endian.
      Its high 6 bytes (offsets 15-20) are constant within a capture
      session (same second); only the low 2 bytes (offsets 13-14) vary
      to encode the sub-second microsecond component.

The FIFO queue model
─────────────────────
  egress_ts  = max(ingress_ts, queue_free_at) + tx_delay
  tx_delay   = pkt_size / (link_speed_mbps / 8)   [µs]
  latency    = egress_ts - ingress_ts
"""

import struct
import csv
import os
import argparse
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
RECORD_SIZE      = 24
# CAIDA 2016 trace: ~1.1M pkts in ~2.3 s ≈ 478k pps, avg ~600 B/pkt → ~2.3 Gbps input
# 10 Gbps link gives ~23% utilisation → realistic queue variation and latency jitter
LINK_SPEED_MBPS  = 10_000
MIN_PKT_SIZE     = 20

# Pre-compiled struct for the big-endian prefix (bytes 0-11)
_PREFIX = struct.Struct('>IIHH')    # src_ip(4) dst_ip(4) sp(2) dp(2)


def ip_to_str(ip_int: int) -> str:
    return (f"{(ip_int>>24)&0xFF}.{(ip_int>>16)&0xFF}"
            f".{(ip_int>>8)&0xFF}.{ip_int&0xFF}")


def make_flow_id(src_ip, dst_ip, src_port, dst_port, proto) -> str:
    return f"{ip_to_str(src_ip)},{ip_to_str(dst_ip)},{src_port},{dst_port},{proto}"


def parse_caida_dat(filepath: str):
    """
    Generator – yields one dict per packet.

    Format (mixed endianness, parsed field-by-field):
        [0:4]   src_ip    big-endian uint32
        [4:8]   dst_ip    big-endian uint32
        [8:10]  src_port  big-endian uint16
        [10:12] dst_port  big-endian uint16
        [12]    protocol  uint8
        [13:21] timestamp little-endian float64  (Unix seconds, µs precision)
        [21:23] pkt_size  little-endian uint16
        [23]    tcp_flags uint8
    """
    with open(filepath, "rb") as f:
        while True:
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                break
            src_ip, dst_ip, src_port, dst_port = _PREFIX.unpack(raw[:12])
            proto    = raw[12]
            ts_sec   = struct.unpack('<d', raw[13:21])[0]   # float64 LE
            pkt_size = struct.unpack('<H', raw[21:23])[0]   # uint16 LE
            flags    = raw[23]
            yield {
                "src_ip":   src_ip,
                "dst_ip":   dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "proto":    proto,
                "flags":    flags,
                "pkt_size": max(pkt_size, MIN_PKT_SIZE),
                "ts_sec":   ts_sec,
            }


def simulate_fifo_queue(packets: list, link_mbps: float = LINK_SPEED_MBPS) -> list:
    """
    FIFO single-port queuing model.

    All packets share one output queue; they are served in arrival order.
    For each packet:
        ingress_us = ts_sec × 1_000_000          (float µs)
        tx_delay   = pkt_size / (link_mbps / 8)  (µs)
        start_tx   = max(ingress_us, queue_free_at)
        egress_us  = start_tx + tx_delay
        latency    = egress_us - ingress_us
    """
    bytes_per_us = link_mbps / 8.0
    packets.sort(key=lambda p: p["ts_sec"])

    queue_free_at = 0.0
    results = []

    for pkt in packets:
        ingress_us = pkt["ts_sec"] * 1_000_000.0
        tx_delay   = pkt["pkt_size"] / bytes_per_us
        start_tx   = max(ingress_us, queue_free_at)
        egress_us  = start_tx + tx_delay
        queue_free_at = egress_us
        latency_us    = egress_us - ingress_us

        results.append({
            "flow_id":    make_flow_id(pkt["src_ip"], pkt["dst_ip"],
                                       pkt["src_port"], pkt["dst_port"], pkt["proto"]),
            "ingress_ts": ingress_us,
            "egress_ts":  egress_us,
            "latency_us": round(latency_us, 4),
            "pkt_size":   pkt["pkt_size"],
        })

    return results


def run(dat_file: str, out_csv: str,
        max_packets: int = 0,
        link_mbps: float = LINK_SPEED_MBPS) -> list:
    """Parse → simulate → write CSV."""
    print(f"[queuing] Reading CAIDA trace : {dat_file}")
    print(f"[queuing] Link speed          : {link_mbps:,} Mbps")
    print(f"[queuing] Format              : "
          "src_ip(4,BE) dst_ip(4,BE) sp(2,BE) dp(2,BE) proto(1) "
          "ts_f64(8,LE) pkt_size(2,LE) flags(1) = 24 bytes")

    packets = []
    for i, pkt in enumerate(parse_caida_dat(dat_file)):
        packets.append(pkt)
        if max_packets and i + 1 >= max_packets:
            break

    if not packets:
        print("[queuing] ERROR: No packets – check file path.")
        return []

    ts_span = packets[-1]["ts_sec"] - packets[0]["ts_sec"]
    print(f"[queuing] Loaded {len(packets):,} packets  |  trace span ≈ {ts_span:.3f} s")

    print("[queuing] Simulating FIFO queue …")
    results = simulate_fifo_queue(packets, link_mbps=link_mbps)

    flows = {r["flow_id"] for r in results}
    lats  = sorted(r["latency_us"] for r in results)
    print(f"[queuing] Distinct flows  : {len(flows):,}")
    print(f"[queuing] Latency (µs)    : "
          f"min={lats[0]:.2f}  median={lats[len(lats)//2]:.2f}  "
          f"p99={lats[int(len(lats)*0.99)]:.2f}  max={lats[-1]:.2f}")

    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"[queuing] Dataset written → {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAIDA FIFO queuing simulation")
    parser.add_argument("dat_file",    help="Path to CAIDA .dat trace file")
    parser.add_argument("--out",       default="data/latency_dataset.csv")
    parser.add_argument("--max-pkts",  type=int,   default=0)
    parser.add_argument("--link-mbps", type=float, default=LINK_SPEED_MBPS,
                        help=f"Link speed Mbps (default {LINK_SPEED_MBPS})")
    args = parser.parse_args()
    run(args.dat_file, args.out, args.max_pkts, args.link_mbps)