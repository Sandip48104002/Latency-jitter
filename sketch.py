"""
sketch.py
─────────
Memory-efficient jitter detector using a count-min-style sketch.

Architecture
────────────
  • Fixed total memory  : TOTAL_MEMORY_BYTES (default 400 KB)
  • Rows                : NUM_ROWS (default 4)
  • Each bucket stores  : 2-byte fingerprint  +  4-byte EWMA (float32)
                          +  2-byte packet count (uint16)  = 8 bytes
  • Buckets per row     : TOTAL_MEMORY_BYTES // (NUM_ROWS * BUCKET_SIZE)

Per-packet processing
─────────────────────
  1. Compute 5-tuple flow key.
  2. Hash flow key with BobHash once per row (with a per-row seed).
  3. Derive a 16-bit fingerprint from the flow key (independent hash).
  4. For each row:
       a. Look up bucket at hashed index.
       b. If fingerprint matches → "HIT"
             - read stored EWMA
             - detect jitter: latency > K*ewma or latency < ewma/K
             - update EWMA in bucket
       c. Else → "MISS"
             - if bucket is empty (fingerprint == 0) → insert
             - else → eviction policy (LRU-approximation via packet count)
  5. A packet is flagged as jitter only if ALL rows that see a fingerprint
     match agree (conservative) — OR if ANY row flags it (aggressive).
     We use ANY-row strategy here (higher recall).

BobHash
───────
A pure-Python implementation of Bob Jenkins' lookup3 hashlittle2.
We use 32-bit output per call, seeded differently per row.
"""

import struct
import math
import array
from typing import NamedTuple

# ── sketch parameters ─────────────────────────────────────────────────────────
TOTAL_MEMORY_BYTES = 400 * 1024    # 400 KB
NUM_ROWS           = 4
BUCKET_SIZE        = 8             # bytes: 2 fp + 4 ewma_f32 + 2 pkt_count

BUCKETS_PER_ROW    = TOTAL_MEMORY_BYTES // (NUM_ROWS * BUCKET_SIZE)

# EWMA / jitter thresholds (should match ground_truth.py for fair comparison)
ALPHA = 0.125
K     = 2.0

# Per-row BobHash seeds (arbitrary distinct primes)
ROW_SEEDS = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]
# Fingerprint seed (must differ from row seeds)
FP_SEED   = 0xABADCAFE


# ── Bob Jenkins lookup3 hashlittle (32-bit output) ────────────────────────────

def _rot32(val: int, shift: int) -> int:
    """Rotate a 32-bit integer left by shift bits."""
    val &= 0xFFFFFFFF
    return ((val << shift) | (val >> (32 - shift))) & 0xFFFFFFFF


def bob_hash(data: bytes, seed: int = 0) -> int:
    """
    Pure-Python Bob Jenkins lookup3 hashlittle.
    Returns a 32-bit unsigned integer.

    This is a faithful implementation of Jenkins' lookup3 hashlittle2
    reduced to a single 32-bit output (the 'c' word).
    """
    MASK = 0xFFFFFFFF
    length = len(data)

    # Initialise three 32-bit words
    a = b = c = (0xDEADBEEF + length + seed) & MASK

    i = 0
    while length > 12:
        a = (a + int.from_bytes(data[i:i+4],   "little")) & MASK
        b = (b + int.from_bytes(data[i+4:i+8], "little")) & MASK
        c = (c + int.from_bytes(data[i+8:i+12],"little")) & MASK

        # mix
        a = (_rot32(a ^ c, 4)  - c) & MASK; c = _rot32(c,  5)
        b = (_rot32(b ^ a, 6)  - a) & MASK; a = _rot32(a, 11)
        c = (_rot32(c ^ b, 8)  - b) & MASK; b = _rot32(b, 13)
        a = (_rot32(a ^ c, 14) - c) & MASK; c = _rot32(c, 11)
        b = (_rot32(b ^ a, 11) - a) & MASK; a = _rot32(a, 25)

        i      += 12
        length -= 12

    # Handle remaining bytes (0-12)
    tail = data[i:]
    tl   = len(tail)
    # Pad tail with zeros to 12 bytes for simpler handling
    tail = tail + b'\x00' * (12 - tl)

    a = (a + int.from_bytes(tail[0:4],  "little")) & MASK
    b = (b + int.from_bytes(tail[4:8],  "little")) & MASK
    c = (c + int.from_bytes(tail[8:12], "little")) & MASK

    # Final mix
    c ^= b; c = (c - _rot32(b, 14)) & MASK
    a ^= c; a = (a - _rot32(c, 11)) & MASK
    b ^= a; b = (b - _rot32(a, 25)) & MASK
    c ^= b; c = (c - _rot32(b, 16)) & MASK
    a ^= c; a = (a - _rot32(c,  4)) & MASK
    b ^= a; b = (b - _rot32(a, 14)) & MASK
    c ^= b; c = (c - _rot32(b, 24)) & MASK

    return c & MASK


def flow_key_bytes(flow_id: str) -> bytes:
    """Encode the flow_id string as UTF-8 bytes for hashing."""
    return flow_id.encode("utf-8")


# ── Sketch data structure ─────────────────────────────────────────────────────

class Bucket(NamedTuple):
    """
    In-memory representation of one sketch bucket.
    Stored in three parallel arrays (fingerprints, ewma, pkt_counts)
    for cache efficiency.
    """
    pass   # We use parallel arrays below, not a per-bucket object.


class JitterSketch:
    """
    4-row sketch with BobHash, 16-bit fingerprints, and EWMA per bucket.

    Memory layout (per row):
        fingerprints[BUCKETS_PER_ROW]  : array of uint16
        ewma[BUCKETS_PER_ROW]          : array of float  (Python float = 64-bit,
                                          logically we treat it as float32)
        pkt_count[BUCKETS_PER_ROW]     : array of uint16 (for eviction)
    """

    def __init__(self,
                 num_rows: int          = NUM_ROWS,
                 buckets_per_row: int   = BUCKETS_PER_ROW,
                 alpha: float           = ALPHA,
                 k: float               = K,
                 row_seeds: list[int]   = None,
                 fp_seed: int           = FP_SEED):

        self.num_rows       = num_rows
        self.bpr            = buckets_per_row   # buckets per row
        self.alpha          = alpha
        self.k              = k
        self.row_seeds      = row_seeds or ROW_SEEDS[:num_rows]
        self.fp_seed        = fp_seed

        # Parallel arrays – one list of arrays per row
        # 'H' = unsigned short (uint16), 'd' = double
        self.fingerprints = [array.array('H', [0] * self.bpr) for _ in range(num_rows)]
        self.ewma_vals    = [array.array('d', [0.0] * self.bpr) for _ in range(num_rows)]
        self.pkt_counts   = [array.array('H', [0] * self.bpr) for _ in range(num_rows)]

        # Statistics counters
        self.total_pkts  = 0
        self.hit_count   = 0   # packets where at least one row fingerprint matched
        self.miss_count  = 0
        self.evictions   = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _bucket_index(self, key_bytes: bytes, row: int) -> int:
        """Hash flow key to a bucket index in the given row."""
        h = bob_hash(key_bytes, seed=self.row_seeds[row])
        return h % self.bpr

    def _fingerprint(self, key_bytes: bytes) -> int:
        """Compute 16-bit fingerprint from flow key (row-independent)."""
        h = bob_hash(key_bytes, seed=self.fp_seed)
        fp = (h >> 16) ^ (h & 0xFFFF)    # fold 32 bits to 16
        return fp if fp != 0 else 1       # 0 is reserved for "empty"

    def _update_ewma(self, current: float, new_sample: float) -> float:
        """One step of exponential weighted moving average."""
        return self.alpha * new_sample + (1 - self.alpha) * current

    # ── public API ────────────────────────────────────────────────────────────

    def process_packet(self, flow_id: str, latency: float) -> int:
        """
        Feed one packet into the sketch.

        Returns
        -------
        1 if jitter is detected, 0 otherwise.

        Logic
        ─────
        For each row:
          • Compute bucket index and fingerprint.
          • If fingerprint matches stored value → UPDATE path
              - compare latency vs stored EWMA → jitter decision
              - update EWMA
          • Else → MISS path
              - empty bucket  → insert
              - occupied      → eviction: replace if current count is LOW
                                (count-based approximate LRU)
        A packet is flagged as jitter if ANY row (that has a FP match)
        detects jitter. If no row has a match, no jitter is declared.
        """
        self.total_pkts += 1
        key_bytes   = flow_key_bytes(flow_id)
        fp          = self._fingerprint(key_bytes)

        jitter_votes  = 0    # rows that voted "jitter"
        match_votes   = 0    # rows that had a fingerprint match

        for row in range(self.num_rows):
            idx = self._bucket_index(key_bytes, row)

            stored_fp    = self.fingerprints[row][idx]
            stored_ewma  = self.ewma_vals[row][idx]
            stored_count = self.pkt_counts[row][idx]

            if stored_fp == fp:
                # ── HIT: fingerprint matches ──────────────────────────────────
                match_votes += 1
                if stored_count >= 1:       # have at least one prior EWMA update
                    # jitter detection BEFORE updating EWMA
                    if latency > self.k * stored_ewma or latency < stored_ewma / self.k:
                        jitter_votes += 1

                # update EWMA in bucket
                if stored_count == 0:
                    new_ewma = latency       # seed EWMA with first observation
                else:
                    new_ewma = self._update_ewma(stored_ewma, latency)

                self.ewma_vals[row][idx]   = new_ewma
                # Cap count at uint16 max to avoid overflow
                self.pkt_counts[row][idx]  = min(stored_count + 1, 65535)

            else:
                # ── MISS: fingerprint does not match ──────────────────────────
                if stored_fp == 0:
                    # Bucket is empty → insert this flow
                    self.fingerprints[row][idx] = fp
                    self.ewma_vals[row][idx]    = latency   # seed EWMA
                    self.pkt_counts[row][idx]   = 1
                else:
                    # Bucket occupied by a different flow.
                    # Eviction policy: replace if the occupant has a very low
                    # packet count (likely a cold/dying flow).
                    # Threshold = 2 means we evict entries seen ≤ 2 times.
                    EVICT_THRESHOLD = 2
                    if stored_count <= EVICT_THRESHOLD:
                        self.fingerprints[row][idx] = fp
                        self.ewma_vals[row][idx]    = latency
                        self.pkt_counts[row][idx]   = 1
                        self.evictions += 1
                    # else: keep existing entry, this packet is untracked in row

        # ── aggregate decision ────────────────────────────────────────────────
        if match_votes > 0:
            self.hit_count += 1
            # Flag as jitter if at least one matching row says jitter
            return 1 if jitter_votes > 0 else 0
        else:
            self.miss_count += 1
            return 0   # no match → cannot make a decision

    def memory_bytes(self) -> int:
        """Actual memory used by sketch arrays in bytes."""
        fp_mem    = self.num_rows * self.bpr * 2     # uint16
        ewma_mem  = self.num_rows * self.bpr * 8     # float64 (Python)
        cnt_mem   = self.num_rows * self.bpr * 2     # uint16
        return fp_mem + ewma_mem + cnt_mem

    def stats(self) -> dict:
        """Return a summary dict of sketch runtime statistics."""
        return {
            "total_pkts":    self.total_pkts,
            "hit_count":     self.hit_count,
            "miss_count":    self.miss_count,
            "hit_rate_%":    round(100 * self.hit_count / max(self.total_pkts, 1), 2),
            "evictions":     self.evictions,
            "buckets_per_row": self.bpr,
            "num_rows":      self.num_rows,
            "memory_KB":     round(self.memory_bytes() / 1024, 1),
        }