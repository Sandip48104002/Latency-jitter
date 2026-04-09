"""
sketch.py
─────────
Memory-efficient jitter detector using a count-min-style sketch.

TWO VARIANTS are implemented in this file side-by-side for easy comparison:

  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │  JitterSketch       │  WITH fingerprint (original)                     │
  │                     │  Bucket = fingerprint(2B) + ewma(8B) + count(2B) │
  │                     │  = 12 B/bucket → 8,533 buckets/row @ 400 KB      │
  │                     │  Has HIT / MISS logic + eviction policy          │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │  JitterSketchNoFP   │  WITHOUT fingerprint (new, this change)          │
  │                     │  Bucket = ewma(8B) + count(2B) = 10 B/bucket     │
  │                     │  → 10,240 buckets/row @ 400 KB                   │
  │                     │  Every hash always hits its bucket (no eviction) │
  │                     │  Hash collisions silently corrupt EWMA            │
  └─────────────────────┴──────────────────────────────────────────────────┘

Without fingerprint:
  • No HIT/MISS distinction – every packet unconditionally reads and writes
    the bucket it hashes to.
  • No eviction needed – the bucket is always "used" by whoever hashes there.
  • Collision effect: two different flows that map to the same bucket will
    mix their EWMA values, producing spurious jitter or masking real jitter.
    This is the core accuracy cost of dropping the fingerprint guard.
  • Benefit: ~20% more buckets for the same memory (10 B vs 12 B per bucket),
    slightly lower collision probability, and simpler/faster per-packet logic.

BobHash
───────
A pure-Python implementation of Bob Jenkins' lookup3 hashlittle2.
32-bit output, seeded independently per row.
"""

import array
import mmh3

# ── shared sketch parameters ──────────────────────────────────────────────────
TOTAL_MEMORY_BYTES = 700 * 1024    # 400 KB hard budget

NUM_ROWS           = 4

# Per-row BobHash seeds (arbitrary distinct constants)
ROW_SEEDS = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]

# Fingerprint seed – must differ from all row seeds (used only by JitterSketch)
FP_SEED = 0xABADCAFE

# EWMA / jitter thresholds – must match ground_truth.py for fair comparison
ALPHA = 0.125
K     = 2.0

# ── bucket sizes and derived bucket counts ────────────────────────────────────
#
#  WITH fingerprint:    fp(2B) + ewma(8B) + count(2B) = 12 bytes / bucket
#  WITHOUT fingerprint:          ewma(8B) + count(2B) = 10 bytes / bucket
#
_BUCKET_SIZE_FP   = 12
_BUCKET_SIZE_NOFP = 10

BUCKETS_PER_ROW_FP   = TOTAL_MEMORY_BYTES // (NUM_ROWS * _BUCKET_SIZE_FP)
BUCKETS_PER_ROW_NOFP = TOTAL_MEMORY_BYTES // (NUM_ROWS * _BUCKET_SIZE_NOFP)

# Keep legacy name for backward-compat with main.py import
BUCKETS_PER_ROW = BUCKETS_PER_ROW_FP


# ── Bob Jenkins lookup3 hashlittle (32-bit) ───────────────────────────────────

def fast_hash(data: bytes, seed: int = 0) -> int:
    """
    Fast, seeded 32-bit hash using the mmh3 library (MurmurHash3).
    Returns an unsigned 32-bit integer.
    """
    return mmh3.hash(data, seed, signed=False)


def flow_key_bytes(flow_id: str) -> bytes:
    """Encode the 5-tuple flow_id string as UTF-8 bytes for hashing."""
    return flow_id.encode("utf-8")


# ═════════════════════════════════════════════════════════════════════════════
#  VARIANT 1 – WITH fingerprint  (original behaviour, unchanged)
# ═════════════════════════════════════════════════════════════════════════════

class JitterSketch:
    """
    4-row sketch WITH 16-bit fingerprints.

    Bucket layout per row (12 bytes each):
        fingerprints[bpr]  : uint16  – identifies which flow owns the bucket
        ewma_vals[bpr]     : float64 – EWMA latency for that flow
        pkt_counts[bpr]    : uint16  – packets seen (used for eviction & seeding)

    Per-packet logic:
        1. Hash flow key → bucket index (per row, different seed each row).
        2. Compute 16-bit fingerprint (single seed, row-independent).
        3. If stored fingerprint matches → HIT: detect jitter, update EWMA.
           If stored fingerprint is 0   → EMPTY: insert this flow.
           Else                         → COLLISION: evict if occupant count ≤ 2.
        4. Flag jitter if ANY matched row voted jitter.
           If no row matched → no decision (miss).
    """

    def __init__(self,
                 num_rows: int        = NUM_ROWS,
                 buckets_per_row: int = BUCKETS_PER_ROW_FP,
                 alpha: float         = ALPHA,
                 k: float             = K,
                 row_seeds            = None,
                 fp_seed: int         = FP_SEED):

        self.num_rows  = num_rows
        self.bpr       = buckets_per_row
        self.alpha     = alpha
        self.k         = k
        self.row_seeds = row_seeds or ROW_SEEDS[:num_rows]
        self.fp_seed   = fp_seed

        # Parallel arrays per row  ('H'=uint16, 'd'=float64)
        self.fingerprints = [array.array('H', [0]   * self.bpr) for _ in range(num_rows)]
        self.ewma_vals    = [array.array('d', [0.0] * self.bpr) for _ in range(num_rows)]
        self.pkt_counts   = [array.array('H', [0]   * self.bpr) for _ in range(num_rows)]

        # Runtime counters
        self.total_pkts = 0
        self.hit_count  = 0   # pkts where ≥1 row had a fingerprint match
        self.miss_count = 0   # pkts where 0 rows matched
        self.evictions  = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bucket_index(self, key_bytes: bytes, row: int) -> int:
        return fast_hash(key_bytes, seed=self.row_seeds[row]) % self.bpr

    def _fingerprint(self, key_bytes: bytes) -> int:
        """16-bit fingerprint; 0 is reserved for 'empty bucket'."""
        h  = fast_hash(key_bytes, seed=self.fp_seed)
        fp = (h >> 16) ^ (h & 0xFFFF)
        return fp if fp != 0 else 1

    def _ewma_step(self, current: float, sample: float) -> float:
        return self.alpha * sample + (1 - self.alpha) * current

    # ── main API ──────────────────────────────────────────────────────────────

    def process_packet(self, flow_id: str, latency: float) -> int:
        """
        Process one packet. Returns 1 if jitter detected, 0 otherwise.

        A 2-pass approach is used:
        1. HIT: Check all rows for a fingerprint match. If any exist, they are
           all updated, and any empty candidate buckets are filled. A jitter
           decision is made if at least one row matched.
        2. MISS: If no row had a matching fingerprint, the packet is a miss.
           The flow is inserted into an empty bucket if available. If all
           candidate buckets are occupied, it evicts the one with the
           lowest packet count across all rows. No jitter decision on a miss.
        """
        self.total_pkts += 1
        key   = flow_key_bytes(flow_id)
        fp    = self._fingerprint(key)

        indices = [self._bucket_index(key, row) for row in range(self.num_rows)]

        # --- Pass 1: Find hits, empty slots, and collisions ---
        hit_rows = []
        empty_rows = []
        collision_rows_with_counts = []

        for row in range(self.num_rows):
            idx = indices[row]
            stored_fp = self.fingerprints[row][idx]
            if stored_fp == fp:
                hit_rows.append(row)
            elif stored_fp == 0:
                empty_rows.append(row)
            else:
                count = self.pkt_counts[row][idx]
                collision_rows_with_counts.append((row, count))

        # --- Process based on findings ---
        if hit_rows:
            # --- HIT PATH ---
            self.hit_count += 1
            jitter_votes = 0

            # Update all matching rows
            for row in hit_rows:
                idx = indices[row]
                stored_ewma = self.ewma_vals[row][idx]
                stored_count = self.pkt_counts[row][idx]

                if stored_count >= 1:
                    if latency > self.k * stored_ewma or latency < stored_ewma / self.k:
                        jitter_votes += 1

                new_ewma = latency if stored_count == 0 else self._ewma_step(stored_ewma, latency)
                self.ewma_vals[row][idx] = new_ewma
                self.pkt_counts[row][idx] = min(stored_count + 1, 65535)

            # Fill all empty buckets
            for row in empty_rows:
                idx = indices[row]
                self.fingerprints[row][idx] = fp
                self.ewma_vals[row][idx] = latency
                self.pkt_counts[row][idx] = 1

            return 1 if jitter_votes > 0 else 0

        else:
            # --- MISS PATH ---
            self.miss_count += 1

            # Insert into an empty bucket if available (preferable to eviction)
            if empty_rows:
                row_to_fill = empty_rows[0]  # just pick the first one
                idx = indices[row_to_fill]
                self.fingerprints[row_to_fill][idx] = fp
                self.ewma_vals[row_to_fill][idx] = latency
                self.pkt_counts[row_to_fill][idx] = 1
                return 0

            # All candidate buckets are full. Evict the one with the lowest count.
            if collision_rows_with_counts:
                row_to_evict, _ = min(collision_rows_with_counts, key=lambda item: item[1])

                idx = indices[row_to_evict]
                self.fingerprints[row_to_evict][idx] = fp
                self.ewma_vals[row_to_evict][idx] = latency
                self.pkt_counts[row_to_evict][idx] = 1
                self.evictions += 1

            return 0

    def memory_bytes(self) -> int:
        return self.num_rows * self.bpr * _BUCKET_SIZE_FP

    def stats(self) -> dict:
        return {
            "variant":       "with_fingerprint",
            "total_pkts":    self.total_pkts,
            "hit_count":     self.hit_count,
            "miss_count":    self.miss_count,
            "hit_rate_%":    round(100 * self.hit_count / max(self.total_pkts, 1), 2),
            "evictions":     self.evictions,
            "buckets_per_row": self.bpr,
            "num_rows":      self.num_rows,
            "memory_KB":     round(self.memory_bytes() / 1024, 1),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  VARIANT 2 – WITHOUT fingerprint  (new, this change)
# ═════════════════════════════════════════════════════════════════════════════

class JitterSketchNoFP:
    """
    4-row sketch WITHOUT fingerprints — EWMA-only buckets.

    Bucket layout per row (10 bytes each):
        ewma_vals[bpr]   : float64 – EWMA latency stored at hashed index
        pkt_counts[bpr]  : uint16  – packets seen at this bucket

    Key differences vs JitterSketch:
    ─────────────────────────────────
    • No fingerprint array → bucket = 10 B instead of 12 B
      → 10,240 buckets/row vs 8,533 (20% more) at same 400 KB.
    • No HIT/MISS: every packet ALWAYS reads and writes its hashed bucket.
      There is no way to tell whether the stored EWMA belongs to this flow
      or to a different flow that hashes to the same bucket (hash collision).
    • No eviction: unnecessary because there is no ownership concept.
    • Collision effect: when two flows share a bucket their EWMA values
      mix, which can create false jitter alerts or suppress real ones.
      The accuracy impact compared to JitterSketch shows the value of the
      fingerprint guard.
    • All rows always produce a jitter vote (count ≥ 1 check still applies
      for the very first packet at a bucket, to avoid comparing against 0).
    """

    def __init__(self,
                 num_rows: int        = NUM_ROWS,
                 buckets_per_row: int = BUCKETS_PER_ROW_NOFP,
                 alpha: float         = ALPHA,
                 k: float             = K,
                 row_seeds            = None):

        self.num_rows  = num_rows
        self.bpr       = buckets_per_row
        self.alpha     = alpha
        self.k         = k
        self.row_seeds = row_seeds or ROW_SEEDS[:num_rows]

        # Only two arrays per row – no fingerprint storage at all
        self.ewma_vals  = [array.array('d', [0.0] * self.bpr) for _ in range(num_rows)]
        self.pkt_counts = [array.array('H', [0]   * self.bpr) for _ in range(num_rows)]

        # Runtime counters
        self.total_pkts   = 0
        self.jitter_count = 0   # total jitter decisions across all packets

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bucket_index(self, key_bytes: bytes, row: int) -> int:
        return fast_hash(key_bytes, seed=self.row_seeds[row]) % self.bpr

    def _ewma_step(self, current: float, sample: float) -> float:
        return self.alpha * sample + (1 - self.alpha) * current

    # ── main API ──────────────────────────────────────────────────────────────

    def process_packet(self, flow_id: str, latency: float) -> int:
        """
        Process one packet. Returns 1 if jitter detected, 0 otherwise.

        No fingerprint check — every packet unconditionally reads the EWMA
        stored at its hashed bucket index, makes a jitter decision, and
        updates the EWMA.

        Jitter is flagged if ANY row votes jitter.
        A row can only vote once it has seen ≥1 prior packet (count ≥ 1),
        to avoid comparing the very first latency sample against 0.
        """
        self.total_pkts += 1
        key = flow_key_bytes(flow_id)

        jitter_votes = 0

        for row in range(self.num_rows):
            idx = self._bucket_index(key, row)

            stored_ewma  = self.ewma_vals[row][idx]
            stored_count = self.pkt_counts[row][idx]

            # ── jitter detection (before EWMA update) ────────────────────────
            # Only test once the bucket has at least one prior sample.
            # Note: this bucket may belong to a DIFFERENT flow (collision),
            # which is exactly the accuracy cost being measured here.
            if stored_count >= 1:
                if latency > self.k * stored_ewma or latency < stored_ewma / self.k:
                    jitter_votes += 1

            # ── unconditional EWMA update ─────────────────────────────────────
            if stored_count == 0:
                new_ewma = latency          # seed: first sample at this bucket
            else:
                new_ewma = self._ewma_step(stored_ewma, latency)

            self.ewma_vals[row][idx]  = new_ewma
            self.pkt_counts[row][idx] = min(stored_count + 1, 65535)

        is_jitter = 1 if jitter_votes > 0 else 0
        self.jitter_count += is_jitter
        return is_jitter

    def memory_bytes(self) -> int:
        return self.num_rows * self.bpr * _BUCKET_SIZE_NOFP

    def stats(self) -> dict:
        return {
            "variant":         "no_fingerprint",
            "total_pkts":      self.total_pkts,
            "jitter_detected": self.jitter_count,
            "buckets_per_row": self.bpr,
            "num_rows":        self.num_rows,
            "memory_KB":       round(self.memory_bytes() / 1024, 1),
        }