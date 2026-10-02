"""
Adaptive Quotient Filter (AQF) for O(1) probabilistic client
authentication (Section 4.2.1 of the paper, Algorithm 1).

A client credential (simulated Unified Authentication Parameter Set:
client id, hashed device id, public-key hash) is hashed with SHA-256 to a
fingerprint; the low `remainder_bits` (r) bits of the upper part form the
remainder stored in a table slot chosen by the quotient (bucket index),
with linear probing on collision.

Adaptive step (matches Algorithm 1 / Theorem 1): after every enrolment the
closed-form false-positive bound  eps = 1 - (1 - 2^-r)^n  is recomputed;
if it exceeds `target_fpr`, the remainder width r is increased by one bit
and every enrolled credential is re-hashed into a fresh table, repeating
until eps <= target_fpr.

Not implemented (the manuscript states this explicitly): any check on the
numerical content of client updates (e.g. gradient-norm range).
"""
from dataclasses import dataclass, field
import hashlib
import math


def make_credential(client_id: str) -> str:
    """Simulated UAPS credential for a client: identifier, hashed device
    id and public-key hash. In a real deployment the device id and key
    hash come from the enrolment process; here they are derived
    deterministically from the client id so experiments are repeatable."""
    device_hash = hashlib.sha256(f"device:{client_id}".encode()).hexdigest()
    pk_hash = hashlib.sha256(f"pubkey:{client_id}".encode()).hexdigest()
    return f"{client_id}|{device_hash}|{pk_hash}"


def _hash_to_int(value: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + str(value)).encode("utf-8")).hexdigest(), 16)


@dataclass
class AdaptiveQuotientFilter:
    capacity: int = 1024
    remainder_bits: int = 16
    target_fpr: float = 0.001
    _table: dict = field(default_factory=dict, repr=False)
    _credentials: list = field(default_factory=list, repr=False)
    n_rebuilds: int = 0

    def __post_init__(self):
        self.capacity = 1 << max(1, int(math.ceil(math.log2(self.capacity))))
        self._quotient_bits = int(math.log2(self.capacity))

    def _fingerprint(self, credential: str):
        h = _hash_to_int(credential, salt="aqf")
        return h % self.capacity, (h >> self._quotient_bits) % (1 << self.remainder_bits)

    def _insert(self, credential: str):
        q, r = self._fingerprint(credential)
        slot = q
        while slot in self._table:
            slot = (slot + 1) % self.capacity
        self._table[slot] = r

    def theoretical_fpr(self) -> float:
        n = max(1, len(self._credentials))
        return 1 - (1 - 2 ** (-self.remainder_bits)) ** n

    def register(self, credential: str):
        """Enrol a legitimate client; may trigger an adaptive rebuild."""
        self._credentials.append(credential)
        if len(self._credentials) > self.capacity // 2:   # keep load <= 0.5
            self.capacity *= 2
            self._quotient_bits += 1
            self._rehash()
        else:
            self._insert(credential)
        while self.theoretical_fpr() > self.target_fpr:
            self.remainder_bits += 1
            self.n_rebuilds += 1
            self._rehash()

    def _rehash(self):
        self._table = {}
        for c in self._credentials:
            self._insert(c)

    def authenticate(self, credential: str) -> bool:
        """True iff the credential's fingerprint matches an enrolled slot
        (no false negatives; false positives with probability <= eps)."""
        q, r = self._fingerprint(credential)
        slot = q
        while slot in self._table:
            if self._table[slot] == r:
                return True
            slot = (slot + 1) % self.capacity
        return False
