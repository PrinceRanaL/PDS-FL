"""
Semaphore Bloom Filter (SemBF) for round-scoped update deduplication
(Section 4.2.2 of the paper, Algorithm 2).

The submission key is SK_{t,k} = SHA-256(client_id || round || nonce_t),
where nonce_t is a fresh server-side random value drawn by reset() at the
start of every round. The first submission for a key sets its k bit
positions (LOCKED) and is accepted; a second submission with the same key
finds all positions set and is rejected. Scope: duplicates within one
round only; the bit array is cleared every round.
"""
from dataclasses import dataclass, field
import hashlib
import secrets


def _positions(key: str, num_hashes: int, size: int):
    for i in range(num_hashes):
        yield int(hashlib.sha256(f"{i}:{key}".encode("utf-8")).hexdigest(), 16) % size


@dataclass
class SemaphoreBloomFilter:
    size: int = 8192
    num_hashes: int = 4
    _bits: bytearray = field(default_factory=bytearray, repr=False)
    _nonce: str = ""
    flags: dict = field(default_factory=dict)   # client_id -> rejected-duplicate count

    def __post_init__(self):
        self.reset()

    def reset(self):
        """Start of a new round: clear all semaphores, draw a fresh nonce."""
        self._bits = bytearray(self.size)
        self._nonce = secrets.token_hex(8)

    def query_and_lock(self, client_id: str, round_idx: int) -> bool:
        """True = first submission this round (accepted, now locked);
        False = duplicate (rejected)."""
        key = f"{client_id}|{round_idx}|{self._nonce}"
        pos = list(_positions(key, self.num_hashes, self.size))
        if all(self._bits[p] for p in pos):
            self.flags[client_id] = self.flags.get(client_id, 0) + 1   # anomaly flag
            return False
        for p in pos:
            self._bits[p] = 1
        return True
