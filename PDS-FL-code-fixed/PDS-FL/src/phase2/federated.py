"""
Phase 2: Secure Federated Learning Deployment (Section 4.2 of the paper).

The manuscript's Eq. (10)-(11) describe FedProx/FedAvg for a fixed-length
parameter vector. A boosted-tree ensemble has no such vector, so this
module implements the functional (federated gradient-boosting) analogue:

  - Each round, every authenticated, non-duplicate client starts from the
    current GLOBAL model prediction on its own data and performs E local
    boosting steps, each fitting a small tree to its current residual.
  - FedProx (mu > 0): the fitting target at each local step is reduced by
    mu * (local_pred - global_start_pred), the client's accumulated drift
    from the round's global model -- the analogue of mu*(w_k - w_t).
  - FedAvg (mu = 0): no such penalty.
  - Aggregation (Eq. 11 analogue): each client's net local update is added
    to the global ensemble with weight eta * n_k / sum_j n_j, and the
    global prediction is updated on EVERY client's data and on the test set.
"""

from dataclasses import dataclass, field
import time

import numpy as np
from lightgbm import LGBMRegressor

from src.phase2.aqf import AdaptiveQuotientFilter, make_credential
from src.phase2.sembf import SemaphoreBloomFilter
from src.utils.metrics import regression_metrics


@dataclass
class FederatedBoostingConfig:
    mu: float = 0.0                 # FedProx proximal coefficient (0 = FedAvg)
    eta: float = 0.15               # global step size
    local_epochs: int = 3           # local boosting steps per round before aggregation
    local_learning_rate: float = 0.1
    local_max_depth: int = 4
    local_num_leaves: int = 15
    reg_lambda: float = 2.0         # L2 leaf regularisation (controls overfitting)
    rounds: int = 60
    seed: int = 42


class FederatedBoostingSimulator:
    def __init__(self, train_by_city: dict, X_test: np.ndarray,
                 y_test: np.ndarray, config: FederatedBoostingConfig):
        """
        train_by_city: {city_name: (X_city, y_city)}
        """
        self.train_by_city = train_by_city
        self.X_test = X_test
        self.y_test = y_test
        self.cfg = config

        self.aqf = AdaptiveQuotientFilter(capacity=64, target_fpr=0.001)
        for city in train_by_city:
            self.aqf.register(make_credential(city))
        self.sembf = SemaphoreBloomFilter(size=4096, num_hashes=4)

        all_y = np.concatenate([y for _, y in train_by_city.values()])
        self.base_prediction = float(np.mean(all_y))

        # Global-model prediction on every client's training rows, kept in
        # one array (slices per city) so that each aggregated update is
        # applied to all clients, not only to the client that produced it.
        self._cities = list(train_by_city)
        self._X_all = np.vstack([train_by_city[c][0] for c in self._cities])
        self._slices, start = {}, 0
        for c in self._cities:
            n = len(train_by_city[c][1])
            self._slices[c] = slice(start, start + n)
            start += n
        self.global_pred_train = np.full(start, self.base_prediction)
        self.cum_pred_test = np.full(len(y_test), self.base_prediction)
        self.ensemble = []  # list of (weight, booster)
        self.history = []   # list of dict per round

    def _round_clients(self, round_idx: int):
        """Authenticate + dedup every registered client for this round.
        Returns the list of city names accepted to contribute."""
        self.sembf.reset()
        accepted = []
        for city in self.train_by_city:
            if not self.aqf.authenticate(make_credential(city)):
                continue
            if not self.sembf.query_and_lock(city, round_idx):
                continue
            accepted.append(city)
        return accepted

    def _local_update(self, city: str, round_idx: int):
        """Perform E local boosting epochs for one client, starting from
        the current global broadcast prediction. Returns the list of
        per-epoch trees fit (each already includes LightGBM's internal
        learning-rate shrinkage, so summing tree.predict(X) over the
        list reproduces this client's net local prediction delta)."""
        cfg = self.cfg
        X_c, y_c = self.train_by_city[city]
        global_start_pred = self.global_pred_train[self._slices[city]].copy()
        local_pred = global_start_pred.copy()
        trees = []
        for e in range(cfg.local_epochs):
            residual = y_c - local_pred
            if cfg.mu > 0:
                drift = local_pred - global_start_pred
                target = residual - cfg.mu * drift
            else:
                target = residual
            tree = LGBMRegressor(
                n_estimators=1,
                max_depth=cfg.local_max_depth,
                num_leaves=cfg.local_num_leaves,
                learning_rate=cfg.local_learning_rate,
                reg_lambda=cfg.reg_lambda,
                min_child_samples=50,
                verbosity=-1,
                random_state=cfg.seed + round_idx * 100 + e,
            )
            tree.fit(X_c, target)
            local_pred = local_pred + tree.predict(X_c)
            trees.append(tree)
        return trees

    def run(self, verbose: bool = True):
        cfg = self.cfg
        t0 = time.time()
        for t in range(1, cfg.rounds + 1):
            accepted = self._round_clients(t)
            contributions = []
            for city in accepted:
                trees = self._local_update(city, t)
                n_c = len(self.train_by_city[city][1])
                contributions.append((city, n_c, trees))

            total_n = sum(n for _, n, _ in contributions)
            for city, n_c, trees in contributions:
                weight = cfg.eta * (n_c / total_n)
                # Apply this client's update to the global model's
                # prediction on ALL clients' data and on the test set.
                train_delta = sum(tree.predict(self._X_all) for tree in trees)
                test_delta = sum(tree.predict(self.X_test) for tree in trees)
                self.global_pred_train += weight * train_delta
                self.cum_pred_test += weight * test_delta
                self.ensemble.extend([(weight, tree) for tree in trees])

            metrics = regression_metrics(self.y_test, self.cum_pred_test)
            metrics["round"] = t
            metrics["n_participants"] = len(accepted)
            self.history.append(metrics)
            if verbose and (t % 10 == 0 or t == 1):
                elapsed = time.time() - t0
                print(f"[round {t:3d}] R2={metrics['r2']:.4f} "
                      f"RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} "
                      f"participants={metrics['n_participants']} "
                      f"({elapsed:.1f}s elapsed)")
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(len(X), self.base_prediction)
        for weight, model in self.ensemble:
            pred += weight * model.predict(X)
        return pred
