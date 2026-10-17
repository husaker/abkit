import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationResult:
    p_values: np.ndarray
    alpha: float

    @property
    def n_simulations(self) -> int:
        return len(self.p_values)

    @property
    def rejection_rate(self) -> float:
        return (self.p_values < self.alpha).mean()

    @property
    def mc_standard_error(self) -> float:

        std = (self.p_values < self.alpha).std(ddof=0)
        sqrt_n = np.sqrt(len(self.p_values))

        return std / sqrt_n


def split_users(n_users: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign treatment group to n_users // 2 users. True means treatment"""

    treated = np.zeros(n_users, dtype=bool) # start with everyone in control
    chosen = rng.permutation(n_users)[: n_users // 2]
    treated[chosen] = True

    return treated


def multiply(values: np.ndarray, lift: float, rng=np.random.default_rng(0)) -> np.ndarray:
    """Multiply values with lift"""

    return values * (1 + lift)

def add_conversions(values: np.ndarray, lift: float, rng=np.random.default_rng(0)) -> np.ndarray:
    """Convert a random share of the non-converters"""

    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Values must be only 0s and 1s")
    new_values = np.ones_like(values)

    base_conv = values.mean() # get base convertion rate
    non_converted = np.where(values == 0)[0] # get idx of non-converters
    q = base_conv * lift / (1 - base_conv) # calculate cr-lift
    print(q)
    if q < 0 or q > 1:
        raise ValueError(f"convertion must be between 0 and 1")

    num_new_converted = np.ceil(q * len(non_converted)).astype(int)
    print(num_new_converted)
    size = min(num_new_converted, len(non_converted))
    print(size)
    chosen = rng.choice(non_converted, size=size, replace=False)

    new_values[non_converted] = 0
    new_values[chosen] = 1 # assign convertion to non-converters

    return new_values

def run_aa(values: np.ndarray, test, n_simulations=1000, alpha=0.05, seed=0) -> SimulationResult:
    """ Simulate AA tests"""

    rng = np.random.default_rng(seed)
    sim = SimulationResult(p_values=np.zeros(n_simulations), alpha=alpha)

    for i in range(n_simulations):
        treated = split_users(len(values), rng=rng)
        control = values[~treated]
        treatment = values[treated]
        p_val = test(control, treatment)
        sim.p_values[i] = p_val

    return sim


def run_ab(values: np.ndarray, test, inject, lift: float, n_simulations=1000, alpha=0.05, seed=0) -> SimulationResult:
    """ Simulate AB tests"""

    rng = np.random.default_rng(seed)
    sim = SimulationResult(p_values=np.zeros(n_simulations), alpha=alpha)

    for i in range(n_simulations):
        treated = split_users(len(values), rng=rng)

        control = values[~treated]
        treatment = inject(values[treated], lift)

        p_val = test(control, treatment)
        sim.p_values[i] = p_val

    return sim








