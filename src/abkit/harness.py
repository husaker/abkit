import numpy as np
from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class SimulationResult:
    """P-values from a batch of simulated experiments.

    run_aa and run_ab build it. In A/A simulations the rejection rate estimates the
    test's Type I error rate; in A/B simulations it estimates the test's power.

    Attributes
    ----------
    p_values : np.ndarray
        One p-value per simulated experiment.
    alpha : float
        Significance level used to count rejections.
    n_simulations : int
        Number of simulated experiments.
    rejection_rate : float
        Share of experiments with a p-value below alpha.
    mc_standard_error : float
        Monte Carlo standard error of the rejection rate, sqrt(r(1 - r) / n_simulations).
    """

    p_values: np.ndarray
    alpha: float

    @property
    def n_simulations(self) -> int:
        return len(self.p_values)

    @property
    def rejection_rate(self) -> float:
        return (self.p_values <= self.alpha).mean()

    @property
    def mc_standard_error(self) -> float:

        std = (self.p_values < self.alpha).std(ddof=0)
        sqrt_n = np.sqrt(len(self.p_values))

        return std / sqrt_n

TestFunction = Callable[[np.ndarray, np.ndarray], float]                   # (control, treatment) -> p-value
Injector = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]  # (values, lift, rng) -> new values


def split_users(n_users: int, rng: np.random.Generator, split: float=0.5) -> np.ndarray:
    """Randomly assign users to control and treatment.

    Exactly int(n_users * split) users, picked uniformly at random, go to treatment.

    Parameters
    ----------
    n_users : int
        Number of users to assign.
    rng : np.random.Generator
        Source of randomness; a seeded generator makes the split reproducible.
    split : float, default 0.5
        Share of users in the treatment group.

    Returns
    -------
    np.ndarray
        Boolean mask of length n_users; True marks a treatment user.
    """

    treated = np.zeros(n_users, dtype=bool) # start with everyone in control
    treatment_group = int(n_users * split)
    chosen = rng.permutation(n_users)[: treatment_group]
    treated[chosen] = True

    return treated


def multiply(values: np.ndarray, lift: float, rng: np.random.Generator) -> np.ndarray:
    """Return the values multiplied by (1 + lift).

    Injects a relative effect into a continuous metric such as revenue. The mean and
    the standard deviation both grow by the factor (1 + lift).

    Parameters
    ----------
    values : np.ndarray
        Treatment-group values.
    lift : float
        Relative effect, e.g. 0.05 for +5%.
    rng : np.random.Generator
        Not used. Every injector takes it, so run_ab can call them all the same way.

    Returns
    -------
    np.ndarray
        A new array; the input isn't modified.
    """

    return values * (1 + lift)

def add_conversions(values: np.ndarray, lift: float, rng: np.random.Generator) -> np.ndarray:
    """Convert random non-converters so the conversion rate grows by a relative lift.

    With current rate p, a share q = p * lift / (1 - p) of the non-converters switch
    from 0 to 1, which moves the rate to about p * (1 + lift). The number of new
    converters is rounded to a whole user.

    Parameters
    ----------
    values : np.ndarray
        Treatment-group conversions, only 0s and 1s.
    lift : float
        Relative effect on the conversion rate, e.g. 0.05 for +5%; can't be negative.
    rng : np.random.Generator
        Picks which non-converters convert.

    Returns
    -------
    np.ndarray
        A new array of 0s and 1s; the input isn't modified.

    Raises
    ------
    ValueError
        If values contain anything other than 0 and 1, or q falls outside [0, 1]
        (a negative lift, or a target rate above 100%).
    """

    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Values must be only 0s and 1s")
    new_values = np.ones_like(values)

    base_conv = values.mean() # get base convertion rate
    non_converted = np.where(values == 0)[0] # get idx of non-converters
    q = base_conv * lift / (1 - base_conv) # calculate cr-lift
    if q < 0 or q > 1:
        raise ValueError(f"convertion must be between 0 and 1")

    num_new_converted = round(q * len(non_converted))
    size = min(num_new_converted, len(non_converted))
    chosen = rng.choice(non_converted, size=size, replace=False)

    new_values[non_converted] = 0
    new_values[chosen] = 1 # assign convertion to non-converters

    return new_values

def run_aa(values: np.ndarray, test: TestFunction, n_simulations:int = 1000,
           alpha: float=0.05, seed: int =0, split: float=0.5) -> SimulationResult:
    """Run simulated A/A tests and collect their p-values.

    Each simulation splits the users at random and compares the two groups with no
    effect added, so the rejection rate estimates the test's Type I error rate.

    Parameters
    ----------
    values : np.ndarray
        One value per randomization unit, e.g. per user.
    test : TestFunction
        Takes (control, treatment) arrays and returns a p-value.
    n_simulations : int, default 1000
        Number of simulated experiments.
    alpha : float, default 0.05
        Significance level stored in the result for counting rejections.
    seed : int, default 0
        Seed for the random splits; the same seed gives the same splits.
    split : float, default 0.5
        Share of users in the treatment group.

    Returns
    -------
    SimulationResult
        One p-value per simulation.
    """

    rng = np.random.default_rng(seed)
    p_values = []

    for i in range(n_simulations):
        treated = split_users(len(values), split=split, rng=rng)
        control = values[~treated]
        treatment = values[treated]
        p_val = test(control, treatment)
        p_values.append(p_val)

    sim = SimulationResult(p_values=np.array(p_values), alpha=alpha)

    return sim


def run_ab(values: np.ndarray, test: TestFunction, inject: Injector, lift: float,
           n_simulations:int =1000, alpha: float=0.05, seed: int=0,
           split: float=0.5) -> SimulationResult:
    """Run simulated A/B tests with a known effect and collect their p-values.

    Each simulation splits the users at random, adds the effect to the treatment group
    with inject, and compares the groups, so the rejection rate estimates the test's
    power for that effect.

    Parameters
    ----------
    values : np.ndarray
        One value per randomization unit, e.g. per user.
    test : TestFunction
        Takes (control, treatment) arrays and returns a p-value.
    inject : Injector
        Takes (values, lift, rng) and returns the treatment values with the effect
        added: add_conversions for a 0/1 metric, multiply for a continuous one.
    lift : float
        Effect passed to inject; relative for both built-in injectors (0.05 = +5%).
    n_simulations : int, default 1000
        Number of simulated experiments.
    alpha : float, default 0.05
        Significance level stored in the result for counting rejections.
    seed : int, default 0
        Seed for the random splits and the injection; the same seed gives the same run.
    split : float, default 0.5
        Share of users in the treatment group.

    Returns
    -------
    SimulationResult
        One p-value per simulation.
    """

    rng = np.random.default_rng(seed)
    p_values = []

    for i in range(n_simulations):
        treated = split_users(len(values), split=split, rng=rng)

        control = values[~treated]
        treatment = inject(values[treated], lift, rng)

        p_val = test(control, treatment)
        p_values.append(p_val)

    sim = SimulationResult(p_values=np.array(p_values), alpha=alpha)

    return sim








