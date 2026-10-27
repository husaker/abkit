from dataclasses import dataclass
import math
from statistics import NormalDist

@dataclass(frozen=True)
class DesignResult:
    """Power, Type S error and exaggeration ratio for an assumed true effect.

    Attributes
    ----------
    power : float
        Probability that the two-sided test is significant, in either direction.
    type_s : float
        Probability that a significant estimate has the wrong sign.
    exaggeration : float
        Expected |estimate| / |true effect| among significant estimates (Type M).
    """

    power: float
    type_s: float
    exaggeration: float


def power(effect: float, sd : float, n_control: int, n_treatment: int,
          alpha: float =0.05, sd_treatment: float | None = None) -> float:

    """Return the power of a two-sided z-test for a true difference in means.

    Parameters
    ----------
    effect : float
        True difference in means, treatment minus control, in the metric's own units
        (e.g. 0.003 for +5% on a 6% conversion rate). Its sign doesn't matter.
    sd : float
        Standard deviation of the metric per user, e.g. sqrt(p(1 - p)) for conversion.
        Used for both groups unless sd_treatment is given.
    n_control, n_treatment : int
        Users in each group.
    alpha : float, default 0.05
        Two-sided significance level.
    sd_treatment : float or None, default None
        Standard deviation per user in treatment, for a change that alters the spread:
        multiplying every value by 1.05 multiplies the SD by 1.05. None means the
        same as sd.

    Returns
    -------
    float
        Probability of a significant result, counting both tails.

    Raises
    ------
    ValueError
        If sd or sd_treatment is not positive, a group is empty, or alpha is outside
        (0, 1).
    """

    if sd_treatment is None:
        sd_treatment = sd
    else:
        sd_treatment = sd_treatment

    if sd <= 0:
        raise ValueError('sd must be positive')
    if sd_treatment <= 0:
        raise ValueError('sd_treatment must be positive')
    if n_control == 0 or n_treatment == 0:
        raise ValueError('control and treatment groups should be at least one')
    if alpha <= 0 or alpha > 1:
        raise ValueError('alpha must be between 0 and 1')

    se = math.sqrt(sd**2 / n_control + sd_treatment**2 / n_treatment)

    z_ast = NormalDist().inv_cdf(1 - alpha / 2)

    power = NormalDist().cdf(effect / se - z_ast) + NormalDist().cdf(-(effect / se) - z_ast)

    return power


def sample_size(effect: float, sd : float, target_power: float =0.8,
                alpha: float=0.05, split: float=0.5) -> tuple[int, int]:

    """Return the number of users each group needs to reach the target power.

    Uses the normal approximation for the total,
    N = (z* + z_beta)^2 * sd^2 * (1/split + 1/(1 - split)) / effect^2,
    where z* = Phi^-1(1 - alpha/2) and z_beta = Phi^-1(target_power), then gives
    each group its share of N, rounded up.

    Parameters
    ----------
    effect : float
        True difference in means to detect, treatment minus control, in the metric's
        own units (e.g. 0.003 for +5% on a 6% conversion rate). Its sign doesn't matter.
    sd : float
        Standard deviation of the metric per user, assumed equal in both groups, e.g.
        sqrt(p(1 - p)) for conversion.
    target_power : float, default 0.8
        Probability of a significant result when the true difference equals effect.
    alpha : float, default 0.05
        Two-sided significance level.
    split : float, default 0.5
        Share of users in the treatment group.

    Returns
    -------
    n_control : int
        Users in control.
    n_treatment : int
        Users in treatment.

    Raises
    ------
    ValueError
        If effect is 0, sd is not positive, or split, target_power or alpha is
        outside (0, 1).
    """

    if effect == 0:
        raise ValueError('effect cannot be 0')
    if sd <= 0:
        raise ValueError('sd must be positive')
    if split < 0 or split >= 1:
        raise ValueError('split must be between 0 and 1')
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be between 0 and 1')
    if target_power < 0 or target_power >= 1:
        raise ValueError('target_power must be between 0 and 1')


    z_ast = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(target_power)

    n = (z_ast + z_b)**2 * sd**2 * (1/split + 1/(1-split)) / effect**2

    n_treatment = math.ceil(n * split)
    n_control = math.ceil(n * (1 - split))

    return n_control, n_treatment



def mde(sd: float, n_control: int, n_treatment: int,
        target_power: float = 0.8, alpha: float = 0.05) -> float:
    """Return the smallest true difference a two-sided z-test detects with the target power.

    Parameters
    ----------
    sd : float
        Standard deviation of the metric per user, e.g. sqrt(p(1 - p)) for conversion.
    n_control, n_treatment : int
        Users in each group.
    target_power : float, default 0.8
        Probability of rejecting H0 when the true difference equals the MDE.
    alpha : float, default 0.05
        Two-sided significance level.

    Returns
    -------
    float
        The MDE, in the metric's own units (divide by the baseline for a relative MDE).

    Raises
    ------
    ValueError
        If sd is not positive, a group is empty, or target_power or alpha is outside (0, 1).
    """

    if sd <= 0:
        raise ValueError('sd must be positive')
    if n_control == 0 or n_treatment == 0:
        raise ValueError('control and treatment groups should be at least one')
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be between 0 and 1')
    if target_power < 0 or target_power > 1:
        raise ValueError('target_power must be between 0 and 1')


    z_ast = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(target_power)

    se = math.sqrt(sd**2 / n_control + sd**2 / n_treatment)

    mde = (z_ast + z_b) * se

    return mde

def design(effect: float, standard_error: float, alpha: float = 0.05) -> DesignResult:

    """Return power, Type S error and exaggeration ratio for an assumed true effect.

    The estimate is taken as normal around the true effect with the given standard
    error. Power and Type S follow Gelman and Carlin (2014); the exaggeration ratio
    uses the closed form of Lu, Qiu and Deng (2019).

    Parameters
    ----------
    effect : float
        Assumed true difference, treatment minus control, in the metric's own units.
        Take it from outside the experiment (earlier tests, a plausible minimum),
        not from the experiment's own estimate.
    standard_error : float
        Standard error of the estimated difference, e.g.
        sqrt(sd_control^2 / n_control + sd_treatment^2 / n_treatment).
    alpha : float, default 0.05
        Two-sided significance level.

    Returns
    -------
    DesignResult
        Power, Type S error and exaggeration ratio.

    Raises
    ------
    ValueError
        If effect is 0, standard_error is not positive, or alpha is outside (0, 1).
    """

    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be between 0 and 1')
    if standard_error <= 0:
        raise ValueError('standard_error must be positive')
    if effect == 0:
        raise ValueError('effect cannot be 0')

    lamd = effect / standard_error

    z_ast = NormalDist().inv_cdf(1 - alpha / 2)

    right = 1 - NormalDist().cdf(z_ast - lamd)
    wrong = NormalDist().cdf(-(z_ast) - lamd)

    power = right + wrong

    type_s = wrong / power
    significant = lamd * right + NormalDist().pdf(z_ast - lamd) - lamd * wrong + NormalDist().pdf(-(z_ast) - lamd)

    exaggeration = significant / (power * lamd)


    return DesignResult(power=power, type_s=type_s, exaggeration=exaggeration)



