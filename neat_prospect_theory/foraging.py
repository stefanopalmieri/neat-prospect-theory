"""Risk-sensitive optimal foraging task for the NEAT prospect-theory experiment.

This module implements the foraging decision used to test Risk-Sensitive
Optimal Foraging Theory (McDermott, Fowler & Smirnov, 2008) as an evolutionary
substrate for prospect-theory-style, reference-dependent risk preferences.

Each foraging bout an agent starts at some ``satiation`` level and chooses
between a low-variance ("safe") and a high-variance ("risky") food patch. It
survives the bout iff its resulting satiation stays at or above a survival
``threshold``. The *energy-budget rule* predicts the agent should be
risk-seeking when it is below threshold (it needs a windfall to survive) and
risk-averse when it is above it (it should protect the sure thing) -- exactly
the S-shaped, reference-dependent value function of prospect theory.

Why the first version of the experiment did not evolve
------------------------------------------------------
* **Fitness was a single coin flip.** Each genome was evaluated on one
  stochastic decision, collapsed to a 0/1 survival flag. Selection therefore
  acted on one Bernoulli trial per genome -- almost pure noise -- so NEAT could
  not tell a good policy from a lucky one. Here we average survival over many
  independent episodes, so fitness reflects the *policy*, not luck.
* **The task barely rewarded the right choice.** With the original parameters
  the survival threshold sat right at the mean outcome and the environmental
  variance dominated, leaving under a two-point survival gap between the safe
  and risky patches (see ``ORIGINAL`` vs. the analysis in the tests). The
  ``RISK_SENSITIVE`` task uses equal means with a variance split so the optimal
  choice genuinely flips with the agent's state.
"""
from dataclasses import dataclass
from math import erf, sqrt
from typing import Tuple

import numpy as np
import neat

_SQRT2 = sqrt(2.0)
_erf = np.vectorize(erf)


def _norm_cdf(z):
    """Standard-normal CDF (vectorised, no SciPy dependency)."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / _SQRT2))


@dataclass(frozen=True)
class ForagingTask:
    """Parameters of a single foraging decision."""

    threshold: float                 # survival floor for satiation
    sat_mean: float                  # mean starting satiation each bout
    sat_stdev: float                 # stdev of starting satiation
    safe: Tuple[float, float]        # (mu, sigma) of the safe patch reward
    risky: Tuple[float, float]       # (mu, sigma) of the risky patch reward


# The original notebook parameters, kept for reproducibility. The safe and
# risky patches have different means AND the threshold sits at the mean, so the
# two options give nearly identical survival odds -- almost no selection signal.
ORIGINAL = ForagingTask(
    threshold=10.0, sat_mean=10.0, sat_stdev=5.0, safe=(2.0, 1.0), risky=(3.0, 5.0)
)

# Equal-mean, variance-split task. The starting satiation spans both the deficit
# and the surplus regime, so the optimal choice flips with the agent's state and
# the energy-budget rule produces a clean prospect-theory signature.
RISK_SENSITIVE = ForagingTask(
    threshold=10.0, sat_mean=10.0, sat_stdev=6.0, safe=(5.0, 1.0), risky=(5.0, 6.0)
)


def decide_risky(net, satiation: float) -> bool:
    """Return ``True`` if the network chooses the risky patch at this state.

    The network sees a constant bias input and its current satiation; an output
    above ``0.5`` is read as "forage the risky patch".
    """
    return net.activate((1.0, satiation))[0] > 0.5


def evaluate_policy(net, task: ForagingTask, episodes: int, rng) -> float:
    """Return the fraction of ``episodes`` the policy survives.

    Averaging over many independent episodes is what turns a noisy single-trial
    coin flip into a stable estimate of the policy's survival probability.
    """
    sat0 = rng.normal(task.sat_mean, task.sat_stdev, episodes)
    choose_risky = np.fromiter(
        (decide_risky(net, s) for s in sat0), dtype=bool, count=episodes
    )
    mu = np.where(choose_risky, task.risky[0], task.safe[0])
    sigma = np.where(choose_risky, task.risky[1], task.safe[1])
    reward = rng.normal(mu, sigma)
    return float(np.mean(sat0 + reward >= task.threshold))


def make_eval_genomes(task: ForagingTask = ORIGINAL, episodes: int = 400, seed: int = 0):
    """Build a reproducible NEAT ``eval_genomes`` callback for ``task``.

    A single seeded generator is created once and reused across every genome and
    generation, so the whole run is deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = evaluate_policy(net, task, episodes, rng)

    return eval_genomes


def policy_curve(net, lo: float = 0.0, hi: float = 20.0, points: int = 101):
    """Probe the hypothesis: P(choose risky) as a function of satiation.

    Returns ``(satiation_grid, risky_choice)`` where ``risky_choice`` is 1.0
    where the network forages the risky patch. Under the energy-budget rule the
    curve should be 1 below the threshold and 0 above it.
    """
    grid = np.linspace(lo, hi, points)
    choice = np.array([1.0 if decide_risky(net, s) else 0.0 for s in grid])
    return grid, choice


# --------------------------------------------------------------------------- #
# Idealized curves: the survival-maximizing "optimal forager" the population   #
# should evolve toward, and the canonical prospect-theory value function.      #
# --------------------------------------------------------------------------- #

def option_survival(task: ForagingTask, satiation, risky: bool):
    """Analytic P(survive) at ``satiation`` if the given option is chosen.

    Survival means ``satiation + reward >= threshold`` with the reward drawn
    from the chosen patch's normal distribution, so this is just a normal CDF.
    """
    mu, sigma = task.risky if risky else task.safe
    satiation = np.asarray(satiation, dtype=float)
    return _norm_cdf((satiation + mu - task.threshold) / sigma)


def optimal_choice(task: ForagingTask, satiation):
    """Boolean mask: ``True`` where going risky maximizes survival probability."""
    return option_survival(task, satiation, True) > option_survival(task, satiation, False)


def optimal_value(task: ForagingTask, satiation):
    """Idealized value function: the best achievable survival probability.

    This is the curve an optimal forager (and, per the evolutionary-origin
    argument, prospect theory's value function) traces out: convex below the
    reference point -- accelerating returns, so risk-seeking pays -- and concave
    above it -- diminishing returns, so risk-aversion pays.
    """
    return np.maximum(
        option_survival(task, satiation, True),
        option_survival(task, satiation, False),
    )


def policy_value(net, task: ForagingTask, satiation):
    """Survival probability actually achieved by ``net``'s policy at each state."""
    satiation = np.asarray(satiation, dtype=float)
    choose_risky = np.array([decide_risky(net, float(s)) for s in satiation])
    return np.where(
        choose_risky,
        option_survival(task, satiation, True),
        option_survival(task, satiation, False),
    )


def mean_policy_value(nets, task: ForagingTask, satiation):
    """Mean survival probability across an ensemble of networks' policies.

    Averaging the top genomes reports the population's *typical* policy and
    smooths the idiosyncrasies of any single champion (e.g. a switch point that
    lands a little early), which show up as small notches in a lone winner's
    value curve.
    """
    return np.mean([policy_value(net, task, satiation) for net in nets], axis=0)


def mean_risky_fraction(nets, satiation):
    """Fraction of the ensemble that chooses the risky patch at each satiation.

    Unlike a single deterministic network (a hard 0/1 step), the ensemble mean
    is a smooth probability that transitions across the range of switch points
    the top genomes evolved.
    """
    satiation = np.atleast_1d(np.asarray(satiation, dtype=float))
    choices = [[decide_risky(net, float(s)) for s in satiation] for net in nets]
    return np.mean(np.asarray(choices, dtype=float), axis=0)


def reference_point(task: ForagingTask, lo: float = -30.0, hi: float = 50.0,
                    points: int = 8001) -> float:
    """Satiation at which the safe and risky patches are equally good.

    This is the prospect-theory *reference point*: below it the agent must beat
    the mean reward to survive (risk-seeking is optimal); above it a below-mean
    draw suffices (risk-aversion is optimal). For equal-mean patches it reduces
    to ``threshold - mean``.
    """
    grid = np.linspace(lo, hi, points)
    advantage = option_survival(task, grid, True) - option_survival(task, grid, False)
    crossings = np.where(np.diff(np.sign(advantage)) < 0)[0]
    if crossings.size == 0:
        return float("nan")
    i = crossings[0]
    x0, x1, y0, y1 = grid[i], grid[i + 1], advantage[i], advantage[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def kt_value(x, alpha: float = 0.88, beta: float = 0.88, lam: float = 2.25):
    """Canonical Kahneman & Tversky (1992) value function, reference at 0.

    Concave for gains (``x >= 0``), convex and steeper for losses (``x < 0``).
    Returned in its native (unbounded) units; rescale before overlaying on a
    survival-probability axis.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.power(np.abs(x), alpha), -lam * np.power(np.abs(x), beta))
