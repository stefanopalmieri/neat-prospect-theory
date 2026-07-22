"""Tests that pin down *why* the original experiment failed to evolve.

Each test documents one part of the diagnosis:

1. A single-trial fitness estimate is dominated by noise; averaging over many
   episodes collapses that variance -- this is the fix that lets NEAT select on
   policy rather than luck.
2. Under the ORIGINAL parameters the safe and risky patches give almost the
   same survival odds, so even a perfect estimate carries little selection
   pressure (the task itself is nearly degenerate).
3. Under the RISK_SENSITIVE parameters the state-dependent energy-budget policy
   strictly beats either fixed policy, so there is a real gradient to climb.
"""
import numpy as np

from neat_prospect_theory.foraging import (
    ORIGINAL,
    RISK_SENSITIVE,
    evaluate_policy,
    mean_policy_value,
    mean_risky_fraction,
    optimal_choice,
    optimal_value,
    option_survival,
    policy_value,
    reference_point,
)


class _ConstNet:
    """A stub network that always returns the same output (fixed policy)."""

    def __init__(self, value):
        self._value = value

    def activate(self, _inputs):
        return [self._value]


class _BudgetNet:
    """Energy-budget rule: go risky iff the *needed* reward exceeds the mean.

    With equal-mean patches the risk-switch happens where the reward required to
    survive (``threshold - satiation``) equals the mean reward -- i.e. at
    ``satiation = threshold - mean_reward`` -- not at the threshold itself.
    Below it you need an above-mean draw (favour high variance); above it you
    need only a below-mean draw (favour low variance).
    """

    def __init__(self, switch_point):
        self._switch_point = switch_point

    def activate(self, inputs):
        _bias, satiation = inputs
        return [1.0 if satiation < self._switch_point else 0.0]


ALWAYS_SAFE = _ConstNet(0.0)
ALWAYS_RISKY = _ConstNet(1.0)


def test_averaging_collapses_single_trial_noise():
    """Single-trial fitness is ~Bernoulli; averaging cuts its variance sharply.

    This is the core bug: selecting on one trial per genome is selecting on a
    coin flip.
    """
    singles = [evaluate_policy(ALWAYS_RISKY, ORIGINAL, 1, np.random.default_rng(s))
               for s in range(300)]
    manys = [evaluate_policy(ALWAYS_RISKY, ORIGINAL, 400, np.random.default_rng(s))
             for s in range(300)]
    assert np.var(singles) > 20 * np.var(manys)


def test_original_task_is_nearly_degenerate():
    """Safe vs. risky differ by < 5 survival points under ORIGINAL params."""
    safe = evaluate_policy(ALWAYS_SAFE, ORIGINAL, 50_000, np.random.default_rng(1))
    risky = evaluate_policy(ALWAYS_RISKY, ORIGINAL, 50_000, np.random.default_rng(2))
    assert abs(safe - risky) < 0.05


def test_budget_rule_wins_on_risk_sensitive_task():
    """On the redesigned task the state-dependent policy beats both fixed ones."""
    switch_point = RISK_SENSITIVE.threshold - RISK_SENSITIVE.safe[0]  # 10 - 5 = 5
    budget = _BudgetNet(switch_point)
    safe = evaluate_policy(ALWAYS_SAFE, RISK_SENSITIVE, 50_000, np.random.default_rng(1))
    risky = evaluate_policy(ALWAYS_RISKY, RISK_SENSITIVE, 50_000, np.random.default_rng(2))
    budget_survival = evaluate_policy(budget, RISK_SENSITIVE, 50_000, np.random.default_rng(3))
    assert budget_survival > safe + 0.02
    assert budget_survival > risky + 0.02


def test_evaluate_policy_is_deterministic_given_seed():
    """Same seed -> same fitness, so whole NEAT runs are reproducible."""
    a = evaluate_policy(ALWAYS_RISKY, RISK_SENSITIVE, 200, np.random.default_rng(7))
    b = evaluate_policy(ALWAYS_RISKY, RISK_SENSITIVE, 200, np.random.default_rng(7))
    assert a == b


def test_reference_point_matches_closed_form_for_equal_means():
    """For equal-mean patches the reference point is threshold - mean."""
    r = reference_point(RISK_SENSITIVE)
    assert abs(r - (RISK_SENSITIVE.threshold - RISK_SENSITIVE.safe[0])) < 0.05


def test_optimal_value_is_s_shaped_around_reference():
    """The idealized value function is convex below and concave above reference.

    Convex-in-losses / concave-in-gains is exactly the prospect-theory value
    function shape, and here it falls out of survival maximization.
    """
    task = RISK_SENSITIVE
    r = reference_point(task)
    below = np.linspace(r - 6, r - 0.5, 40)
    above = np.linspace(r + 0.5, r + 6, 40)
    assert np.all(np.diff(optimal_value(task, below), 2) > 0)   # convex (losses)
    assert np.all(np.diff(optimal_value(task, above), 2) < 0)   # concave (gains)


def test_optimal_value_dominates_each_fixed_option():
    task = RISK_SENSITIVE
    s = np.linspace(-5, 25, 60)
    v = optimal_value(task, s)
    assert np.all(v >= option_survival(task, s, True) - 1e-12)
    assert np.all(v >= option_survival(task, s, False) - 1e-12)


def test_optimal_choice_flips_once_from_risky_to_safe():
    task = RISK_SENSITIVE
    s = np.linspace(-10, 30, 400)
    choice = optimal_choice(task, s).astype(int)  # 1 risky, 0 safe
    # exactly one transition, and it goes risky (1) -> safe (0)
    transitions = np.abs(np.diff(choice))
    assert transitions.sum() == 1
    assert choice[0] == 1 and choice[-1] == 0


def test_policy_value_of_optimal_policy_equals_optimal_value():
    """A network that follows the optimal choice achieves the optimal value."""
    task = RISK_SENSITIVE
    r = reference_point(task)

    class _OptimalNet:
        def activate(self, inputs):
            _bias, satiation = inputs
            return [1.0 if satiation < r else 0.0]

    s = np.linspace(-5, 25, 60)
    assert np.allclose(policy_value(_OptimalNet(), task, s), optimal_value(task, s))


def test_mean_policy_value_of_single_net_equals_policy_value():
    task = RISK_SENSITIVE
    s = np.linspace(-5, 25, 40)
    assert np.allclose(mean_policy_value([ALWAYS_RISKY], task, s),
                       policy_value(ALWAYS_RISKY, task, s))


def test_mean_risky_fraction_averages_across_ensemble():
    """A safe+risky pair chooses risky half the time everywhere; the mean of two
    early/late switchers ramps smoothly between their switch points."""
    s = np.linspace(-5, 25, 40)
    assert np.allclose(mean_risky_fraction([ALWAYS_SAFE, ALWAYS_RISKY], s), 0.5)

    early, late = _ConstSwitch(3.0), _ConstSwitch(7.0)
    frac = mean_risky_fraction([early, late], s)
    assert np.all((frac == 0.0) | (frac == 0.5) | (frac == 1.0))
    assert np.all(frac[s < 3.0] == 1.0)      # both go risky below both switches
    assert np.all(frac[s > 7.0] == 0.0)      # both go safe above both switches
    assert np.any(frac == 0.5)               # exactly one switches in between


class _ConstSwitch:
    """Risky iff satiation is below a fixed switch point (a stub 'genome')."""

    def __init__(self, switch_point):
        self._switch_point = switch_point

    def activate(self, inputs):
        _bias, satiation = inputs
        return [1.0 if satiation < self._switch_point else 0.0]
