## NEAT Prospect Theory

The purpose of this repo is to empirically validate the Risk Sensitive Optimal Foraging Theory presented by Rose Mcdermott et al in `On the Evolutionary Origin of Prospect Theory Preferences`. The experiment performed in the notebook uses the Neuroevolution of Augmenting Topologies (NEAT) algorithm to simulate the evolution of a population that is evaluated on the foraging decision task formulated by Mcdermott et al.

## Motivation and Potential Use Cases

Understanding the origins of the Psychological Value Function defined by Kahneman and Tservsky could have potential value in aligning human and AI in scenarious dealing with risk.

## Analysis

The first version of the experiment showed no evolution: average fitness stayed
pinned near the base survival rate (~0.66) and the "winning" genome was an
unevolved member of the initial population. The problem was the **evaluation
function**, not the hypothesis.

* **Fitness was a single coin flip.** Each genome was scored on one stochastic
  foraging decision, collapsed to a 0/1 survival flag. Selection therefore acted
  on one Bernoulli trial per genome — almost pure noise — so NEAT could not tell
  a good policy from a lucky one.
* **The task barely rewarded the right choice.** With the original parameters the
  survival threshold sat right at the mean outcome and the environmental variance
  dominated, leaving under a two-point survival gap between the safe and risky
  patches (always-safe ≈ 65.3%, always-risky ≈ 66.4%). Even a perfect estimate of
  survival carried almost no selection pressure.

**The fix** (see `neat_prospect_theory/foraging.py` and `tests/test_foraging.py`):
score each genome by the *fraction* of many independent episodes it survives, and
use an equal-mean / variance-split task whose optimal choice actually flips with
the agent's state. With those changes the population evolves (average survival
climbs from ~0.75 toward ~0.79) and the evolved policy closely matches the
*idealized optimal forager*. Averaging the top 5 genomes into an ensemble — to
show the typical evolved policy rather than one champion's idiosyncrasies — the
ensemble's value function stays within ~0.001 survival-points of optimal across
the whole range.

The notebook plots the evolved ensemble against the idealized curve. Two
features emerge:

* **Reference dependence.** The agent is risk-seeking below a reference point and
  risk-averse above it. With equal-mean patches that reference point is where the
  reward needed to survive equals the mean reward (`satiation = threshold − mean`,
  here `10 − 5 = 5`), *not* the survival threshold itself.
* **An S-shaped value function.** The value (survival probability) of a state is
  convex below the reference (risk-seeking / "losses") and concave above it
  (risk-averse / "gains") — exactly the shape of Kahneman & Tversky's
  prospect-theory value function, which the notebook overlays. This is the
  evolutionary-origin claim of McDermott et al.: the prospect-theory value
  function is what survival maximization looks like around a threshold.
