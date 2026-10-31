# Online Causal Graph Falsification with CC-RuLSIF

CC-RuLSIF is an online change-detection method developed as part of my master's thesis in Computer Science at Northwestern University.

This repository contains a refactored adaptation of the original research implementation, conducted in 2025. This reorganizes the experimental code into a standalone Python package with clearer documentation, automated tests, and reproducible experiments.

Original research: [*Online Causal Graph Falsification with CC-RuLSIF* (NU-CS-2025-13, June 2025)](https://www.mccormick.northwestern.edu/computer-science/documents/stephen-savas-nu-cs-2025-13.pdf)

## Overview

In time-series data, existing causal discovery algorithms are capable of learning causal graphs from data, and newer methods have recently been extended to handle changes in causal relationships. However, these models are often impractical, as they make strong assumptions about the underlying data. To account for a changing causal graph, it is important to understand when an existing graph no longer holds, at which point users can retrain on recent data that reflects the updated ground truth.

My thesis proposed a novel method for detecting when a causal graph no longer accurately represents the data, using an adaptation of the Relative unconstrained Least Squares Importance Fitting (RuLSIF) method for estimating conditional probability density ratios, CC-RuLSIF (Conditional Causal RuLSIF). The proposed method leverages estimates the conditional density ratio between a reference dataset and a batch of new, online data points, and assessing their divergence.

## Methodology

CC-RuLSIF adapts Relative unconstrained Least Squares Importance Fitting (RuLSIF) to monitor a user-supplied directed acyclic graph (DAG) using streaming observations.

The method consists of four primary stages:

1. Graph decomposition: Extract each variable and its causal parent set from the supplied DAG.

2. Density-ratio estimation: Apply RuLSIF to parent-child observations, comparing historical reference data against recent observations.

3. Online monitoring: Maintain a sliding window of incoming samples and update divergence estimates.

4. Change detection: Compare estimated divergence against bootstrap-calibrated thresholds to identify statistically significant changes.

The method estimates changes in the joint distribution of a variable and its parents. When the parent distribution remains stable, the monitored density ratio corresponds to a conditional density ratio. Parent-distribution shifts can also produce alarms and therefore require careful interpretation.

Evaluation

The original thesis evaluated CC-RuLSIF on synthetic data featuring a relatively simple graph and was compared to other methods for causal graph falsification and change point detection. The results show that the proposed method achieves a strong balance of low detection delay and low false-positive rates, offering a useful alternative in settings where both accuracy and speed are critical

## Installation and Quickstart

TBD

## Code Structure

TBD

## Assumptions and Limitations

The implementation is designed for monitoring an existing causal graph supplied by the user, rather than discovering a new graph from scratch. 

With this, there are two key assumptions.
- Correct baseline DAG: The user–supplied graph is assumed causally sufficient (no unmeasured common causes) and structurally invariant during the reference window; only the conditional mechanisms may drift moving forward.
- Stationary reference window. The first nr points are assumed to come from a single, stable mechanism:

Limitations to this method include:
- Moderate parent dimensionality: Kernel methods suffer from the curse of dimensionality and degrade when |U| becomes too large (roughly > 10).
- No explicit parent-shift compensation. A change in a parent node can propagate in-direct divergences to its children. We can mitigate this by hierarchical interpretation and stricter child thresholds, but a principled correction is left as future work.
