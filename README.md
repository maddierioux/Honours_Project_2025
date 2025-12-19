# Distributional Prospective Coding in Reinforcement Learning

## Overview

This repository contains the code used for my Honours thesis investigating how **distributional reinforcement learning**, combined with **prospective value coding**, can give rise to heterogeneous temporal response patterns resembling those observed in serotonergic neurons.

The project builds on the prospective coding framework introduced by **Harkin et al. (2025)**, which proposes that serotonin encodes a transformed representation of value incorporating both expected future reward and its temporal derivative. While the original framework treats value as a single scalar signal, this work extends it by introducing **distributional temporal-difference (TD) learning** along two complementary dimensions:
1. **Heterogeneous discounting timescales**, distributing value across temporal horizons.
2. **Expectile-based asymmetric TD learning**, distributing value across optimistic and pessimistic outcome sensitivities.

Learned value signals are subsequently transformed using a prospective coding mechanism to generate firing-rate–like activity, which is analyzed at both the single-channel and population levels.


## Repository Contents

### Original Code (This Project)

The following files were written specifically for this project and implement the core modeling and analysis pipeline:

- **`true_online_distributional_TD.py`**  
  Implements distributional temporal-difference learning using the true online TD(λ) algorithm, with support for:
  - expectile-weighted asymmetric prediction errors  
  - heterogeneous discounting timescales  
  - parallel learning across multiple distributional channels  

- **`Expectile_Asymmetry_TD.ipynb`**  
  Jupyter notebook exploring how expectile-based asymmetry shapes learning dynamics, convergence rates, and value magnitude across channels.

- **`Diff_learning_rates.ipynb`**  
  Jupyter notebook examining the effects of heterogeneous discounting timescales on value learning, temporal structure, and prospective-coded activity.

These files together generate the value trajectories, prospective-coded signals, and population-level heatmaps reported in the thesis.



### Conceptual and Methodological Basis

All models implemented here are conceptually grounded in the prospective coding framework, and all files except the ones talked about above are introduced in:

> **Harkin, E. F., et al. (2025).**  
> *A prospective code for value in serotonin.*  
> **Nature Neuroscience, 28**, 952–959.




