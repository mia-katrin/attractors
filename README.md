# Stability and Geometry of Attractors in Neural Cellular Automata

Throughout the literature on Neural Cellular Automata
(NCAs), it is often taken for granted that the systems learn
attractors. This is shown through evolving the system for
many timesteps and noting visual similarity to the goal state.
There remain many questions after such an analysis. Namely,
what kind of attractors do we have? Is their behavior
ordered or chaotic? Can we estimate stability over very long
time horizons? What really happens in the attractor when
perturbations are applied? In this paper, we present a case
study to help answer these questions, with methods drawn
from the literature on dynamical systems theory.

We use the growing gecko NCA of Mordvintsev et al. (2020)
with deterministic cell updates as a case study. To the best
of the authors’ knowledge, we present the first visualizations
of NCA attractor dynamics. We also analyze them using
the Lyapunov and Fourier spectra, to reveal that the NCA
displays oscillatory, periodic and quasi-periodic behavior,
and that these behaviors arise early during training. This
challenges the belief that NCAs learn fixed point attractors.
Finally, we show that large perturbations to the attractor states
can throw the NCAs into a secondary mode separate from the
original attractor. We hope that this initial foray into NCA
attractor dynamics expands the toolkit for NCA researchers
to analyze the robustness and stability of their systems.

### Guide to the repository

All experiments were conducted in the jupyter notebooks included. Any additional code can be found in src. 

The models are uploaded in the "model" folder. These are the models used for our experiments. 
