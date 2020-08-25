# LAC network structure

## Comments

    - We need an actor for the next state because we want to use stabilization in the cost function.

## Cost functions

### Stabilization

Cost: Euclidean norm between s and s_/

### Tracking

Cost: Is the norm of the current state and a reference signal (state or even picture).
State:
    - s1: States which are used for tracking
    - s2: Contains the rest of the states

## Lyapunov functions

- Value function.
- Sum of cost over a finite time horizon

## Network

### Changes made to SAC

Soft-Critic -> Lyapunov critic --> We use a target network which follows using (EMA)
SquashedGaussianActor -> LyapunovSquashedGaussian actor + also normal actor.

RESULT:

- 1x SquashedGaussianActor (Trainable):
- 1x LyapunovSquashedGaussianActor (Trainable):
- 1x LyapunovCritic (Trainable):
- 1x LyapunovCritic (Not trainable):
= 4 networks

##### Additional found in han 2019
There is a twin actor that has its weights updated using ema. Why? Not needed in sac.
