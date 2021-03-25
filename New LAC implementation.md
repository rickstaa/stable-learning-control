# HIGHER ORDER LAC IMPLEMENTATION

## Whats changing

1.  Value function instead of action value.
2.  We also want to have information about the previous states -> Change replay buffer
    1.  We need to find a way to include it in the actor loss.

### Higher order method

<https://en.wikipedia.org/wiki/Finite_difference>

Finally, use central difference.
