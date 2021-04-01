# LAC VERSIONS

Below is a quick explanation of all the LAC version I tried:

-   lac: Original from Han et al.
-   lac2: Version in which the lyapunov constrained has been removed but the lyapunov critic is kept. It is similar to a
    SAC with a lyapunov critic.
-   lac3: Now the lyapunov constrained is added to the critic.

# SAC VERSIONS

-   sac: Regular sac of Haarnoja et al.
-   sac2: Sac but now without the double Q trick.

## Things in which I was not successful

### SAC with 1 L-critic

I could not get the LAC algorithm to work when removing the lyapunov constraint from the actor loss function. Some questions I looked at:

-   Does the SAC algorithm work with one Critic?
    -   Altough the performance is worse it does seem to work!
