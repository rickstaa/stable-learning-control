# LAC VERSIONS

Below is a quick explanation of all the LAC version I tried:

-   lac: Original from Han et al.
-   lac2: Version in which the lyapunov constrained has been removed but the lyapunov critic is kept. It is similar to a
    SAC with a SINGLE lyapunov critic.
-   lac3: Version without Lyapunov constraint but with double-q trick.
-   lac4: Now let's add the constraint back to the lyapunov critic.
-   lac5: Now we add the entropy term of sac to the L_target. This is more in line with what SAC does.

# SAC VERSIONS

-   sac: Regular sac of Haarnoja et al.
-   sac2: Sac but now without the double Q trick.

## Things in which I was not successful

### SAC with 1 L-critic

I could not get the LAC algorithm to work when removing the lyapunov constraint from the actor loss function. Some questions I looked at:

-   Does the SAC algorithm work with one Critic?
    -   Altough the performance is worse it does seem to work!
