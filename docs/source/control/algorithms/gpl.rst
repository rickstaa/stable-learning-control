.. _gpl:

===
GPL
===

.. contents:: Table of Contents

Background
==========

Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Lyapunov-based Actor-Critic (LAC)}
        \label{alg1}
    \begin{algorithmic}[1]
        \REQUIRE Maximum episode length $N$ and maximum update steps $M$
        \REPEAT
            \STATE Samples $s_{0}$ according to $\rho$
            \FOR{$t=1$ to $N$}
                \STATE Sample $a$ from $\pi(a|s)$ and step forward
                \STATE Observe $s'$, $c$ and store ($s$, $a$, $c$, $s'$) in $\mathcal{D}$
            \ENDFOR
            \FOR{$i=1$ to $M$}
                \STATE Sample mini-batches of transitions from $D$ and update $L_{c}$, $\pi$, Lagrance multipliers with eq. (7) and (14) of Han et al., 2020
            \ENDFOR
        \UNTIL{eq. 11 of Han et al., 2020 is statisfied}
    \end{algorithmic}
    \end{algorithm}

.. _`11 of Han et al., 2020`: https://arxiv.org/pdf/2004.14288.pdf
.. _`eq. (7) and (14) from Han et al., 2020`: https://arxiv.org/pdf/2004.14288.pdf
