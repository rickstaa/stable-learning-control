# Differences that still exists

1. I think the policy loss function is wrong. Han et al. 2019 uses policy prior where
   I use the L value that is calculated based on the current policy (Main actor net).
   See L491 in my script vs L161 in HAn.

2. Han also uses log_lambda (L131) why i sn't this lambda as it appears to be in the
   this is also more in relation with the alpha optimziation.

# Other possible misunderstandings

1. Do I use the lyapunov actor in the right way on L564. Judging from eq 12 I think I do
   since it is Lc(s',fo(e,s)) and not L'c.
