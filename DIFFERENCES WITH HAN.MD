# Differences that still exists

1. I think the policy loss function is wrong. Han et al. 2019 uses policy prior where
   I use the L value that is calculated based on the current policy (Main actor net).
   See L491 in my script vs L161 in HAn.

2. Where is the beta from fromula 12?

3. Han uses log_alpha (L136) in the alpha_loss other implementations including
   [denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac) and
   [softlearning](https://github.com/rail-berkeley/softlearning.git) use alpha.

4. Han also uses log_lambda (L131) why isn't this lambda as it appears to be in the
   this is also more in relation with the alpha optimziation.

5. It is a littlebit confusing but COST_SAC uses (alpha3 + 1.0) (L132) where both your code
   and the LAC_V1 code of han does not.

# Other possible misunderstandings

1. Do I use the lyapunov actor in the right way on L564. Judging from eq 12 I think I do
   since it is Lc(s',fo(e,s)) and not L'c.
