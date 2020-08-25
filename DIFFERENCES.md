# Han et al. 2019/ SpinningUp comparison

## Possible differences

1. Spinning up doesn't have the ability to do finite horizons.
   - Shouldn't mather to much.
2. Han clipped the actions that come out of the policy. Spinning up only scales the actions, meaning in the oscillator environment it also gives negative values.

   - [x] This is now added.

3. The SquashedGaussian actor is slightly different in han 2019.

   The state tensor is used as an input to the log_sigma layer instead of the output of the
   fully connected layer.

   ```python
   log_sigma = tf.layers.dense(
      s, self.a_dim, None, trainable=trainable
   )  # QUESTION: Why s and not hidden layer output?
   ```

   vs

   ```python
   log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)
   ```

4. The Han LAC implementation contains the adaptive alpha method of SAC. Spinningup does not.

   - [x] I added this to the spinning up there is however one different. Han uses log_alpha where I think it should be alpha (L227). Doesn't mather to much i think because they are linearly dependent but log alpha is more stable and easier to compute for adam.

5. Han uses adaptive learning rate.

   - [x] I added this to the spinning up

6. Loss function of Han (L253) uses policy prior in the actor loss where spinning up uses the q-value for the current action that has the highest log likelihood (L362). The spinning up method is alos used in the code of [Haarnoja 2019](https://github.com/rail-berkeley/softlearning/blob/bc18386e49f8eba9455bf117e961984b3b472cf7/softlearning/algorithms/sac.py#L227). Han samples from normal distribution.

7. The actor loss is an array in Han et al. 2019 where in the spinning up code it is a scalar.
   - The difference comes from how the actor_loss is defined. Han et al. 2019 defines it as:

      ```python
         a_loss = (
            0 * self.labda * self.l_derta
            + self.alpha * tf.reduce_mean(log_pis)
            - policy_prior_log_probs
         )  # QUESTION: WHY POlicy prior
      ```

      Where I now define it to be:

      ```python
      a_loss = (log_alpha.exp() * logp_pi - q_pi).mean()
      ```

      Meaning I get the mean actor loss where han only takes the mean over the log
      likelihoods. I think this is related to the fact that he is sampling a prior from
      the normal distribution.

8. The spinning up never has a episode that is shorter than 400 steps. In han et al. 2019
   the first episodes are around 333.5 steps.