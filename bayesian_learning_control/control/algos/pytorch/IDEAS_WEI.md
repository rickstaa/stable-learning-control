# Use previous state

```python
    a_previous, _ = self.ac.pi(o_previous)  # NOTE: Target actions come fr*current* policy
    a2, _ = self.ac.pi(o_)  # NOTE: Target actions come from *current* policy
    lya_l_ = self.ac.L(o_, a2)
    lya_l_previous = self.ac.L(o_previous, a_previous)
```
