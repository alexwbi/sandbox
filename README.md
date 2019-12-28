# sandbox
Random experiments


### KNN timing tests
- Code is in `k_nearest_neighbors_timing_experiments.py`
- k = 3 in all cases
- Inspired by Jake VanderPlas. Github: @jakevdp
    - https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html


##### Python implementation:
n = 10:
`550 µs ± 4.58 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)`

n = 100: `57.8 ms ± 161 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)`

n = 1000: `6.18 s ± 137 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)`


##### NumPy implementation:
n = 10: `23.4 µs ± 2.3 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)`

n = 100: `330 µs ± 22.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)`

n = 1000: `32.2 ms ± 1.47 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)`

n = 10000: `5.66 s ± 167 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)`

##### A few takeaways
NumPy is consistently ~1 order of magnitude faster.

Before using `np.where()` to replace zeros (zero distance between a point and itself), I was using list/filter inside
the final dict comprehension, which was contributing 10% of overall time (through %lprun). Switching to `np.where()`
improved performance by another ~10%.


### Logistic regression from scratch
Inspired by Nick Becker. Github: @beckernick
https://beckernick.github.io/logistic-regression-from-scratch/

##### Results / Takeaways
Using all default parameters, gradient descent reached convergence in 163 iterations.

- Iteration 0. Cross entropy loss: 0.5123195409784311
- Iteration 100. Cross entropy loss: 0.46516682845075313
- Iteration 163. Cross entropy loss: 0.4651664631221388

```
Confusion matrix - custom model:
[[5047 4953]
 [  76 9924]]
Accuracy - custom_model: 0.74855

Confusion matrix - scikit-learn:
[[5046 4954]
 [  76 9924]]
Accuracy - scikit-learn: 0.7485
```