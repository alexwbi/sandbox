# sandbox
place to try stuff out


### KNN timing tests
Inspired by Jake VanderPlas. Github: @jakevdp
https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html

##### Implemented in python:
k = 3 in all cases

n = 10:
`550 µs ± 4.58 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)`

n = 100: `57.8 ms ± 161 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)`

n = 1000: `6.18 s ± 137 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)`


##### Implemented in NumPy:
n = 10: `34.8 µs ± 375 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)`

n = 100: `566 µs ± 58.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)`

n = 1000: `37.9 ms ± 1.53 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)`

n = 10000: `6.16 s ± 80.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)`

NumPy is consistently ~1 order of magnitude faster.