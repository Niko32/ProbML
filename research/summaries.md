---
bibliography: [bibliography.bib]
---

# The origins of Kriging [@cressie1990origins]
Contains citations for original conceiver of kriging methods. Apparently, predicting a statistic between some known observations has been called kriging in geostatistics.

[Comprehensive article on Kriging](https://www.publichealth.columbia.edu/research/population-health-methods/kriging-interpolation#:~:text=Kriging%20can%20be%20understood%20as,blocks%20across%20the%20spatial%20field).

Note that geostatisticians often talk about creating "Variograms". What they are doing is estimating the kernel parameters and then calculating covariance matrices. In opposition to our view, they seem to see these as a function over distance values, not as a function over simply two points.

The kernels typically used seem to be the Exponential Kernel, linear kernel and spherical kernel.

They too simply vary those across problems.

Most kriging methods assume stationarity, aka that the mean of the data does not change across space.
Universal kriging suggests the addition of a mean model. [@armstrong1984problems] offers an explanation in
his paper discussing problems with universal kriging.

# Kernels
I found a comprehensive [page that visualizes various kernels](https://peterroelants.github.io/posts/gaussian-process-kernels/).
I suggest the Rational quadratic kernel which supposedly captures a mixture of Radial Basis
Functions and is seemingly quite good at offering a distance-related covariance function
that captures fine structures (depending on it's parameters.) Honestly, this choice
is very much made by gut feeling that this kernel would better be able to capture the fine grained influence
of a tertiary feature such as urban density.

