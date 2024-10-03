# KMeansPNNSmoothing.jl

This code implements the PNN-smoothing seeding method for k-means described in the paper
*"Systematically and efficiently improving k-means initialization
by pairwise-nearest-neighbor smoothing"* by C. Baldassi, TMLR, (2022) [TMLR][tmlr_paper], [arXiv][pnns_paper].

The code is written in [Julia]. It requires Julia 1.6 or later.

It provides a multi-threaded implementation of Lloyd's algorithm, also known as k-means,
with several seeding options:
* sample centroids uniformly at random from the dataset, without replacement
* [kmeans++][km++], including the "greedy" variant which is also used by
  [scikit-learn][sklearnkmeans]
* furthest-point heuristic, also called "maxmin" from [this paper][maxmin]
* [kmeans‖][scalable], also called "scalable kmeans++"
* [pairwise nearest-neighbor][PNN] hierarchical clustering (note: this scales more
  than quadratically with the number of points)
* the PNN-smoothing meta-method
* the [refine][refine] meta-method

It also implements a number of different techniques that can accelerate the iterations
of Lloyd's algorithm:
* the naive standard one (no acceleration)
* the "reduced comparison" method from [this paper][reduced_comparison]
* methods based on [Hamerly 2010][hamerly]
* methods based on [Elkan 2003][elkan]
* variants of the "yinyang" method from [this paper][yinyang]
* the "exponion" method from [this paper][exponion]
* the "ball" method from [this paper][ball]

The package also provides two functions to compute the centroid index as defined in [this paper][CI],
an asymmetric one called `CI` and a symmetric one called `CI_sym`. These are not exported.

It also provides a function to compute the variation of information metric to quantify the
distance between two partitions as defined in [this paper][VI]. The function is called `VI` and is
not exported.

### Installation and setup

To install the module, just clone it from GitHub into some directory. Then enter in such directory
and run julia with the "project" option:

```
$ julia --project
```

(Alternatively, if you start Julia from some other directory, you can press <kbd>;</kbd> to enter
in shell mode, `cd` into the project's directory, enter in pkg mode with <kbd>]</kbd> and use the
`activate` command.)

The first time you do this, you will then need to setup the project's environment. To do that,
when you're in the Julia REPL, press the <kbd>]</kbd> key to enter in pkg mode, then resolve the
dependencies:

```
(KMeansPNNSmoothing) pkg> resolve
```

This should download all the required packages. You can subsequently type `test` to check that
everything works. After this, you can press the backspace key to get back to the standard Julia
prompt, and load the package:

```
julia> using KMeansPNNSmoothing
```

### Usage

The format of the data must be a `Matrix{Float64}` with the data points organized by column.
(Typically, this means that if you're reading a dataset you'll need to transpose it. See for
example the `runfile.jl` script in the `test` directory.)

Here is an example run, assuming we want to cluster a `data` matrix into `k` clusters with
the original kmeans++ algorithm (the `{1}` type parameter deactivates the "greedy" version)
and using the "reduced yinyang" acceleration method:
```julia
result = kmeans(data, k; kmseeder=KMSeed.PlusPlus{1}(), accel=KMAccel.Ryy)
```
and here is an example running the PNN-smoothing scheme, using the non-greedy kmeans++ to
seed the initial sub-sets (this is actually the default if no keyword arguments are
passed):
```julia
result = kmeans(data, k; kmseeder=KMSeed.PNNS(KMSeed.PlusPlus{1}()))
```
and here is again PNN-smoothing but this time using "maxmin" at 2 levels of recursion:
```julia
result = kmeans(data, k; kmseeder=KMSeed.PNNS(KMSeed.MaxMin(), rlevel=2))
```

For the complete documentation you can use the Julia help (press the <kbd>?</kbd> key in
the REPL, then type `kmeans`, or `KMSeed` or `KMAccel`)

All codes are parallellized (in most cases over the data points) if there are threads
available: either run Julia with the `-t` option or use the `JULIA_NUM_THREADS` environment
variable.

## Licence

The code is released under the MIT licence.

The code has originated from the code of [RecombinatorKMeans.jl][reckmeans_repo], see the
licence information there.

[pnns_paper]: https://arxiv.org/abs/2202.03949
[tmlr_paper]: https://openreview.net/pdf?id=FTtFAg3pek
[Julia]: https://julialang.org
[km++]: https://scholar.google.com/scholar?cluster=16794944444927209316
[sklearnkmeans]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/_kmeans.py
[maxmin]: https://ieeexplore.ieee.org/document/329844
[scalable]: https://arxiv.org/abs/1203.6402
[PNN]: https://ieeexplore.ieee.org/document/35395
[refine]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.8528&rep=rep1&type=pdf
[CI]: https://www.sciencedirect.com/science/article/abs/pii/S0031320314001150
[VI]: https://www.sciencedirect.com/science/article/pii/S0047259X06002016?via%3Dihub
[reckmeans_repo]: https://github.com/carlobaldassi/RecombinatorKMeans.jl
[reduced_comparison]: https://doi.org/10.1109/TPAMI.2020.3008694
[hamerly]: https://doi.org/10.1137/1.9781611972801.12
[elkan]: https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
[yinyang]: https://proceedings.mlr.press/v37/ding15.html
[exponion]: https://proceedings.mlr.press/v48/newling16.html
[ball]: https://doi.org/10.1109/TPAMI.2020.3008694

