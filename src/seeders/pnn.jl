"""
    PNN(;accel=:auto)

The seeder for the pairwise nearest-neighbor hierarchical clustering
method. Note that this scales somewhere between the square and the cube of the number
of points in the dataset.
The `accel` keyword argument determines what accelerator mathod should be used. Available
methods are:
* `:FK` the method by Franti and Kaurokanta (1998)
* `:tri` a method that exploits the triangle inequality (unpublished)
Using `:auto` will select the first method for problems with fewer than 100 dimensions and
the second one otherwise.

See also: `kmeans`, `KMSeed`.
"""
struct PNN <: Seeder
    accel::Symbol
    function PNN(;accel::Symbol = :auto)
        allowed_accels = [:auto, :FK, :tri]
        accel ∈ allowed_accels || throw(ArgumentError("accel must be one of $allowed_accels"))
        return new(accel)
    end
end

function init_centroids(S::PNN, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    @extract S : accel
    m, n = size(data)

    accel == :auto && (accel = m < 100 ? :FK : :tri)
    pnn =
        accel == :FK ? pairwise_nn :
        accel == :tri ? pairwise_nn_savedist :
        error()

    centroids = copy(data)
    c = collect(1:n)
    costs = zeros(n)
    config0 = Configuration{Naive}(data, c, costs, centroids)
    config = pnn(config0, k, data, A)
    return config
end

