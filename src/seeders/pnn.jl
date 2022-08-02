"""
    PNN()

The seeder for the pairwise nearest-neighbor hierarchical clustering
method. Note that this scales somewhere between the square and the cube of the number
of points in the dataset.

See also: `kmeans`, `KMSeed`.
"""
struct PNN <: Seeder
end

function init_centroids(::PNN, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    m, n = size(data)

    centroids = copy(data)
    c = collect(1:n)
    costs = zeros(n)
    config0 = Configuration{Naive}(data, c, costs, centroids)
    config = pairwise_nn(config0, k, data, A)
    return config
end

