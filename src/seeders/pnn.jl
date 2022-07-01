"""
    KMPNN()

A `KMeansSeeder` for the pairwise nearest-neighbor hierarchical clustering
method. Note that this scales somewhere between the square and the cube of the number
of points in the dataset.
"""
struct KMPNN <: KMeansSeeder
end

function init_centroids(::KMPNN, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    DataLogging.@push_prefix! "INIT_PNN"
    m, n = size(data)
    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    t = @elapsed config = begin
        centroids = copy(data)
        c = collect(1:n)
        costs = zeros(n)
        config0 = Configuration{Naive}(data, c, costs, centroids)
        config = pairwise_nn(config0, k, data, A)
        config
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

