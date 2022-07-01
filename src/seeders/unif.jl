"""
    KMUnif()

The most basic `KMeansSeeder`: sample centroids uniformly at random from the
dataset, without replacement.
"""
struct KMUnif <: KMeansSeeder
end

function init_centroids(::KMUnif, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    DataLogging.@push_prefix! "INIT_UNIF"
    m, n = size(data)
    DataLogging.@log "INPUTS m: $m n: $n k: $k"
    t = @elapsed config = begin
        # NOTE: the sampling uses Fisher-Yates when n < 24k, which is O(n+k);
        #       or self-avoid-sample (keep a set, resample if collisions happen)
        #       otherwise, which is O(k)
        centroid_inds = sample(1:n, k; replace = false)
        centroids = zeros(m, k)
        for j = 1:k
            # i = rand(1:n)
            i = centroid_inds[j]
            centroids[:,j] .= @view data[:,i]
        end
        Configuration{A}(data, KMMatrix(centroids))
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

