"""
    Unif()

The most basic seeder: sample centroids uniformly at random from the
dataset (without replacement).

See also: `kmeans`, `KMSeed`.
"""
struct Unif <: Seeder
end

function init_centroids(::Unif, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    m, n = size(data)
    config = @inbounds begin
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
    return config
end

