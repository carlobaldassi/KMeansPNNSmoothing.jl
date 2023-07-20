"""
   Naive

This is the standard alternated optimization method; at each step
everything is recomputed from scratch. It is inefficient and only
provided for reference.

See also: [`kmeans`](@ref), [`KMAccel`](@ref).
"""
struct Naive <: Accelerator
    config::Configuration{Naive}
    Naive(config::Configuration{Naive}) = new(config)
end

function Base.copy(accel::Naive; config::Union{Nothing,Configuration{Naive}} = nothing)
    return Naive(config ≡ nothing ? accel.config : config)
end
reset!(accel::Naive) = accel

complete_initialization!(config::Configuration{Naive}, data::Mat64) = _complete_initialization_none!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{Naive}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids
    @assert size(data) == (m, n)

    num_chgd_th = zeros(Int, Threads.nthreads())
    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            wi = w ≡ nothing ? 1 : w[i]
            v, x = Inf, 0
            for j in 1:k
                # @views v′ = wi * _cost(data[:,i], centroids[:,j])
                v′ = wi * costsij[j]
                if v′ < v
                    v, x = v′, j
                end
            end
            x ≠ ci && (num_chgd_th[Threads.threadid()] += 1)
            costs[i], c[i] = v, x
        end
    end
    num_chgd = sum(num_chgd_th)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    return num_chgd
end

partition_from_centroids!(config::Configuration{Naive}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) =
    partition_from_centroids_from_scratch!(config, data, w)


function centroids_from_partition!(config::Configuration{Naive}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes
    @extract centroids: cmat=dmat
    @assert size(data) == (m, n)

    new_centroids = Cache.new_centroids(m, k)
    zs = Cache.zs(k)

    _sum_clustered_data!(cmat, zs, data, c, nothing, w)

    @inbounds for j = 1:k
        z = zs[j]
        z > 0 || continue
        for l = 1:m
            cmat[l,j] /= z
        end
    end
    update_quads!(centroids)
    return config
end
