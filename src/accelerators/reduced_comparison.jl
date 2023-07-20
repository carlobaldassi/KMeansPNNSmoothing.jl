"""
   ReducedComparison

The method from Kaukoranta et al. (Proceedings of DCC 1999). Better than
[`Naive`](@ref), but often worse than alternative accelerators.

See also: [`kmeans`](@ref), [`KMAccel`](@ref).
"""
struct ReducedComparison <: Accelerator
    config::Configuration{ReducedComparison}
    active::BitVector
    stable::BitVector
    function ReducedComparison(config::Configuration{ReducedComparison})
        @extract config : centroids
        m, k = size(centroids)
        active = trues(k)
        stable = falses(k)
        return new(config, active, stable)
    end
    function Base.copy(accel::ReducedComparison; config::Union{Nothing,Configuration{ReducedComparison}} = nothing)
        @extract accel : active stable
        new_config::Configuration{ReducedComparison} = config ≡ nothing ? accel.config : config
        return new(new_config, copy(active), copy(stable))
    end
end

function reset!(accel::ReducedComparison)
    @extract accel : active stable
    fill!(active, true)
    fill!(stable, false)
    return accel
end

complete_initialization!(config::Configuration{ReducedComparison}, data::Mat64) = _complete_initialization_none!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{ReducedComparison}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: stable
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH[REDCOMP]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
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
            costs[i], c[i] = v, x
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    DataLogging.@exec dist_comp = n * k

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{ReducedComparison}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: active stable
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C[REDCOMP]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"
    DataLogging.@exec dist_comp = 0

    @assert all(c .> 0)

    active_inds = findall(active)

    num_fullsearch_th = zeros(Int, Threads.nthreads())
    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            wi = w ≡ nothing ? 1 : w[i]
            datai = @view data[:,i]
            old_v = costs[i]
            @views v = wi * _cost(datai, centroids[:,ci])
            DataLogging.@exec dist_comp += 1
            fullsearch = active[ci] && (v > old_v)
            num_fullsearch_th[Threads.threadid()] += fullsearch

            if fullsearch
                costsij = costsij_th[Threads.threadid()]
                _costs_1_vs_all!(costsij, data, i, centroids)
                DataLogging.@exec dist_comp += k
                x = ci
                for j in 1:k
                    j == ci && continue
                    v′ = wi * costsij[j]
                    if v′ < v
                        v, x = v′, j
                    end
                end
            else
                x = ci
                for j in active_inds
                    j == ci && continue
                    @views v′ = wi * _cost(datai, centroids[:,j])
                    if v′ < v
                        v, x = v′, j
                    end
                end
                DataLogging.@exec dist_comp += length(active_inds) - active[ci]
            end
            if x ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x] = false
                    stable[ci] = false
                    csizes[x] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i] = v, x
        end
    end
    num_fullsearch = sum(num_fullsearch_th)
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp fullsearches: $num_fullsearch / $n"
    DataLogging.@pop_prefix!
    return num_chgd
end

function centroids_from_partition!(config::Configuration{ReducedComparison}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract centroids: cmat=dmat
    @extract accel: active stable
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "C_FROM_P[REDCOMP]"

    new_centroids = Cache.new_centroids(m, k)
    zs = Cache.zs(k)

    _sum_clustered_data!(new_centroids, zs, data, c, stable, w)

    fill!(active, false)
    @inbounds for j = 1:k
        stable[j] && continue
        z = zs[j]
        z > 0 || continue
        new_centroids[:,j] ./= z
        active[j] = true
        for l = 1:m
            cmat[l,j] = new_centroids[l,j]
        end
    end
    update_quads!(centroids, stable)
    DataLogging.@pop_prefix!
    return config
end
