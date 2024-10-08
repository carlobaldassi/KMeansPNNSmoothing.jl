"""
   Ball

The "ball k-means" method by Xia et al. (TPAMI 2020).

See also: [`kmeans`](@ref), [`KMAccel`](@ref).
"""
struct Ball <: Accelerator
    config::Configuration{Ball}
    δc::Vector{Float64}
    r::Vector{Float64}
    cdist::Matrix{Float64}
    neighb::Vector{Vector{Int}}
    stable::Vector{Bool} # there's too much hopping around for BitVector
    nstable::Vector{Bool}
    function Ball(config::Configuration{Ball})
        # @extract config : centroids
        centroids = config.centroids.dmat # XXX
        m, k = size(centroids)
        δc = zeros(k)
        r = fill(Inf, k)
        cdist = [@inbounds @views √_cost(centroids[:,i], centroids[:,j]) for i = 1:k, j = 1:k] # TODO
        neighb = [deleteat!(collect(1:k), j) for j = 1:k]
        stable = fill(false, k)
        nstable = fill(false, k)
        return new(config, δc, r, cdist, neighb, stable, nstable)
    end
    function Base.copy(accel::Ball; config::Union{Nothing,Configuration{Ball}} = nothing)
        @extract accel : δc r cdist neighb stable nstable
        new_config::Configuration{Ball} = config ≡ nothing ? accel.config : config
        return new(new_config, copy(δc), copy(r), copy(cdist), copy.(neighb), copy(stable), copy(nstable))
    end
end

function reset!(accel::Ball)
    @extract accel : config δc r cdist neighb stable nstable
    @extract config : k
    centroids = config.centroids.dmat # XXX
    fill!(δc, 0.0)
    fill!(r, Inf)
    @inbounds for j = 1:k, i = 1:k
        cdist[i,j] = √_cost(centroids[:,i], centroids[:,j])
    end
    neighb .= [deleteat!(collect(1:k), j) for j = 1:k]
    fill!(stable, false)
    fill!(nstable, false)
    return accel
end

complete_initialization!(config::Configuration{Ball}, data::Mat64) = _complete_initialization_none!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{Ball}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ball accelerator method")

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    @bthreads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            costs[i], c[i] = findmin(costsij)
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Ball}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc r cdist neighb stable nstable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ball accelerator method")

    # @assert all(c .> 0)

    new_stable = fill!(similar(stable), true)
    sorted_neighb = copy.(neighb)
    did_sort = fill(false, k)
    num_chgd = 0
    lk = Threads.SpinLock()
    lk2 = ReentrantLock()
    @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            nstable[ci] && continue
            nci = sorted_neighb[ci]
            length(nci) == 0 && continue
            # @views v = _cost(data[:,i], centroids[:,ci])
            v = costs[i] # was set when computing r
            d = ✓(v)
            # @assert d ≤ r[ci]
            if !did_sort[ci]
                @lock lk2 begin
                    if !did_sort[ci] # maybe some thread did it while we were waiting...
                        sort!(nci, by=j′->cdist[j′,ci], alg=QuickSort)
                        did_sort[ci] = true
                    end
                end
            end
            # @assert cdist[first(nci), ci] == minimum(cdist[nci, ci])
            d ≤ cdist[nci[1], ci] / 2 && continue
            x = ci
            datai = @view data[:,i]
            nn = length(nci)
            for h = 1:nn
                j = nci[h]
                @views v′ = _cost(datai, centroids[:,j])
                if v′ < v
                    v, x = v′, j
                end
                h == nn && break
                d ≤ cdist[nci[h+1], ci] / 2 && break
            end
            if x ≠ ci
                @lock lk begin
                    num_chgd += 1
                    new_stable[ci] = false
                    new_stable[x] = false
                    csizes[ci] -= 1
                    csizes[x] += 1
                end
            end
            costs[i], c[i] = v, x
        end
    end
    copy!(stable, new_stable)
    cost = sum(costs)

    config.cost = cost
    return num_chgd
end

function centroids_from_partition!(config::Configuration{Ball}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc r cdist neighb stable nstable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ball accelerator method")

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

    r[.~stable] .= 0.0
    lk = Threads.SpinLock()
    @inbounds @bthreads for i = 1:n
        j = c[i]
        stable[j] && continue
        # r[j] = max(r[j], @views 2 * ✓(_cost(centroids[:,j], data[:,i])))
        v = @views _cost(centroids[:,j], data[:,i])
        costs[i] = v
        sv = 2 * ✓(v)
        if sv > r[j]
            @lock lk begin
                r[j] = sv
            end
        end
    end

    @inbounds for j = 1:k, j′ = 1:k
        cd = cdist[j′, j]
        δ, δ′ = δc[j], δc[j′]
        rj = r[j]
        if cd ≥ rj + δ + δ′
            cdist[j′, j] = cd - δ - δ′
        else
            @views cd = ✓(_cost(centroids[:,j′], centroids[:,j]))
            cdist[j′, j] = cd
        end
    end
    fill!(nstable, false)
    old_nj = Int[]
    sizehint!(old_nj, k-1)
    @inbounds for j = 1:k
        nj = neighb[j]
        resize!(old_nj, length(nj))
        copy!(old_nj, nj)
        resize!(nj, k-1)
        ind = 0
        allstable = stable[j]
        for j′ = 1:k
            j′ == j && continue
            if cdist[j′, j] < r[j]
                ind += 1
                nj[ind] = j′
                stable[j′] || (allstable = false)
            end
        end
        resize!(nj, ind)
        allstable && nj == old_nj && (nstable[j] = true)
    end

    return config
end
