function gen_groups(k, G)
    b = k÷G
    r = k - b*G
    # gr = [(1:(b+(f≤r))) .+ (f≤r ? (b+1)*(f-1) : (b+1)*r + b*(f-r-1)) for f = 1:G]
    groups = [(1:(b+(f≤r))) .+ ((b+1)*(f-1) - max(f-r-1,0)) for f = 1:G]
    @assert vcat(groups...) == 1:k gr,vcat(groups...),1:k
    return groups
end

function cluster_centroids!(centroids::Mat64, G::Int)
    G == 1 && return [1:size(centroids,2)]
    ## we save/restore the RNG to make results comparable across accelerators
    ## (especially relevant with KMPNNS seeders)
    rng_bk = copy(Random.GLOBAL_RNG)
    result = kmeans(centroids.dmat, G; seed=rand(UInt64), kmseeder=PlusPlus{1}(), verbose=false, accel=ReducedComparison, _clear_cache=false)
    copy!(Random.GLOBAL_RNG, rng_bk)
    groups = UnitRange{Int}[]
    new_dmat = similar(centroids.dmat)
    new_dquads = similar(centroids.dquads)
    ind = 0
    for f = 1:G
        gr_inds = findall(result.labels .== f)
        gr_size = length(gr_inds)
        r = ind .+ (1:gr_size)
        new_dmat[:,r] = centroids.dmat[:,gr_inds]
        new_dquads[r] = centroids.dquads[gr_inds]
        push!(groups, r)
        ind += gr_size
    end
    new_centroids = KMMatrix(new_dmat, new_dquads)
    # @assert vcat(groups...) == 1:k
    copy!(centroids, new_centroids)
    return groups
end


"""
  Yinyang

The "simplified yinyang" method as described in Newling and Fleuret (PMLR 2016),
which is a simplified version of the method by Ding et al. (ICML 2015). Ofen outperformed
by [`Ryy`](@ref).

Note: during kmeans iteration with the `verbose` option, the intermediate costs that
get printed when using this method are not accurate unless "[synched]" is printed too
(the output cost is always correct though).

See also: [`kmeans`](@ref), [`KMAccel`](@ref), [`Ryy`](@ref).
"""
struct Yinyang <: Accelerator
    config::Configuration{Yinyang}
    G::Int
    δc::Vector{Float64}
    δcₘ::Vector{Float64}
    δcₛ::Vector{Float64}
    jₘ::Vector{Float64}
    ub::Vector{Float64}
    groups::Vector{UnitRange{Int}}
    gind::Vector{Int}
    lb::Matrix{Float64}
    stable::BitVector
    function Yinyang(config::Configuration)
        @extract config : n k centroids
        G = max(1, round(Int, k / 10))
        δc = zeros(k)
        δcₘ = zeros(G)
        δcₛ = zeros(G)
        jₘ = zeros(Int, G)
        ub = fill(Inf, n)
        gind = zeros(Int, n)
        lb = zeros(G, n)
        stable = falses(k)

        ## cluster the centroids
        # groups = gen_groups(k, G)
        groups = cluster_centroids!(centroids, G)
        return new(config, G, δc, δcₘ, δcₛ, jₘ, ub, groups, gind, lb, stable)
    end
    function Base.copy(accel::Yinyang)
        @extract accel : config δc δcₘ δcₛ jₘ ub groups gind lb stable
        return new(config, G, copy(δc), copy(δcₘ), copy(δcₛ), copy(jₘ), copy(ub), copy(groups), copy(gind), copy(lb), copy(stable))
    end

end

function reset!(accel::Yinyang)
    @extract accel : config δc δcₘ δcₛ jₘ ub groups gind lb stable
    @extract config : n c
    fill!(δc, 0.0)
    fill!(δcₘ, 0.0)
    fill!(δcₛ, 0.0)
    fill!(jₘ, 0)
    fill!(ub, Inf)
    @inbounds for i = 1:n
        ci = c[i]
        for (f,gr) in enumerate(groups)
            if last(gr) ≥ ci
                gind[i] = f
                break
            end
        end
    end
    fill!(lb, 0.0)
    fill!(stable, false)
    return accel
end

function complete_initialization!(config::Configuration{Yinyang}, data::Mat64)
    @extract config: n c costs accel
    @extract accel: ub groups gind

    @inbounds @simd for i = 1:n
        ub[i] = √̂(costs[i])
    end
    @inbounds for i = 1:n
        ci = c[i]
        for (f,gr) in enumerate(groups)
            if last(gr) ≥ ci
                gind[i] = f
                break
            end
        end
    end

    return config
end

function partition_from_centroids_from_scratch!(config::Configuration{Yinyang}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: ub groups gind lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Yinyang accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH[YIN]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v, x = findmin(costsij)
            costs[i], c[i] = v, x
            ub[i] = √̂(v)
            for (f,gr) in enumerate(groups)
                if last(gr) ≥ x
                    gind[i] = f
                    break
                end
            end
            lbi = @view lb[:,i]
            for (f,gr) in enumerate(groups)
                v′ = Inf
                for j in gr
                    j == x && continue
                    v′ = min(v′, costsij[j])
                end
                lbi[f] = √̂(v′)
            end
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

function partition_from_centroids!(config::Configuration{Yinyang}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: G δc ub groups gind lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Yinyang accelerator method")

    DataLogging.@push_prefix! "P_FROM_C[YIN]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    DataLogging.@exec dist_comp = 0

    # @assert all(1 .≤ gind .≤ G)
    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            # @assert 1 ≤ ci ≤ k
            # @assert 1 ≤ fi ≤ G
            ubi = ub[i]
            lbi = @view lb[:,i]
            skip = true
            for lbif in lbi
                lbif ≤ ubi && (skip = false; break)
            end
            skip && continue

            ci = c[i]
            fi = gind[i]

            @views v = _cost(data[:,i], centroids[:,ci])
            DataLogging.@exec dist_comp += 1
            sv = √̂(v)
            ubi = sv
            x = ci
            for (f,gr) in enumerate(groups)
                lbi[f] > ubi && continue
                costsij = costsij_th[Threads.threadid()]
                _costs_1_vs_range!(costsij, data, i, centroids, gr)
                DataLogging.@exec dist_comp += length(gr)
                v1, v2, x1 = Inf, Inf, 0
                for j in gr
                    j == x && continue
                    # @views v′ = _cost(data[:,i], centroids[:,j])
                    v′ = costsij[j]
                    if v′ < v1
                        v2 = v1
                        v1, x1 = v′, j
                    elseif v′ < v2
                        v2 = v′
                    end
                end
                if v1 < v
                    @assert x1 ≠ x
                    lbi[f] = √̂(v2)
                    if f ≠ fi
                        lbi[fi] = min(lbi[fi], ubi)
                        fi = f
                    else
                        lbi[f] = min(lbi[f], √̂(v))
                    end
                    v, x = v1, x1
                    ubi = √̂(v)
                else
                    lbi[f] = √̂(v1)
                end
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
            @assert 1 ≤ x ≤ k
            costs[i], c[i] = v, x
            gind[i] = fi
            ub[i] = ubi
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

sync_costs!(config::Configuration{Yinyang}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = _sync_costs_ub!(config, data, w)

function centroids_from_partition!(config::Configuration{Yinyang}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c centroids csizes accel
    @extract accel: G δc δcₘ δcₛ jₘ ub groups gind lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Yinyang accelerator method")

    DataLogging.@push_prefix! "C_FROM_P[YIN]"
    DataLogging.@exec dist_comp = 0

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
    DataLogging.@exec dist_comp += k - count(stable)

    fill!(δcₘ, 0.0)
    fill!(δcₛ, 0.0)
    fill!(jₘ, 0)
    for (f,gr) in enumerate(groups)
        @inbounds for j in gr
            δcj = δc[j]
            if δcj > δcₘ[f]
                δcₛ[f] = δcₘ[f]
                δcₘ[f], jₘ[f] = δcj, j
            elseif δcj > δcₛ[f]
                δcₛ[f] = δcj
            end
        end
    end
    @inbounds for i = 1:n
        ci = c[i]
        ub[i] += δc[ci]
    end
    @inbounds for i = 1:n
        lbi = @view lb[:,i]
        @simd for f = 1:G
            lbi[f] -= δcₘ[f]
        end
    end
    @inbounds for i = 1:n
        ci = c[i]
        fi = gind[i]
        ci == jₘ[fi] || continue
        lb[fi,i] -= (δcₛ[fi] - δcₘ[fi])
    end
    DataLogging.@log "dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end


"""
  Ryy

The "reduced-comparison simplified yinyang" method. This is the same as [`Yinyang`](@ref) except that it
doesn't use the upper bound, but it compensates by using the same technique as in [`ReducedComparison`](@ref),
often resulting in a better or equal performance.

See also: [`kmeans`](@ref), [`KMAccel`](@ref), [`Yinyang`](@ref), [`ReducedComparison`](@ref).
"""
struct Ryy <: Accelerator
    config::Configuration{Ryy}
    G::Int
    δc::Vector{Float64}
    δcₘ::Vector{Float64}
    δcₛ::Vector{Float64}
    jₘ::Vector{Float64}
    ub::Vector{Float64}
    groups::Vector{UnitRange{Int}}
    gind::Vector{Int}
    lb::Matrix{Float64}
    stable::BitVector
    active::BitVector
    gactive::BitVector
    function Ryy(config::Configuration)
        @extract config : n k centroids
        G = max(1, round(Int, k / 10))
        δc = zeros(k)
        δcₘ = zeros(G)
        δcₛ = zeros(G)
        jₘ = zeros(Int, G)
        ub = fill(Inf, n)
        gind = zeros(Int, n)
        lb = zeros(G, n)
        stable = falses(k)
        active = trues(k)
        gactive = trues(G)

        ## cluster the centroids
        # groups = gen_groups(k, G)
        groups = cluster_centroids!(centroids, G)
        return new(config, G, δc, δcₘ, δcₛ, jₘ, ub, groups, gind, lb, stable, active, gactive)
    end
    function Base.copy(accel::Ryy)
        @extract accel : config δc δcₘ δcₛ jₘ ub groups gind lb stable active gactive
        return new(config, G, copy(δc), copy(δcₘ), copy(δcₛ), copy(jₘ), copy(ub), copy(groups), copy(gind), copy(lb), copy(stable), copy(active), copy(gactive))
    end

end

function reset!(accel::Ryy)
    @extract accel : config δc δcₘ δcₛ jₘ ub groups gind lb stable active gactive
    @extract config : n c
    fill!(δc, 0.0)
    fill!(δcₘ, 0.0)
    fill!(δcₛ, 0.0)
    fill!(jₘ, 0)
    fill!(ub, Inf)
    @inbounds for i = 1:n
        ci = c[i]
        for (f,gr) in enumerate(groups)
            if last(gr) ≥ ci
                gind[i] = f
                break
            end
        end
    end
    fill!(lb, 0.0)
    fill!(stable, false)
    fill!(active, true)
    fill!(gactive, true)
    return accel
end

function complete_initialization!(config::Configuration{Ryy}, data::Mat64)
    @extract config: n c costs accel
    @extract accel: groups gind

    @inbounds for i = 1:n
        ci = c[i]
        for (f,gr) in enumerate(groups)
            if last(gr) ≥ ci
                gind[i] = f
                break
            end
        end
    end

    return config
end

function partition_from_centroids_from_scratch!(config::Configuration{Ryy}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: groups gind lb stable active gactive
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ryy accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH[RYY]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
        costsij = costsij_th[Threads.threadid()]
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v, x = findmin(costsij)
            costs[i], c[i] = v, x
            for (f,gr) in enumerate(groups)
                if last(gr) ≥ x
                    gind[i] = f
                    break
                end
            end
            lbi = @view lb[:,i]
            for (f,gr) in enumerate(groups)
                v′ = Inf
                for j in gr
                    j == x && continue
                    v′ = min(v′, costsij[j])
                end
                lbi[f] = √̂(v′)
            end
        end
    end
    num_chgd = n
    fill!(active, true)
    fill!(gactive, true)
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    DataLogging.@exec dist_comp = n * k

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Ryy}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: G δc groups gind lb stable active gactive
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ryy accelerator method")

    DataLogging.@push_prefix! "P_FROM_C[RYY]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    DataLogging.@exec dist_comp = 0

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    # @assert all(1 .≤ gind .≤ G)
    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            fi = gind[i]
            # @assert 1 ≤ ci ≤ k
            # @assert 1 ≤ fi ≤ G
            datai = @view data[:,i]
            if active[ci]
                old_v = costs[i]
                @views v = _cost(datai, centroids[:,ci])
                DataLogging.@exec dist_comp += 1
                costs[i] = v
                fullsearch = (v > old_v)
            else
                v = costs[i]
                fullsearch = false
            end
            sv = √̂(v)
            lbi = @view lb[:,i]
            skip = true
            for lbif in lbi
                lbif ≤ sv && (skip = false; break)
            end
            skip && continue

            old_fi = fi
            x = ci
            for (f,gr) in enumerate(groups)
                lbi[f] > sv && continue
                !fullsearch && !gactive[f] && continue
                costsij = costsij_th[Threads.threadid()]
                _costs_1_vs_range!(costsij, data, i, centroids, gr)
                DataLogging.@exec dist_comp += length(gr)
                v1, v2, x1 = Inf, Inf, 0
                for j in gr
                    j == x && continue
                    # @views v′ = _cost(datai, centroids[:,j])
                    v′ = costsij[j]
                    if v′ < v1
                        v2 = v1
                        v1, x1 = v′, j
                    elseif v′ < v2
                        v2 = v′
                    end
                end
                if v1 < v
                    @assert x1 ≠ x
                    lbi[f] = √̂(v2)
                    if f ≠ fi
                        lbi[fi] = min(lbi[fi], sv)
                        fi = f
                    else
                        lbi[f] = min(lbi[f], sv)
                    end
                    v, x = v1, x1
                    sv = √̂(v)
                else
                    lbi[f] = √̂(v1)
                end
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
            @assert 1 ≤ x ≤ k
            costs[i], c[i] = v, x
            gind[i] = fi
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

function centroids_from_partition!(config::Configuration{Ryy}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c centroids csizes accel
    @extract accel: G δc δcₘ δcₛ jₘ groups gind lb stable active gactive
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ryy accelerator method")

    DataLogging.@push_prefix! "C_FROM_P[KBALL]"
    DataLogging.@exec dist_comp = 0

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
    DataLogging.@exec dist_comp += k - count(stable)

    fill!(δcₘ, 0.0)
    fill!(δcₛ, 0.0)
    fill!(jₘ, 0)
    fill!(active, false)
    fill!(gactive, false)
    for (f,gr) in enumerate(groups)
        j₀ = first(gr) - 1
        @inbounds for j in gr
            δcj = δc[j]
            if δcj > δcₘ[f]
                δcₛ[f] = δcₘ[f]
                δcₘ[f], jₘ[f] = δcj, j
            elseif δcj > δcₛ[f]
                δcₛ[f] = δcj
            end
            if δcj > 0
                active[j] = true
                gactive[f] = true
            end
        end
    end
    @inbounds for i = 1:n
        lbi = @view lb[:,i]
        @simd for f = 1:G
            lbi[f] -= δcₘ[f]
        end
    end
    @inbounds for i = 1:n
        ci = c[i]
        fi = gind[i]
        ci == jₘ[fi] || continue
        lb[fi,i] -= (δcₛ[fi] - δcₘ[fi])
    end
    DataLogging.@log "dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end
