include("Annuli.jl")
using .Annuli

"""
  Exponion

The exponion method as described in Newling and Fleuret (PMLR 2016).

Note: during kmeans iteration with the `verbose` option, the intermediate costs that
get printed when using this method are not accurate unless "[synched]" is printed too
(the output cost is always correct though).

See also: [`kmeans`](@ref), [`KMAccel`](@ref).
"""
struct Exponion <: Accelerator
    config::Configuration{Exponion}
    G::Int
    δc::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    cdist::Matrix{Float64}
    s::Vector{Float64}
    ann::Vector{SimplerAnnuli}
    stable::BitVector
    function Exponion(config::Configuration{Exponion})
        @extract config : n k
        G = ceil(Int, log2(k))
        δc = zeros(k)
        lb = zeros(n)
        ub = fill(Inf, n)
        # cdist = [@inbounds @views √_cost(centroids[:,j], centroids[:,j′]) for j = 1:k, j′ = 1:k]
        # s = [@inbounds minimum(j′ ≠ j ? cdist[j′,j] : Inf for j′ = 1:k) for j = 1:k]
        cdist = ones(k, k)
        for j = 1:k
           cdist[j,j] = 0.0
        end
        s = zeros(k)
        # ann = [@views SortedAnnuli(cdist[:,j], j) for j in 1:k]
        ann = [SimplerAnnuli(k-1, j) for j in 1:k]
        stable = falses(k)
        return new(config, G, δc, lb, ub, cdist, s, ann, stable)
    end
    function Base.copy(accel::Exponion; config::Union{Nothing,Configuration{Exponion}} = nothing)
        @extract accel : G δc lb ub cdist s ann stable
        new_config::Configuration{Exponion} = config ≡ nothing ? accel.config : config
        return new(new_config, G, copy(δc), copy(lb), copy(ub), copy(cdist), copy(s), copy(ann), copy(stable))
    end
end

function reset!(accel::Exponion)
    @extract accel : config δc lb ub cdist s ann stable
    @extract config : k
    centroids = config.centroids.dmat # XXX
    fill!(δc, 0.0)
    fill!(lb, 0.0)
    fill!(ub, Inf)
    @inbounds for j = 1:k
        cdistj = @view cdist[:,j]
        mincd = Inf
        for j′ = 1:k
            cdistj[j′] = @views √_cost(centroids[:,j], centroids[:,j′])
            j′ ≠ j && (mincd = min(mincd, cdistj[j′]))
        end
        s[j] = mincd # minimum(j′ ≠ j ? cdist[j′,j] : Inf for j′ = 1:k)
        update!(ann[j], cdistj)
    end
    fill!(stable, false)
    return accel
end

complete_initialization!(config::Configuration{Exponion}, data::Mat64) = _complete_initialization_ub!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{Exponion}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb ub stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly or Exponion accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH[EXP]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)
            costs[i], c[i] = v1, x1
            ub[i] = √̂(v1)
            lb[i] = √̂(v2)
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

function partition_from_centroids!(config::Configuration{Exponion}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub s ann stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Exponion accelerator method")

    DataLogging.@push_prefix! "P_FROM_C[EXP]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    DataLogging.@exec dist_comp = 0

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            lbi, ubi = lb[i], ub[i]
            hs = s[ci] / 2
            lbr = ifelse(lbi > hs, lbi, hs) # max(lbi, hs) # max accounts for NaN and signed zeros...
            lbr > ubi && continue
            datai = @view data[:,i]
            v = _cost(datai, @view centroids[:,ci])
            DataLogging.@exec dist_comp += 1
            costs[i] = v
            ub[i] = √̂(v)
            lbr > ub[i] && continue

            ri = 2 * (ubi + hs)
            anni = ann[ci]
            # f = min(searchsortedfirst(anni.es, ri), anni.G)
            # f = 1
            # for outer f in 1:anni.G
            #     anni.es[f] ≥ ri && break
            # end
            # js = anni.cws[f]
            js = get_inds(anni, ri)

            v1, v2, x1 = v, Inf, ci
            for j in js
                v′ = _cost(datai, @view centroids[:,j])
                if v′ < v1
                    v2 = v1
                    v1, x1 = v′, j
                elseif v′ < v2
                    v2 = v′
                end
            end
            DataLogging.@exec dist_comp += length(js)
            if x1 ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x1] = false
                    stable[ci] = false
                    csizes[x1] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i], = v1, x1
            ub[i], lb[i] = √̂(v1), √̂(v2)
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

sync_costs!(config::Configuration{Exponion}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = _sync_costs_ub!(config, data, w)

function centroids_from_partition!(config::Configuration{Exponion}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract accel: G δc lb ub cdist s ann stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Exponion accelerator method")

    DataLogging.@push_prefix! "C_FROM_P[EXP]"
    DataLogging.@exec dist_comp = 0

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
    DataLogging.@exec dist_comp += k - count(stable)

    δcₘ, δcₛ, jₘ = 0.0, 0.0, 0
    @inbounds for j = 1:k
        δcj = δc[j]
        if δcj > δcₘ
            δcₛ = δcₘ
            δcₘ, jₘ = δcj, j
        elseif δcj > δcₛ
            δcₛ = δcj
        end
    end
    @inbounds for i = 1:n
        ci = c[i]
        ub[i] += δc[ci]
        lb[i] -= ifelse(ci == jₘ, δcₛ, δcₘ)
    end

    @inbounds for j = 1:k
        stj = stable[j]
        cdj = @view cdist[:,j]
        s[j] = Inf
        for j′ = 1:k
            j′ == j && continue
            if stj && stable[j′]
                cd = cdj[j′]
            # elseif j′ < j
            #     cd = cdist[j, j′]
            #     cdist[j′,j] = cd
            else
                @views cd = √̂(_cost(centroids[:,j′], centroids[:,j]))
                DataLogging.@exec dist_comp += 1
                cdj[j′] = cd
            end
            if cd < s[j]
                s[j] = cd
            end
        end
        # update!(ann[j], cdj, stj ? stable : nothing) # only for SortedAnnuli
        update!(ann[j], cdj)
    end

    DataLogging.@log "dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end
