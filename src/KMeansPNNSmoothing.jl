# grouping method from Kaukoranta, Fränti, Nevlainen, "Reduced comparison for the exact GLA"

module KMeansPNNSmoothing

using Random
using LinearAlgebra
using Statistics
using StatsBase
using ExtractMacro

using DelimitedFiles

include("DataLogging.jl")
using .DataLogging

export kmeans,
       KMeansSeeder, KMMetaSeeder,
       KMUnif, KMMaxMin, KMScala, KMPlusPlus, KMPNN,
       KMPNNS, KMPNNSR, KMRefine, KMAFKMC2

include("KMMatrices.jl")
using .KMMatrices

## due to floating point approx, we may end up with tiny negative cost values
## we use this rectified square root for that
Θ(x) = ifelse(x > 0, x, 0.0)
√̂(x) = √Θ(x)

abstract type Accelerator end

mutable struct Configuration{A<:Accelerator}
    m::Int
    k::Int
    n::Int
    c::Vector{Int}
    cost::Float64
    costs::Vector{Float64}
    centroids::Mat64
    csizes::Vector{Int}
    accel::A

    function Configuration{A}(data::Mat64, c::Vector{Int}, costs::Vector{Float64}, centroids::Mat64) where {A<:Accelerator}
        m, n = size(data)
        @assert length(c) == n
        @assert length(costs) == n
        k = size(centroids, 2)
        @assert size(centroids, 1) == m
        @assert all(1 .≤ c .≤ k)

        cost = sum(costs)
        csizes = zeros(Int, k)
        config = new{A}(m, k, n, c, cost, costs, centroids, csizes)
        update_csizes!(config)
        config.accel = A(config)
        complete_initialization!(config, data)
        return config
    end
    function Configuration{A}(data::Mat64, centroids::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) where {A<:Accelerator}
        m, n = size(data)
        k = size(centroids, 2)
        @assert size(centroids, 1) == m

        c = zeros(Int, n)
        costs = fill(Inf, n)
        csizes = zeros(Int, k)
        config = new{A}(m, k, n, c, 0.0, costs, centroids, csizes)
        config.accel = A(config)
        partition_from_centroids_from_scratch!(config, data, w)
        return config
    end
    function copy(config::Configuration{A}) where {A<:Accelerator}
        @extract config : m k n c cost costs centroids csizes accel
        return new{A}(m, k, n, copy(c), cost, copy(costs), copy(centroids), copy(csizes), copy(accel))
    end
end


include("accelerators.jl")

function update_csizes!(config::Configuration)
    @extract config: n c csizes
    fill!(csizes, 0)
    @inbounds for i in 1:n
        csizes[c[i]] += 1
    end
end

complete_initialization!(config::Configuration{<:Union{Naive,ReducedComparison,KBall,SHam,RElk}}, data::Mat64) = config

function complete_initialization!(config::Configuration{<:Union{Hamerly,SElk,Exponion}}, data::Mat64)
    @extract config: n costs accel
    @extract accel: ub

    @inbounds @simd for i = 1:n
        ub[i] = √̂(costs[i])
    end

    return config
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


function partition_from_centroids_from_scratch!(config::Configuration{Naive}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    num_chgd_th = zeros(Int, Threads.nthreads())
    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
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
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{ReducedComparison}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: stable
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
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

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{KBall}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with KBall accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
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
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{<:Union{Hamerly,Exponion}}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb ub stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly or Exponion accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
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

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{SHam}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SHam accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)
            costs[i], c[i] = v1, x1
            lb[i] = √̂(v2)
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{SElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: ub lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v, x = findmin(costsij)
            costs[i], c[i] = v, x
            lb[:,i] .= .√̂(costsij)
            ub[i] = lb[x,i]
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{RElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb active stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with RElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            costs[i], c[i] = findmin(costsij)
            lb[:,i] .= .√̂(costsij)
        end
    end
    num_chgd = n
    fill!(active, true)
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{Yinyang}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: ub groups gind lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Yinyang accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
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

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids_from_scratch!(config::Configuration{Ryy}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: groups gind lb stable active gactive
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ryy accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH"
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

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end


partition_from_centroids!(config::Configuration{Naive}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) =
    partition_from_centroids_from_scratch!(config, data, w)

function partition_from_centroids!(config::Configuration{ReducedComparison}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: active stable
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    @assert all(c .> 0)

    active_inds = findall(active)

    num_fullsearch_th = zeros(Int, Threads.nthreads())
    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            wi = w ≡ nothing ? 1 : w[i]
            datai = @view data[:,i]
            old_v = costs[i]
            @views v = wi * _cost(datai, centroids[:,ci])
            fullsearch = active[ci] && (v > old_v)
            num_fullsearch_th[Threads.threadid()] += fullsearch

            if fullsearch
                costsij = costsij_th[Threads.threadid()]
                _costs_1_vs_all!(costsij, data, i, centroids)
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
    DataLogging.@log "DONE time: $t cost: $cost fullsearches: $num_fullsearch / $n"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{KBall}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc r cdist neighb stable nstable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with KBall accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    # @assert all(c .> 0)

    new_stable = fill!(similar(stable), true)
    sorted_neighb = copy.(neighb)
    did_sort = fill(false, k)
    num_chgd = 0
    lk = Threads.SpinLock()
    lk2 = ReentrantLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            nstable[ci] && continue
            nci = sorted_neighb[ci]
            length(nci) == 0 && continue
            # @views v = _cost(data[:,i], centroids[:,ci])
            v = costs[i] # was set when computing r
            d = √̂(v)
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
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Hamerly}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub s stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            lbi, ubi = lb[i], ub[i]
            hs = s[ci] / 2
            lbr = ifelse(lbi > hs, lbi, hs) # max(lbi, hs) # max accounts for NaN and signed zeros...
            lbr > ubi && continue
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            ub[i] = √̂(v)
            lbr > ub[i] && continue

            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)
            if x1 ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x1] = false
                    stable[ci] = false
                    csizes[x1] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i] = v1, x1
            ub[i], lb[i] = √̂(v1), √̂(v2)
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Exponion}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub s ann stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Exponion accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            lbi, ubi = lb[i], ub[i]
            hs = s[ci] / 2
            lbr = ifelse(lbi > hs, lbi, hs) # max(lbi, hs) # max accounts for NaN and signed zeros...
            lbr > ubi && continue
            datai = @view data[:,i]
            v = _cost(datai, @view centroids[:,ci])
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
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{SHam}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb s stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SHam accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    num_chgd = 0
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            lbi = lb[i]
            hs = s[ci] / 2
            lbr = ifelse(lbi > hs, lbi, hs) # max(lbi, hs) # max accounts for NaN and signed zeros...
            lbr > √̂(v) && continue

            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)

            if x1 ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x1] = false
                    stable[ci] = false
                    csizes[x1] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i] = v1, x1
            lb[i] = √̂(v2)
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{SElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            ubi = ub[i]
            lbi = @view lb[:,i]

            skip = true
            for j = 1:k
                lbi[j] ≤ ubi && j ≠ ci && (skip = false; break)
            end
            skip && continue

            datai = @view data[:,i]
            @views v = _cost(datai, centroids[:,ci])
            x = ci
            sv = √̂(v)
            ubi = sv
            lbi[ci] = sv

            for j in 1:k
                (lbi[j] > ubi || j == ci) && continue
                @views v′ = _cost(datai, centroids[:,j])
                sv′ = √̂(v′)
                lbi[j] = sv′
                if v′ < v
                    v, x = v′, j
                    ubi = sv′
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
            costs[i], c[i], ub[i] = v, x, ubi
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{RElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb active stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with RElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    num_fullsearch_th = zeros(Int, Threads.nthreads())

    active_inds = findall(active)
    all_inds = collect(1:k)

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            datai = @view data[:,i]
            if active[ci]
                old_v = costs[i]
                @views v = _cost(datai, centroids[:,ci])
                fullsearch = (v > old_v)
            else
                v = costs[i]
                fullsearch = false
            end
            lbi = @view lb[:,i]
            ubi = √̂(v)
            if active[ci]
                lbi[ci] = ubi
            end
            num_fullsearch_th[Threads.threadid()] += fullsearch

            x = ci
            inds = fullsearch ? all_inds : active_inds
            for j in inds
                (lbi[j] > ubi || j == ci) && continue
                @views v′ = _cost(datai, centroids[:,j])
                sv′ = √̂(v′)
                lbi[j] = sv′
                if v′ < v
                    v, x = v′, j
                    ubi = sv′
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
            costs[i], c[i] = v, x
        end
    end
    num_fullsearch = sum(num_fullsearch_th)
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost fullsearches: $num_fullsearch / $n"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Yinyang}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: G δc ub groups gind lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Yinyang accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

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
            sv = √̂(v)
            ubi = sv
            x = ci
            for (f,gr) in enumerate(groups)
                lbi[f] > ubi && continue
                costsij = costsij_th[Threads.threadid()]
                _costs_1_vs_range!(costsij, data, i, centroids, gr)
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
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Ryy}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: G δc groups gind lb stable active gactive
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Ryy accelerator method")

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

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
    DataLogging.@log "DONE time: $t cost: $cost"
    DataLogging.@pop_prefix!
    return num_chgd
end

sync_costs!(config::Configuration, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = config

function sync_costs!(config::Configuration{<:Union{Hamerly,Yinyang,Exponion}}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: ub
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly, Yinyang or Exponion accelerator method")

    DataLogging.@push_prefix! "SYNC"
    # t = @elapsed Threads.@threads for i in 1:n
    t = @elapsed for i in 1:n
        @inbounds begin
            ci = c[i]
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            ub[i] = √̂(v)
        end
    end
    config.cost = sum(costs)
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function sync_costs!(config::Configuration{SElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb ub
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "SYNC"
    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            ub[i] = √̂(v)
            lb[ci,i] = √̂(v)
        end
    end
    config.cost = sum(costs)
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end



let centroidsdict = Dict{NTuple{3,Int},Matrix{Float64}}(),
    centroidsthrdict = Dict{NTuple{3,Int},Vector{Matrix{Float64}}}(),
    zsdict = Dict{NTuple{2,Int},Vector{Float64}}(),
    zsthrdict = Dict{NTuple{2,Int},Vector{Vector{Float64}}}()

    function _sum_clustered_data!(new_centroids, data, c, stable)
        m, k = size(new_centroids)
        @assert size(data, 1) == m
        n = size(data, 2)
        new_centroids_thr = get!(centroidsthrdict, (Threads.threadid(),m,k)) do
            [zeros(Float64, m, k) for id in 1:Threads.nthreads()]
        end
        foreach(nc_thr->fill!(nc_thr, 0.0), new_centroids_thr)
        Threads.@threads for i = 1:n
            @inbounds begin
                j = c[i]
                stable[j] && continue
                id = Threads.threadid()
                ncj = @view new_centroids_thr[id][:,j]
                datai = @view data[:,i]
                @simd for l = 1:m
                    ncj[l] += datai[l]
                end
            end
        end
        fill!(new_centroids, 0.0)
        for nc_thr in new_centroids_thr
            new_centroids .+= nc_thr
        end
        return new_centroids
    end

    function _sum_clustered_data!(new_centroids, zs, data, c, stable, w)
        m, k = size(new_centroids)
        @assert size(data, 1) == m
        n = size(data, 2)
        @assert length(zs) == k

        new_centroids_thr = get!(centroidsthrdict, (Threads.threadid(),m,k)) do
            [zeros(Float64, m, k) for id in 1:Threads.nthreads()]
        end
        zs_thr = get!(zsthrdict, (Threads.threadid(),k)) do
            [zeros(Float64, k) for id in 1:Threads.nthreads()]
        end

        foreach(nc_thr->fill!(nc_thr, 0.0), new_centroids_thr)
        foreach(z->fill!(z, 0.0), zs_thr)
        Threads.@threads for i = 1:n
            @inbounds begin
                j = c[i]
                wi = w ≡ nothing ? 1 : w[i]
                stable ≢ nothing && stable[j] && continue
                id = Threads.threadid()
                ncj = @view new_centroids_thr[id][:,j]
                datai = @view data[:,i]
                @simd for l = 1:m
                    ncj[l] += wi * datai[l]
                end
                zs_thr[id][j] += wi
            end
        end
        fill!(new_centroids, 0.0)
        for nc_thr in new_centroids_thr
            new_centroids .+= nc_thr
        end
        fill!(zs, 0.0)
        for zz in zs_thr
            zs .+= zz
        end
    end

    function _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
        m, k = size(centroids)
        fill!(δc, 0.0)
        @inbounds for j = 1:k
            stable[j] && continue
            z = csizes[j]
            z > 0 || continue
            centrj = @view centroids[:,j]
            ncentrj = @view new_centroids[:,j]
            ncentrj ./= z
            δc[j] = √̂(_cost(centrj, ncentrj))
            centrj[:] = ncentrj
        end
        update_quads!(centroids)
    end

    global function centroids_from_partition!(config::Configuration{Naive}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes
        @extract centroids: cmat=dmat
        @assert size(data) == (m, n)

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end
        zs = get!(zsdict, (Threads.threadid(),k)) do
            zeros(Float64, k)
        end

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

    global function centroids_from_partition!(config::Configuration{ReducedComparison}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract centroids: cmat=dmat
        @extract accel: active stable
        @assert size(data) == (m, n)

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end
        zs = get!(zsdict, (Threads.threadid(),k)) do
            zeros(Float64, k)
        end

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
        return config
    end

    global function centroids_from_partition!(config::Configuration{KBall}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract accel: δc r cdist neighb stable nstable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with KBall accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

        r[.~stable] .= 0.0
        lk = Threads.SpinLock()
        @inbounds Threads.@threads for i = 1:n
            j = c[i]
            stable[j] && continue
            # r[j] = max(r[j], @views 2 * √̂(_cost(centroids[:,j], data[:,i])))
            v = @views _cost(centroids[:,j], data[:,i])
            costs[i] = v
            sv = 2 * √̂(v)
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
                @views cd = √̂(_cost(centroids[:,j′], centroids[:,j]))
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

    global function centroids_from_partition!(config::Configuration{Hamerly}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract accel: δc lb ub s stable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with Hamerly accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

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
            lb[i] -= ifelse(ci == jₘ, δcₛ, δcₘ)
            ub[i] += δc[c[i]]
        end

        @inbounds for j = 1:k
            s[j] = Inf
            for j′ = 1:k
                j′ == j && continue
                @views cd = √̂(_cost(centroids[:,j′], centroids[:,j]))
                if cd < s[j]
                    s[j] = cd
                end
            end
        end
        return config
    end

    global function centroids_from_partition!(config::Configuration{Exponion}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract accel: G δc lb ub cdist s ann stable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with Exponion accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

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
                    cdj[j′] = cd
                end
                if cd < s[j]
                    s[j] = cd
                end
            end
            # update!(ann[j], cdj, stj ? stable : nothing) # only for SortedAnnuli
            update!(ann[j], cdj)
        end

        return config
    end

    global function centroids_from_partition!(config::Configuration{SHam}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract accel: δc lb s stable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with SHam accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

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
            lb[i] -= ifelse(ci == jₘ, δcₛ, δcₘ)
        end

        @inbounds for j = 1:k
            s[j] = Inf
            for j′ = 1:k
                j′ == j && continue
                @views cd = √̂(_cost(centroids[:,j′], centroids[:,j]))
                if cd < s[j]
                    s[j] = cd
                end
            end
        end
        return config
    end

    global function centroids_from_partition!(config::Configuration{SElk}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract accel: δc lb ub stable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with SElk accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

        @inbounds for i = 1:n
            ci = c[i]
            lbi = @view lb[:,i]
            @simd for j in 1:k
                lbi[j] -= δc[j]
            end
        end

        @inbounds for i = 1:n
            ci = c[i]
            ub[i] += δc[ci]
        end

        return config
    end

    global function centroids_from_partition!(config::Configuration{RElk}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids csizes accel
        @extract accel: δc lb active stable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with RElk accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

        active .= .~stable

        @inbounds for i = 1:n
            lbi = @view lb[:,i]
            @simd for j in 1:k
                lbi[j] -= δc[j]
            end
        end

        return config
    end

    global function centroids_from_partition!(config::Configuration{Yinyang}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c centroids csizes accel
        @extract accel: G δc δcₘ δcₛ jₘ ub groups gind lb stable
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with Yinyang accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

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
        return config
    end

    global function centroids_from_partition!(config::Configuration{Ryy}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c centroids csizes accel
        @extract accel: G δc δcₘ δcₛ jₘ groups gind lb stable active gactive
        @assert size(data) == (m, n)

        w ≡ nothing || error("w unsupported with Ryy accelerator method")

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end

        _sum_clustered_data!(new_centroids, data, c, stable)

        _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

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
        return config
    end

    global function clear_cache!()
        empty!(centroidsdict)
        empty!(centroidsthrdict)
        empty!(zsdict)
        empty!(zsthrdict)
    end
end

function check_empty!(config::Configuration, data::Mat64)
    @extract config: m k n c costs centroids csizes accel
    nonempty = csizes .> 0
    num_nonempty = sum(nonempty)
    num_centroids = min(n, k)
    gap = num_centroids - num_nonempty
    gap == 0 && return false
    to_fill = findall(.~(nonempty))[1:gap]
    stable = trues(k)
    for j in to_fill
        local i::Int
        while true
            i = rand(1:n)
            csizes[c[i]] > 1 && break
        end
        ci = c[i]
        z = csizes[ci]
        datai = @view data[:,i]
        y = @view centroids[:,ci]
        centroids.dmat[:,ci] .= (z .* y - datai) ./ (z - 1)
        csizes[ci] -= 1
        config.cost -= costs[i]
        centroids.dmat[:,j] .= datai
        c[i] = j
        csizes[j] = 1
        costs[i] = 0.0
        stable[ci] = false
        stable[j] = false
    end
    update_quads!(centroids, stable)
    reset!(accel) # TODO improve?
    return true
end


"""
A `KMeansSeeder` object is used to specify the seeding algorithm.
Currently the following basic objects are defined:

* `KMUnif`: sample centroids uniformly at random from the dataset, without replacement
* `KMPlusPlus`: kmeans++ by Arthur and Vassilvitskii (2006)
* `KMMaxMin`: furthest-point heuristic, by Katsavounidis et al. (1994)
* `KMScala`: kmeans‖ also called "scalable kmeans++", by Bahmani et al. (2012)
* `KMPNN`: pairwise nearest-neighbor hierarchical clustering, by Equitz (1989)

There are also meta-methods, whose parent type is `KMMetaSeeder`, 

* `KMPNNS`: the PNN-smoothing meta-method
* `KMRefine`: the refine meta-method by Bradely and Fayyad (1998)

For each of these object, there is a corresponding implementation of
`init_centroids(::KMeansSeeder, data, k)`, which concretely performs the initialization.

The documentation of each method explains the arguments that can be passed to
control the initialization process.
"""
abstract type KMeansSeeder end

"""
    KMUnif()

The most basic `KMeansSeeder`: sample centroids uniformly at random from the
dataset, without replacement.
"""
struct KMUnif <: KMeansSeeder
end

"""
    KMPlusPlus()
    KMPlusPlus{NC}()

A `KMeansSeeder` that uses kmeans++. The parameter `NC` determines the number of
candidates for the "greedy" version of the algorithm. By default, it is `nothing`,
meaning that it will use the value `log(2+k)`. Otherwise it must be an integer.
The value `1` corresponds to the original (non-greedy) algorithm, which has a
specialized implementation.
"""
struct KMPlusPlus{NC} <: KMeansSeeder
end

KMPlusPlus() = KMPlusPlus{nothing}()


"""
    KMAFKMC2{M}()

A `KMeansSeeder` that uses Assumption-Free K-MC². The parameter `M` determines the
number of Monte Carlo steps. This algorithm is implemented in a way that is O(kND)
instead of O(k²M) because we still need to compute the partition by the end. So it
is provided only for testing; for practical purposes `KMPlusPlus` should be preferred.
"""
struct KMAFKMC2{M} <: KMeansSeeder
end


"""
    KMMaxMin()

A `KMeansSeeder` for the furthest-point heuristic, also called maxmin.
"""
struct KMMaxMin <: KMeansSeeder
end

"""
    KMScala(;rounds=5, ϕ=2.0)

A `KMeansSeeder` for "scalable kmeans++" or kmeans‖. The `rounds` option
determines the number of sampling rounds; the `ϕ` option determines the
oversampling factor, which is then computed as `ϕ * k`.
"""
struct KMScala <: KMeansSeeder
    rounds::Int
    ϕ::Float64
    KMScala(;rounds = 5, ϕ = 2.0) = new(rounds, ϕ)
end

"""
    KMPNN()

A `KMeansSeeder` for the pairwise nearest-neighbor hierarchical clustering
method. Note that this scales somewhere between the square and the cube of the number
of points in the dataset.
"""
struct KMPNN <: KMeansSeeder
end

"""
A `KMMetaSeeder{S0<:KMeansSeeder}` object is a sub-type of `KMeansSeeder` representing a meta-method,
using an object of type `S0` as an internal seeder.
"""
abstract type KMMetaSeeder{S0<:KMeansSeeder} <: KMeansSeeder end

struct _KMSelf <: KMeansSeeder
end

"""
    KMPNNS(init0=KMPlusPlus{1}(); ρ=0.5, rlevel=1)

A `KMMetaSeeder` to use the PNN-smoothing algorithm. The inner method `init0` can be any
`KMeansSeeder`. The argument `ρ` sets the number of sub-sets, using the formula ``⌈√(ρ N / k)⌉``
where ``N`` is the number of data points and ``k`` the number of clusters, but the result is
clamped between `1` and `N÷k`. The argument `rlevel` sets the recursion level.

See `KMPNNSR` for the fully-recursive version.
"""
struct KMPNNS{S<:KMeansSeeder} <: KMMetaSeeder{S}
    init0::S
    ρ::Float64
end
function KMPNNS(init0::S = KMPlusPlus{1}(); ρ = 0.5, rlevel::Int = 1) where {S <: KMeansSeeder}
    @assert rlevel ≥ 1
    kmseeder = init0
    for r = rlevel:-1:1
        kmseeder = KMPNNS{typeof(kmseeder)}(kmseeder, ρ)
    end
    return kmseeder
end

const KMPNNSR = KMPNNS{_KMSelf}

"""
    KMPNNSR(;ρ=0.5)

The fully-recursive version of the `KMPNNS` seeder. It keeps splitting the dataset until the
number of points is ``≤2k``, at which point it uses `KMPNN`. The `ρ` option is documented in `KMPNNS`.
"""
KMPNNSR(;ρ = 0.5) = KMPNNS{_KMSelf}(_KMSelf(), ρ)

"""
    KMRefine(init0=KMPlusPlus{1}(); J=10, rlevel=1)

A `KMMetaSeeder` to use the "refine" algorithm. The inner method `init0` can be any
`KMeansSeeder`. The argument `J` sets the number of sub-sets. The argument `rlevel` sets the
recursion level.
"""
struct KMRefine{S<:KMeansSeeder} <: KMMetaSeeder{S}
    init0::S
    J::Int
end
function KMRefine(init0::S = KMPlusPlus{1}(); J = 10, rlevel::Int = 1) where {S <: KMeansSeeder}
    @assert rlevel ≥ 1
    kmseeder = init0
    for r = rlevel:-1:1
        kmseeder = KMRefine{typeof(kmseeder)}(kmseeder, J)
    end
    return kmseeder
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

function compute_costs_one!(costs::Vector{Float64}, data::Mat64, x::AbstractVector{Float64}, w::Nothing = nothing)
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    Threads.@threads for i = 1:n
        @inbounds costs[i] = _cost(@view(data[:,i]), x)
    end
    return costs
end

function compute_costs_one!(costs::Vector{Float64}, data::Mat64, x::AbstractVector{Float64}, w::AbstractVector{<:Real})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m
    @assert length(w) == n

    Threads.@threads for i = 1:n
        @inbounds costs[i] = w[i] * _cost(@view(data[:,i]), x)
    end
    return costs
end
compute_costs_one(data::Mat64, args...) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, args...)

init_centroids(::KMPlusPlus{nothing}, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) =
    init_centroids(KMPlusPlus{floor(Int, 2 + log(k))}(), data, k, A; kw...)

function init_centroids(::KMPlusPlus{NC}, data::Mat64, k::Int, A::Type{<:Accelerator}; w = nothing) where NC
    DataLogging.@push_prefix! "INIT_PP"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    ncandidates::Int = NC

    DataLogging.@log "LOCAL_VARS n: $n ncandidates: $ncandidates"

    t = @elapsed @inbounds config = begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = @view data[:,y]
        centr[:,1] = datay

        # costs = compute_costs_one(data, datay, w)
        costs = zeros(n)
        _costs_1_vs_all!(costs, data, y, data, w)

        curr_cost = sum(costs)
        c = ones(Int, n)

        new_costs, new_c = similar(costs), similar(c)
        new_costs_best, new_c_best = similar(costs), similar(c)
        for j = 2:k
            for i in 1:n
                costs[i] = Θ(costs[i])
            end
            pw = Weights(w ≡ nothing ? costs : costs .* w)
            nonz = count(pw .≠ 0)
            candidates = sample(1:n, pw, min(ncandidates,n,nonz), replace = false)
            cost_best = Inf
            y_best = 0
            for y in candidates
                _costs_1_vs_all!(new_costs, data, y, data, w)
                # datay = @view data[:,y]
                # compute_costs_one!(new_costs, data, datay, w)
                cost = 0.0
                for i = 1:n
                    v = new_costs[i]
                    v′ = costs[i]
                    if v < v′
                        new_c[i] = j
                        cost += v
                    else
                        new_costs[i] = v′
                        new_c[i] = c[i]
                        cost += v′
                    end
                end
                if cost < cost_best
                    cost_best = cost
                    y_best = y
                    new_costs_best, new_costs = new_costs, new_costs_best
                    new_c_best, new_c = new_c, new_c_best
                end
            end
            @assert y_best ≠ 0 && cost_best < Inf
            centr[:,j] .= @view data[:,y_best]
            costs, new_costs_best = new_costs_best, costs
            c, new_c_best = new_c_best, c
        end
        # returning config
        Configuration{A}(data, c, costs, KMMatrix(centr))
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function init_centroids(::KMAFKMC2{L}, data::Mat64, k::Int, A::Type{<:Accelerator}; w = nothing) where L
    DataLogging.@push_prefix! "INIT_AFKMC2"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    @assert L isa Int

    DataLogging.@log "LOCAL_VARS n: $n L: $L"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = zeros(n)
        _costs_1_vs_all!(costs, data, y, data, w)
        q = costs ./ 2 .+ (1 / 2n)
        pw = Weights(w ≡ nothing ? q : q .* w)

        curr_cost = sum(costs)
        c = ones(Int, n)

        new_costs = similar(costs)
        for j = 2:k
            y = sample(pw)
            v = costs[y]
            cost_best = Inf
            for t = 2:L
                y′ = sample(pw)
                v′ = costs[y′]
                ρ = (v′ * q[y]) / (v * q[y′])
                if rand() < ρ
                    v, y = v′, y′
                end
            end
            _costs_1_vs_all!(new_costs, data, y, data, w)
            # NOTE: here we update all costs, which is O(n)
            # this in some way defies the purpose of the algorithm, which is aimed at
            # getting O(mk²). We could recompute the costs on the fly every time.
            # However, the costs between each point and each centroid would still need
            # to be computed at the end, so we might as well do it here...
            @inbounds for i = 1:n
                v = new_costs[i]
                if v < costs[i]
                    costs[i], c[i] = v, j
                end
            end
            centr[:,j] .= @view data[:,y]
        end
        # returning config
        Configuration{A}(data, c, costs, KMMatrix(centr))
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function update_costs_one!(costs::Vector{Float64}, c::Vector{Int}, j::Int, data::Mat64, x::AbstractVector{Float64}, w::Nothing = nothing)
    m, n = size(data)
    @assert length(costs) == n
    @assert length(c) == n
    @assert length(x) == m

    Threads.@threads for i = 1:n
        @inbounds begin
            old_v = costs[i]
            new_v = _cost(@view(data[:,i]), x)
            if new_v < old_v
                costs[i] = new_v
                c[i] = j
            end
        end
    end
    return costs
end

function update_costs_one!(costs::Vector{Float64}, c::Vector{Int}, j::Int, data::Mat64, x::AbstractVector{Float64}, w::AbstractVector{<:Real})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(c) == n
    @assert length(x) == m

    Threads.@threads for i = 1:n
        @inbounds begin
            old_v = costs[i]
            new_v = w[i] * _cost(@view(data[:,i]), x)
            if new_v < old_v
                costs[i] = new_v
                c[i] = j
            end
        end
    end
    return costs
end

function init_centroids(::KMPlusPlus{1}, data::Mat64, k::Int, A::Type{<:Accelerator}; w = nothing)
    DataLogging.@push_prefix! "INIT_PP"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    DataLogging.@log "LOCAL_VARS n: $n ncandidates: 1"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay, w)

        c = ones(Int, n)

        @inbounds for j = 2:k
            pw = Weights(w ≡ nothing ? costs : costs .* w)
            y = sample(1:n, pw)
            datay = @view data[:,y]

            update_costs_one!(costs, c, j, data, datay, w)

            centr[:,j] .= datay
        end
        # returning config
        Configuration{A}(data, c, costs, KMMatrix(centr))
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function init_centroids(::KMMaxMin, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    DataLogging.@push_prefix! "INIT_MAXMIN"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = rand(1:n)
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)

        c = ones(Int, n)

        for j = 2:k
            y = argmax(costs)
            datay = @view data[:,y]

            update_costs_one!(costs, c, j, data, datay)

            centr[:,j] .= datay
        end
        # returning config
        Configuration{A}(data, c, costs, KMMatrix(centr))
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

Base.@propagate_inbounds function _merge_cost(centroids::AbstractMatrix{<:Float64}, z::Int, z′::Int, j::Int, j′::Int)
    return @views (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
end

function _get_nns(vs, j, k, centroids, csizes)
    k < 500 && return _get_nns(j, k, centroids, csizes)
    z = csizes[j]
    fill!(vs, (Inf, 0))
    Threads.@threads for j′ = 1:k
        j′ == j && continue
        @inbounds begin
            v1 = _merge_cost(centroids, z, csizes[j′], j, j′)
            id = Threads.threadid()
            if v1 < vs[id][1]
                vs[id] = v1, j′
            end
        end
    end
    v, x = Inf, 0
    for id in 1:Threads.nthreads()
        if vs[id][1] < v
            v, x = vs[id]
        end
    end
    return v, x
end

function _get_nns(j, k, centroids, csizes)
    z = csizes[j]
    v, x = Inf, 0
    @inbounds for j′ = 1:k
        j′ == j && continue
        z′ = csizes[j′]
        v1 = _merge_cost(centroids, z, csizes[j′], j, j′)
        if v1 < v
            v, x = v1, j′
        end
    end
    return v, x
end


function pairwise_nn(config::Configuration, tgt_k::Int, data::Mat64, ::Type{A}) where {A<:Accelerator}
    @extract config : m k n centroids csizes
    @extract centroids : cmat=dmat
    DataLogging.@push_prefix! "PNN"
    DataLogging.@log "INPUTS k: $k tgt_k: $tgt_k"
    if k < tgt_k
        get_logger() ≢ nothing && pop_logger!()
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    if k == tgt_k
        DataLogging.@pop_prefix!
        return Configuration{A}(data, centroids)
    end

    # csizes′ = zeros(Int, k)
    # for i = 1:n
    #     csizes′[c[i]] += 1
    # end
    # @assert all(csizes′ .> 0)
    # @assert csizes == csizes′
    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))
    t_costs = @elapsed @inbounds for j = 1:k
        nns_costs[j], nns[j] = _get_nns(vs, j, k, cmat, csizes)
    end

    t_fuse = @elapsed @inbounds while k > tgt_k
        jm = argmin(@view(nns_costs[1:k]))
        js = nns[jm]
        @assert nns_costs[js] == nns_costs[jm]
        @assert jm < js

        DataLogging.@push_prefix! "K=$k"
        DataLogging.@log "jm: $jm js: $js cost: $(nns_costs[jm])"

        # update centroid
        zm, zs = csizes[jm], csizes[js]
        for l = 1:m
            cmat[l,jm] = (zm * cmat[l,jm] + zs * cmat[l,js]) / (zm + zs)
            cmat[l,js] = cmat[l,k]
        end

        # update csizes
        csizes[jm] += zs
        csizes[js] = csizes[k]
        csizes[k] = 0

        # update partition
        # not needed since we don't use c in this computation and
        # we call partition_from_centroids! right after this function
        # for i = 1:n
        #     ci = c[i]
        #     if ci == js
        #         c[i] = jm
        #     elseif ci == k
        #         c[i] = js
        #     end
        # end

        # update nns
        nns[js] = nns[k]
        nns[k] = 0
        nns_costs[js] = nns_costs[k]
        nns_costs[k] = Inf

        num_fullupdates = 0
        for j = 1:(k-1)
            # 1) merged cluster jm, or clusters which used to point to either jm or js
            #    perform a full update
            if j == jm || nns[j] == jm || nns[j] == js
                num_fullupdates += 1
                nns_costs[j], nns[j] = _get_nns(vs, j, k-1, cmat, csizes)
            # 2) clusters that did not point to jm or js
            #    only compare the old cost with the cost for the updated cluster
            else
                z = csizes[j]
                # note: what used to point at k now must point at js
                v, x = nns_costs[j], (nns[j] ≠ k ? nns[j] : js)
                j′ = jm
                z′ = csizes[j′]
                v′ = _merge_cost(cmat, z, z′, j, j′)
                if v′ < v
                    v, x = v′, j′
                end
                nns_costs[j], nns[j] = v, x
                @assert nns[j] ≠ j
            end
        end
        DataLogging.@log "fullupdates: $num_fullupdates"
        DataLogging.@pop_prefix!

        k -= 1
    end

    @assert all(csizes[1:k] .> 0)

    mconfig = Configuration{A}(data, KMMatrix(cmat[:,1:k]))

    DataLogging.@log "DONE t_costs: $t_costs t_fuse: $t_fuse"
    DataLogging.@pop_prefix!
    return mconfig
end


function inner_init(S::KMPNNSR, data::Mat64, k::Int, A::Type{<:Accelerator})
    m, n = size(data)
    if n ≤ 2k
        return init_centroids(KMPNN(), data, k, A)
    else
        return init_centroids(S, data, k, A)
    end
end

inner_init(S::KMMetaSeeder{S0}, data::Mat64, k::Int, A::Type{<:Accelerator}) where S0 = init_centroids(S.init0, data, k, A)

function gen_fair_splits(n, J)
    b = n÷J
    r = n - b*J
    split = zeros(Int, n)
    for j in 1:J
        rng = (1:(b+(j≤r))) .+ ((b+1)*(j-1) - max(j-r-1,0))
        split[rng] .= j
    end
    shuffle!(split)
    return split
end

function gen_random_splits(n, J, k)
    split = rand(1:J, n)
    while any(sum(split .== a) < k for a = 1:J)
        split = rand(1:J, n)
    end
    shuffle!(split)
    return split
end

function gen_random_splits_quickndirty(n, J, k)
    return shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
end

function init_centroids(S::KMPNNS{S0}, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) where S0
    @extract S : ρ
    DataLogging.@push_prefix! "INIT_METANN"
    m, n = size(data)
    J = clamp(ceil(Int, √(ρ * n / k)), 1, n ÷ k)
    @assert J * k ≤ n
    # (J == 1 || J == n ÷ k) && @warn "edge case: J = $J"
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"

    t = @elapsed mconfig = begin
        tpp = @elapsed configs = begin
            # split = gen_random_splits(n, J, k)
            split = gen_fair_splits(n, J)
            # @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration{A}}(undef, J)
            Threads.@threads for a = 1:J
                rdata = KMMatrix(data.dmat[:,split .== a])
                DataLogging.@push_prefix! "SPLIT=$a"
                config = inner_init(S, rdata, k, A)
                lloyd!(config, rdata, 1_000, 0.0, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        DataLogging.@log "PPDONE time: $tpp"

        tnn = @elapsed mconfig = begin
            centroids_new = hcat((config.centroids.dmat for config in configs)...)
            c_new = zeros(Int, n)
            costs_new = zeros(n)
            inds = zeros(Int, J)
            for i = 1:n
                a = split[i]
                inds[a] += 1
                c_new[i] = configs[a].c[inds[a]] + k * (a-1)
                costs_new[i] = configs[a].costs[inds[a]]
            end
            jconfig = Configuration{Naive}(data, c_new, costs_new, KMMatrix(centroids_new))
            mconfig = pairwise_nn(jconfig, k, data, A)
            mconfig
        end
        DataLogging.@log "NNDONE time: $tnn"
        mconfig
    end
    DataLogging.@log "DONE time: $t cost: $(mconfig.cost)"
    DataLogging.@pop_prefix!
    return mconfig
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

function init_centroids(S::KMRefine{S0}, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) where S0
    @extract S : J
    DataLogging.@push_prefix! "INIT_REFINE"
    m, n = size(data)
    @assert J * k ≤ n
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"
    t = @elapsed mconfig = begin
        tinit = @elapsed configs = begin
            # split = gen_random_splits(n, J, k)
            split = gen_fair_splits(n, J)
            # @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration{A}}(undef, J)
            Threads.@threads for a = 1:J
                rdata = KMMatrix(data.dmat[:,split .== a])
                DataLogging.@push_prefix! "SPLIT=$a"
                config = inner_init(S, rdata, k, A)
                lloyd!(config, rdata, 1_000, 1e-4, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        DataLogging.@log "INITDONE time: $tinit"
        pool = KMMatrix(hcat((configs[a].centroids.dmat for a in 1:J)...))
        tref = @elapsed begin
            pconfigs = Vector{Configuration{A}}(undef, J)
            for a = 1:J
                config = Configuration{A}(pool, configs[a].centroids)
                DataLogging.@push_prefix! "SPLIT=$a"
                lloyd!(config, pool, 1_000, 1e-4, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        DataLogging.@log "REFINEDONE time: $tref"
        a_best = argmin([configs[a].cost for a in 1:J])
        Configuration{A}(data, configs[a_best].centroids)
    end
    DataLogging.@log "DONE time: $t cost: $(mconfig.cost)"
    DataLogging.@pop_prefix!
    return mconfig
end

function init_centroids(S::KMScala, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    @extract S : rounds ϕ
    DataLogging.@push_prefix! "INIT_SCALA"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    t = @elapsed config = begin
        centr = zeros(m, 1)
        y = rand(1:n)
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)

        cost = sum(costs)
        c = ones(Int, n)

        k′ = 1
        for r in 1:rounds
            w = (ϕ * k) / cost .* costs
            add_inds = findall(rand(n) .< w)
            add_k = length(add_inds)
            add_centr = data.dmat[:,add_inds]
            Threads.@threads for i in 1:n
                @inbounds begin
                    v, x = costs[i], c[i]
                    for j in 1:add_k
                        @views v′ = _cost(data[:,i], add_centr[:,j])
                        if v′ < v
                            v, x = v′, k′ + j
                        end
                    end
                    costs[i], c[i] = v, x
                end
            end
            cost = sum(costs)
            centr = hcat(centr, add_centr)
            k′ += add_k
        end
        @assert k′ ≥ k
        z = zeros(k′)
        for i = 1:n
            z[c[i]] += 1
        end
        # @assert all(z .> 0)
        centroids = KMMatrix(centr)
        cconfig = init_centroids(KMPlusPlus{1}(), centroids, k, ReducedComparison; w=z)
        lloyd!(cconfig, centroids, 1_000, 0.0, false, z)
        Configuration{A}(data, cconfig.centroids)
        # mconfig = Configuration{A}(m, k′, n, c, costs, centr)
        # pairwise_nn!(mconfig, k)
        # partition_from_centroids!(mconfig, data)
        # mconfig
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function lloyd!(
        config::Configuration,
        data::Mat64,
        max_it::Int,
        tol::Float64,
        verbose::Bool,
        w::Union{Vector{<:Real},Nothing} = nothing
    )
    DataLogging.@push_prefix! "LLOYD"
    DataLogging.@log "INPUTS max_it: $max_it tol: $tol"
    cost0 = config.cost
    converged = false
    it = 0
    t = @elapsed for outer it = 1:max_it
        centroids_from_partition!(config, data, w)
        old_cost = config.cost
        found_empty = check_empty!(config, data)
        num_chgd = partition_from_centroids!(config, data, w)
        new_cost = config.cost
        synched = false
        if tol > 0 && new_cost ≥ old_cost * (1 - tol) && !found_empty
            old_new_cost = new_cost
            sync_costs!(config, data, w)
            new_cost = config.cost
            # println(" > syncing $old_new_cost -> $new_cost")
            synched = (new_cost ≠ old_new_cost)
        end
        DataLogging.@log "it: $it cost: $(config.cost) num_chgd: $(num_chgd)$(found_empty ? " [found_empty]" : "")$(synched ? " [synched]" : "")"
        if num_chgd == 0 || (tol > 0 && new_cost ≥ old_cost * (1 - tol) && !found_empty && !synched)
            if !synched
                old_new_cost = new_cost
                sync_costs!(config, data, w)
                new_cost = config.cost
                synched = (new_cost ≠ old_new_cost)
            end
            verbose && println("converged cost = $new_cost")
            converged = true
            break
        end
        if it == max_it && !synched
            old_new_cost = new_cost
            sync_costs!(config, data, w)
            new_cost = config.cost
            synched = (new_cost ≠ old_new_cost)
        end
        verbose && println("lloyd it = $it cost = $new_cost num_chgd = $num_chgd" * (synched ? " [synched]" : ""))
    end
    DataLogging.@log "DONE time: $t iters: $it converged: $converged cost0: $cost0 cost1: $(config.cost)"
    DataLogging.@pop_prefix!
    return converged
end

struct Results
    exit_status::Symbol
    labels::Vector{Int}
    centroids::Matrix{Float64}
    cost::Float64
end

Results(exit_status, config::Configuration) = Results(exit_status, config.c, config.centroids, config.cost)

"""
  kmeans(data::Mat64, k::Integer; keywords...)

Runs k-means using Lloyd's algorithm on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column).

It returns an object of type `Results`, which contains the following fields:
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take two
  values, `:converged` or `:maxiters`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* cost: the final cost

The keyword arguments are:

* `max_it`: maximum number of iterations (default=1000). Normally the algorithm stops at fixed
  points.
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `kmseeder`: seeder for the initial configuration (default=`KMPNNS()`). It can be a `KMeansSeeder`
  or a `Matrix{Float64}`. If it's a matrix, it represents the initial centroids (by column).
  See the documentation for `KMeansSeeder` for a list of available seeding algorithms and their
  options.
* `tol`: a `Float64`, relative tolerance for detecting convergence (default=1e-5).
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.
"""
function kmeans(
        data::Matrix{Float64}, k::Integer;
        max_it::Integer = 1000,
        seed::Union{Integer,Nothing} = nothing,
        kmseeder::Union{KMeansSeeder,Matrix{Float64}} = KMPNNS(),
        verbose::Bool = true,
        tol::Float64 = 1e-5,
        accel::Type{<:Accelerator} = ReducedComparison,
        logfile::AbstractString = "",
    )

    logger = if !isempty(logfile)
        if DataLogging.logging_on
            init_logger(logfile, "w")
        else
            @warn "logging is off, ignoring logfile"
            nothing
        end
    end

    DataLogging.@push_prefix! "KMEANS"

    if seed ≢ nothing
        Random.seed!(seed)
        if VERSION < v"1.7-"
            Threads.@threads for h = 1:Threads.nthreads()
                Random.seed!(seed + h)
            end
        end
    end
    m, n = size(data)
    DataLogging.@log "INPUTS m: $m n: $n k: $k seed: $seed"

    mdata = KMMatrix(data)

    if kmseeder isa KMeansSeeder
        config = init_centroids(kmseeder, mdata, k, accel)
    else
        size(kmseeder) == (m, k) || throw(ArgumentError("Incompatible kmseeder and data dimensions, data=$((m,k)) kmseeder=$(size(kmseeder))"))
        centroids = KMMatrix(kmseeder)
        config = Configuration{accel}(mdata, centroids)
    end

    verbose && println("initial cost = $(config.cost)")
    DataLogging.@log "INIT_COST" config.cost
    converged = lloyd!(config, mdata, max_it, tol, verbose)
    DataLogging.@log "FINAL_COST" config.cost

    DataLogging.@pop_prefix!
    logger ≢ nothing && pop_logger!()

    exit_status = converged ? :converged : :maxiters

    clear_cache!()

    return Results(exit_status, config)
end

function gen_seeder(
        init::AbstractString = "pnns"
        ;
        init0::AbstractString = "",
        ρ::Float64 = 0.5,
        ncandidates::Union{Nothing,Int} = nothing,
        J::Int = 10,
        rlevel::Int = 1,
        rounds::Int = 5,
        ϕ::Float64 = 2.0,
    )
    all_basic_methods = ["++", "unif", "pnn", "maxmin", "scala"]
    all_rec_methods = ["refine", "pnns"]
    all_methods = [all_basic_methods; all_rec_methods]
    init ∈ all_methods || throw(ArgumentError("init should either be a matrix or one of: $all_methods"))
    if init ∈ all_rec_methods
        if init0 == ""
            init0="++"
            ncandidates ≡ nothing && (ncandidates = 1)
        end
        if init0 ∈ all_basic_methods
            init == "pnns" && rlevel < 1 && throw(ArgumentError("when init=$init and init0=$init0 rlevel must be ≥ 1"))
        elseif init0 == "self"
            init == "pnns" || throw(ArgumentError("init0=$init0 unsupported with init=$init"))
            rlevel = 0
        else
            throw(ArgumentError("when init=$init, init0 should be \"self\" or one of: $all_basic_methods"))
        end
    else
        init0 == "" || @warn("Ignoring init0=$init0 with init=$init")
    end

    if init ∈ all_basic_methods
        return init == "++"    ? KMPlusPlus{ncandidates}() :
               init == "unif"   ? KMUnif() :
               init == "pnn"    ? KMPNN() :
               init == "maxmin" ? KMMaxMin() :
               init == "scala"  ? KMScala(J, ϕ) :
               error("wat")
    elseif init == "pnns" && init0 == "self"
        @assert rlevel == 0
        return KMPNNSR(;ρ)
    else
        @assert rlevel ≥ 1
        kmseeder0 = init0 == "++"     ? KMPlusPlus{ncandidates}() :
                    init0 == "unif"   ? KMUnif() :
                    init0 == "pnn"    ? KMPNN() :
                    init0 == "maxmin" ? KMMaxMin() :
                    init0 == "scala"  ? KMScala(J, ϕ) :
                    error("wat")

        return init == "pnns"   ? KMPNNS(kmseeder0; ρ, rlevel) :
               init == "refine" ? KMRefine(kmseeder0; J, rlevel) :
               error("wut")
    end
end
# Centroid Index
# P. Fränti, M. Rezaei and Q. Zhao, Centroid index: cluster level similarity measure, Pattern Recognition, 2014
function CI(true_centroids::Matrix{Float64}, centroids::Matrix{Float64})
    m, tk = size(true_centroids)
    @assert size(centroids, 1) == m
    k = size(centroids, 2)

    matched = falses(tk)
    @inbounds for j = 1:k
        v = Inf
        p = 0
        for tj = 1:tk
            @views v1 = _cost(true_centroids[:,tj], centroids[:,j])
            if v1 < v
                v = v1
                p = tj
            end
        end
        matched[p] = true
    end

    return tk - count(matched)
end

# CI_sym(centroids1::Matrix{Float64}, centroids2::Matrix{Float64}) =
#     max(CI(centroids1, centroids2), CI(centroids2, centroids1))

function CI_sym(centroids1::Matrix{Float64}, centroids2::Matrix{Float64})
    m, k1 = size(centroids1)
    @assert size(centroids2, 1) == m
    k2 = size(centroids2, 2)

    a12 = zeros(Int, k1)
    a21 = zeros(Int, k2)
    v12 = fill(Inf, k1)
    v21 = fill(Inf, k2)
    @inbounds for j1 = 1:k1, j2 = 1:k2
        @views v = _cost(centroids1[:,j1], centroids2[:,j2])
        if v < v12[j1]
            v12[j1] = v
            a12[j1] = j2
        end
        if v < v21[j2]
            v21[j2] = v
            a21[j2] = j1
        end
    end

    return max(k1 - length(BitSet(a12)), k2 - length(BitSet(a21)))
end

# Variation of Information
# M. Meilă, Comparing clusterings—an information based distance, Journal of multivariate analysis, 2007

xlogx(x) = ifelse(iszero(x), zero(x), x * log(x))
entropy(p) = -sum(xlogx, p)

function VI(c1::Vector{Int}, c2::Vector{Int})
    n = length(c1)
    length(c2) == n || throw(ArgumentError("partitions bust have the same length, given: $(n) and $(length(c2))"))
    a, k1 = extrema(c1)
    a ≥ 1 || throw(ArgumentError("partitions elements must be ≥ 1, found $(a)"))
    a, k2 = extrema(c2)
    a ≥ 1 || throw(ArgumentError("partitions elements must be ≥ 1, found $(a)"))
    o = zeros(k1, k2)
    o1 = zeros(k1)
    o2 = zeros(k2)
    for i = 1:n
        j1, j2 = c1[i], c2[i]
        o[j1, j2] += 1
        o1[j1] += 1
        o2[j2] += 1
    end
    o ./= n
    o1 ./= n
    o2 ./= n
    vi = 2entropy(o) - entropy(o1) - entropy(o2)
    @assert vi > -1e-12
    return max(0.0, vi)
end

end # module
