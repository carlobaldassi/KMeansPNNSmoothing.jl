# grouping method from Kaukoranta, Fränti, Nevlainen, "Reduced comparison for the exact GLA"

module KMeansPNNSmoothing

using Random
using Statistics
using StatsBase
using ExtractMacro

include("DataLogging.jl")
using .DataLogging

export kmeans,
       KMeansSeeder, KMMetaSeeder,
       KMUnif, KMMaxMin, KMScala, KMPlusPlus, KMPNN,
       KMPNNS, KMPNNSR, KMRefine

mutable struct Configuration
    m::Int
    k::Int
    n::Int
    c::Vector{Int}
    cost::Float64
    costs::Vector{Float64}
    centroids::Matrix{Float64}
    active::BitVector
    nonempty::BitVector
    csizes::Vector{Int}
    function Configuration(m::Int, k::Int, n::Int, c::Vector{Int}, costs::Vector{Float64}, centroids::Matrix{Float64})
        @assert length(c) == n
        @assert length(costs) == n
        @assert size(centroids) == (m, k)
        cost = sum(costs)
        active = trues(k)
        nonempty = trues(k)
        csizes = zeros(Int, k)
        if !all(c .== 0)
            for i = 1:n
                csizes[c[i]] += 1
            end
            nonempty .= csizes .> 0
        end
        return new(m, k, n, c, cost, costs, centroids, active, nonempty, csizes)
    end
    function Base.copy(config::Configuration)
        @extract config : m k n c cost costs centroids active nonempty csizes
        return new(m, k, n, copy(c), cost, copy(costs), copy(centroids), copy(active), copy(nonempty), copy(csizes))
    end
end

function Configuration(data::Matrix{Float64}, centroids::Matrix{Float64}, w::Union{Vector{<:Real},Nothing} = nothing)
    m, n = size(data)
    k = size(centroids, 2)
    @assert size(centroids, 1) == m

    c = zeros(Int, n)
    costs = fill(Inf, n)
    config = Configuration(m, k, n, c, costs, centroids)
    partition_from_centroids!(config, data, w)
    return config
end

function remove_empty!(config::Configuration)
    @extract config: m k n c costs centroids active nonempty csizes
    DataLogging.@push_prefix! "RM_EMPTY"

    k_new = sum(nonempty)
    DataLogging.@log "k_new: $k_new k: $k"
    if k_new == k
        DataLogging.@pop_prefix!
        return config
    end
    centroids = centroids[:, nonempty]
    new_inds = cumsum(nonempty)
    for i = 1:n
        @assert nonempty[c[i]]
        c[i] = new_inds[c[i]]
    end
    csizes = csizes[nonempty]
    nonempty = trues(k_new)
    active = trues(k_new)

    config.k = k_new
    config.centroids = centroids
    config.csizes = csizes
    config.nonempty = nonempty
    config.active = active

    DataLogging.@pop_prefix!
    return config
end

Base.@propagate_inbounds function _cost(d1, d2)
    v1 = 0.0
    @simd for l = 1:length(d1)
        v1 += (d1[l] - d2[l])^2
    end
    return v1
end


function partition_from_centroids!(config::Configuration, data::Matrix{Float64}, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids active nonempty csizes
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    active_inds = findall(active)
    all_inds = collect(1:k)

    fill!(nonempty, false)
    fill!(csizes, 0)

    num_fullsearch_th = zeros(Int, Threads.nthreads())

    t = @elapsed Threads.@threads for i in 1:n
        @inbounds begin
            ci = c[i]
            wi = w ≡ nothing ? 1 : w[i]
            if ci > 0 && active[ci]
                old_v′ = costs[i]
                @views new_v′ = wi * _cost(data[:,i], centroids[:,ci])
                fullsearch = (new_v′ > old_v′)
            else
                fullsearch = (ci == 0)
            end
            num_fullsearch_th[Threads.threadid()] += fullsearch

            v, x, inds = fullsearch ? (Inf, 0, all_inds) : (costs[i], ci, active_inds)
            for j in inds
                @views v′ = wi * _cost(data[:,i], centroids[:,j])
                if v′ < v
                    v, x = v′, j
                end
            end
            costs[i], c[i] = v, x
        end
    end
    cost = sum(costs)
    for i in 1:n
        ci = c[i]
        csizes[ci] += 1
        nonempty[ci] = true
    end
    num_fullsearch = sum(num_fullsearch_th)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost fullsearches: $num_fullsearch / $n"
    DataLogging.@pop_prefix!
    return config
end

let centroidsdict = Dict{NTuple{3,Int},Matrix{Float64}}(),
    centroidsthrdict = Dict{NTuple{3,Int},Vector{Matrix{Float64}}}(),
    zsdict = Dict{NTuple{2,Int},Vector{Float64}}(),
    zsthrdict = Dict{NTuple{2,Int},Vector{Vector{Float64}}}()

    global function centroids_from_partition!(config::Configuration, data::Matrix{Float64}, w::Union{AbstractVector{<:Real},Nothing})
        @extract config: m k n c costs centroids active nonempty csizes
        @assert size(data) == (m, n)

        new_centroids = get!(centroidsdict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end
        new_centroids_thr = get!(centroidsthrdict, (Threads.threadid(),m,k)) do
            [zeros(Float64, m, k) for id in 1:Threads.nthreads()]
        end
        zs = get!(zsdict, (Threads.threadid(),k)) do
            zeros(Float64, k)
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
                id = Threads.threadid()
                nc = new_centroids_thr[id]
                for l = 1:m
                    nc[l,j] += wi * data[l,i]
                end
                zs_thr[id][j] += wi
            end
        end
        fill!(new_centroids, 0.0)
        for nc_thr in new_centroids_thr
            new_centroids .+= nc_thr
        end
        zs = zeros(k)
        for zz in zs_thr
            zs .+= zz
        end
        fill!(active, false)
        @inbounds for j = 1:k
            z = zs[j]
            z > 0 || continue
            @assert nonempty[j]
            for l = 1:m
                new_centroids[l,j] /= z
                new_centroids[l,j] ≠ centroids[l,j] && (active[j] = true)
                centroids[l,j] = new_centroids[l,j]
            end
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

function check_empty!(config::Configuration, data::Matrix{Float64})
    @extract config: m k n c costs centroids active nonempty csizes
    num_nonempty = sum(nonempty)
    num_centroids = min(config.n, config.k)
    gap = num_centroids - num_nonempty
    gap == 0 && return false
    to_fill = findall(.~(nonempty))[1:gap]
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
        centroids[:,ci] .= (z .* y - datai) ./ (z - 1)
        csizes[ci] -= 1
        config.cost -= costs[i]
        centroids[:,j] .= data[:,i]
        c[i] = j
        csizes[j] = 1
        nonempty[j] = true
        active[j] = true
        costs[i] = 0.0
    end
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





function init_centroids(::KMUnif, data::Matrix{Float64}, k::Int; kw...)
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
            centroids[:,j] .= data[:,i]
        end
        Configuration(data, centroids)
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function compute_costs_one!(costs::Vector{Float64}, data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64}, w::Nothing = nothing)
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    Threads.@threads for i = 1:n
        @inbounds @views costs[i] = _cost(data[:,i], x)
    end
    return costs
end

function compute_costs_one!(costs::Vector{Float64}, data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64}, w::AbstractVector{<:Real})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m
    @assert length(w) == n

    Threads.@threads for i = 1:n
        @inbounds @views costs[i] = w[i] * _cost(data[:,i], x)
    end
    return costs
end
compute_costs_one(data::AbstractMatrix{<:Float64}, args...) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, args...)

init_centroids(::KMPlusPlus{nothing}, data::Matrix{Float64}, k::Int; kw...) =
    init_centroids(KMPlusPlus{floor(Int, 2 + log(k))}(), data, k; kw...)

function init_centroids(::KMPlusPlus{NC}, data::Matrix{Float64}, k::Int; w = nothing) where NC
    DataLogging.@push_prefix! "INIT_PP"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    ncandidates::Int = NC

    DataLogging.@log "LOCAL_VARS n: $n ncandidates: $ncandidates"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay, w)

        curr_cost = sum(costs)
        c = ones(Int, n)

        new_costs, new_c = similar(costs), similar(c)
        new_costs_best, new_c_best = similar(costs), similar(c)
        for j = 2:k
            pw = Weights(w ≡ nothing ? costs : costs .* w)
            nonz = count(pw .≠ 0)
            candidates = sample(1:n, pw, min(ncandidates,n,nonz), replace = false)
            cost_best = Inf
            y_best = 0
            for y in candidates
                datay = data[:,y]
                compute_costs_one!(new_costs, data, datay, w)
                cost = 0.0
                @inbounds for i = 1:n
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
            datay = data[:,y_best]
            centr[:,j] .= datay
            costs, new_costs_best = new_costs_best, costs
            c, new_c_best = new_c_best, c
        end
        # returning config
        Configuration(m, k, n, c, costs, centr)
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end


function update_costs_one!(costs::Vector{Float64}, c::Vector{Int}, j::Int, data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64}, w::Nothing = nothing)
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

function update_costs_one!(costs::Vector{Float64}, c::Vector{Int}, j::Int, data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64}, w::AbstractVector{<:Real})
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

function init_centroids(::KMPlusPlus{1}, data::Matrix{Float64}, k::Int; w = nothing)
    DataLogging.@push_prefix! "INIT_PP"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    DataLogging.@log "LOCAL_VARS n: $n ncandidates: 1"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay, w)

        c = ones(Int, n)

        for j = 2:k
            pw = Weights(w ≡ nothing ? costs : costs .* w)
            y = sample(1:n, pw)
            datay = data[:,y]

            update_costs_one!(costs, c, j, data, datay, w)

            centr[:,j] .= datay
        end
        # returning config
        Configuration(m, k, n, c, costs, centr)
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function init_centroids(::KMMaxMin, data::Matrix{Float64}, k::Int; kw...)
    DataLogging.@push_prefix! "INIT_MAXMIN"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = rand(1:n)
        datay = data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)

        c = ones(Int, n)

        for j = 2:k
            y = argmax(costs)
            datay = data[:,y]

            update_costs_one!(costs, c, j, data, datay)

            centr[:,j] .= datay
        end
        # returning config
        Configuration(m, k, n, c, costs, centr)
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


function pairwise_nn!(config::Configuration, tgt_k::Int)
    @extract config : m k n centroids csizes
    DataLogging.@push_prefix! "PNN"
    DataLogging.@log "INPUTS k: $k tgt_k: $tgt_k"
    if k < tgt_k
        get_logger() ≢ nothing && pop_logger!()
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    if k == tgt_k
        DataLogging.@pop_prefix!
        return config
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
        nns_costs[j], nns[j] = _get_nns(vs, j, k, centroids, csizes)
    end

    t_fuse = @elapsed @inbounds while k > tgt_k
        jm = findmin(@view(nns_costs[1:k]))[2]
        js = nns[jm]
        @assert nns_costs[js] == nns_costs[jm]
        @assert jm < js

        DataLogging.@push_prefix! "K=$k"
        DataLogging.@log "jm: $jm js: $js cost: $(nns_costs[jm])"

        # update centroid
        zm, zs = csizes[jm], csizes[js]
        for l = 1:m
            centroids[l,jm] = (zm * centroids[l,jm] + zs * centroids[l,js]) / (zm + zs)
            centroids[l,js] = centroids[l,k]
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
                nns_costs[j], nns[j] = _get_nns(vs, j, k-1, centroids, csizes)
            # 2) clusters that did not point to jm or js
            #    only compare the old cost with the cost for the updated cluster
            else
                z = csizes[j]
                # note: what used to point at k now must point at js
                v, x = nns_costs[j], (nns[j] ≠ k ? nns[j] : js)
                j′ = jm
                z′ = csizes[j′]
                v′ = _merge_cost(centroids, z, z′, j, j′)
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

    config.k = k
    config.centroids = centroids[:,1:k]
    config.csizes = csizes[1:k]
    @assert all(config.csizes .> 0)
    config.active = trues(k)
    config.nonempty = trues(k)
    fill!(config.c, 0) # reset in order for partition_from_centroids! to work

    DataLogging.@log "DONE t_costs: $t_costs t_fuse: $t_fuse"
    DataLogging.@pop_prefix!
end


function inner_init(S::KMPNNSR, data::Matrix{Float64}, k::Int)
    m, n = size(data)
    if n ≤ 2k
        return init_centroids(KMPNN(), data, k)
    else
        return init_centroids(S, data, k)
    end
end

inner_init(S::KMMetaSeeder{S0}, data::Matrix{Float64}, k::Int) where S0 = init_centroids(S.init0, data, k)

function init_centroids(S::KMPNNS{S0}, data::Matrix{Float64}, k::Int; kw...) where S0
    @extract S : ρ
    DataLogging.@push_prefix! "INIT_METANN"
    m, n = size(data)
    J = clamp(ceil(Int, √(ρ * n / k)), 1, n ÷ k)
    @assert J * k ≤ n
    # (J == 1 || J == n ÷ k) && @warn "edge case: J = $J"
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"

    t = @elapsed mconfig = begin
        tpp = @elapsed configs = begin
            split = shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
            @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration}(undef, J)
            Threads.@threads for a = 1:J
                rdata = data[:,split .== a]
                DataLogging.@push_prefix! "SPLIT=$a"
                config = inner_init(S, rdata, k)
                lloyd!(config, rdata, 1_000, 1e-4, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        DataLogging.@log "PPDONE time: $tpp"

        tnn = @elapsed mconfig = begin
            centroids_new = hcat((config.centroids for config in configs)...)
            c_new = zeros(Int, n)
            costs_new = zeros(n)
            inds = zeros(Int, J)
            for i = 1:n
                a = split[i]
                inds[a] += 1
                c_new[i] = configs[a].c[inds[a]] + k * (a-1)
                costs_new[i] = configs[a].costs[inds[a]]
            end
            mconfig = Configuration(m, k * J, n, c_new, costs_new, centroids_new)
            pairwise_nn!(mconfig, k)
            partition_from_centroids!(mconfig, data)
            mconfig
        end
        DataLogging.@log "NNDONE time: $tnn"
        mconfig
    end
    DataLogging.@log "DONE time: $t cost: $(mconfig.cost)"
    DataLogging.@pop_prefix!
    return mconfig
end

function init_centroids(::KMPNN, data::Matrix{Float64}, k::Int; kw...)
    DataLogging.@push_prefix! "INIT_PNN"
    m, n = size(data)
    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    t = @elapsed config = begin
        centroids = copy(data)
        c = collect(1:n)
        costs = zeros(n)
        config = Configuration(m, n, n, c, costs, centroids)
        pairwise_nn!(config, k)
        partition_from_centroids!(config, data)
        config
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function init_centroids(S::KMRefine{S0}, data::Matrix{Float64}, k::Int; kw...) where S0
    @extract S : J
    DataLogging.@push_prefix! "INIT_REFINE"
    m, n = size(data)
    @assert J * k ≤ n
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"
    t = @elapsed mconfig = begin
        tinit = @elapsed configs = begin
            split = shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
            @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration}(undef, J)
            Threads.@threads for a = 1:J
                rdata = data[:,split .== a]
                DataLogging.@push_prefix! "SPLIT=$a"
                config = inner_init(S, rdata, k)
                lloyd!(config, rdata, 1_000, 1e-4, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        DataLogging.@log "INITDONE time: $tinit"
        pool = hcat((configs[a].centroids for a in 1:J)...)
        tref = @elapsed begin
            pconfigs = Vector{Configuration}(undef, J)
            for a = 1:J
                config = Configuration(pool, configs[a].centroids)
                DataLogging.@push_prefix! "SPLIT=$a"
                lloyd!(config, pool, 1_000, 1e-4, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        DataLogging.@log "REFINEDONE time: $tref"
        a_best = argmin([configs[a].cost for a in 1:J])
        Configuration(data, configs[a_best].centroids)
    end
    DataLogging.@log "DONE time: $t cost: $(mconfig.cost)"
    DataLogging.@pop_prefix!
    return mconfig
end

function init_centroids(S::KMScala, data::Matrix{Float64}, k::Int; kw...)
    @extract S : rounds ϕ
    DataLogging.@push_prefix! "INIT_SCALA"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    t = @elapsed config = begin
        centr = zeros(m, 1)
        y = rand(1:n)
        datay = data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)

        cost = sum(costs)
        c = ones(Int, n)

        k′ = 1
        for r in 1:rounds
            w = (ϕ * k) / cost .* costs
            add_inds = findall(rand(n) .< w)
            add_k = length(add_inds)
            add_centr = data[:,add_inds]
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
        cconfig = init_centroids(KMPlusPlus{1}(), centr, k; w=z)
        lloyd!(cconfig, centr, 1_000, 1e-4, false, z)
        Configuration(data, cconfig.centroids)
        # mconfig = Configuration(m, k′, n, c, costs, centr)
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
        data::Matrix{Float64},
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
        partition_from_centroids!(config, data, w)
        new_cost = config.cost
        DataLogging.@log "it: $it cost: $(config.cost)$(found_empty ? "[found_empty]" : ""))"
        if new_cost ≥ old_cost * (1 - tol) && !found_empty
            verbose && println("converged cost = $new_cost")
            converged = true
            break
        end
        verbose && println("lloyd it = $it cost = $new_cost")
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
  kmeans(data::Matrix{Float64}, k::Integer; keywords...)

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
        kmseeder::Union{KMeansSeeder,Matrix{Float64}} = KMPNNS{KMPlusPlus{1}},
        verbose::Bool = true,
        tol::Float64 = 1e-5,
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

    if kmseeder isa KMeansSeeder
        config = init_centroids(kmseeder, data, k)
    else
        centroids = kmseeder
        size(centroids) == (m, k) || throw(ArgumentError("Incompatible kmseeder and data dimensions, data=$((m,k)) kmseeder=$(size(centroids))"))
        config = Configuration(data, centroids)
    end

    verbose && println("initial cost = $(config.cost)")
    DataLogging.@log "INIT_COST" config.cost
    converged = lloyd!(config, data, max_it, tol, verbose)
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
