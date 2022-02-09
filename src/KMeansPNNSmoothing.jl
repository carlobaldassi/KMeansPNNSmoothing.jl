# grouping method from Kaukoranta, Fränti, Nevlainen, "Reduced comparison for the exact GLA"

module KMeansPNNSmoothing

using Random
using Statistics
using StatsBase
using ExtractMacro

export kmeans


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

    k_new = sum(nonempty)
    k_new == k && return config
    centroids = centroids[:, nonempty]
    new_inds = cumsum(nonempty)
    for i = 1:n
        @assert nonempty[c[i]]
        c[i] = new_inds[c[i]]
    end
    csizes = csizes[nonempty]
    nonempty = trues(k_new)

    config.k = k_new
    config.centroids = centroids
    config.csizes = csizes
    config.nonempty = nonempty

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

    active_inds = findall(active)
    all_inds = collect(1:k)

    fill!(nonempty, false)
    fill!(csizes, 0)

    num_fullsearch_th = zeros(Int, Threads.nthreads())

    Threads.@threads for i in 1:n
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
        i = rand(1:n)
        centroids[:,j] .= data[:,i]
        active[j] = true
    end
    return true
end

function init_centroid_unif(data::Matrix{Float64}, k::Int; kw...)
    m, n = size(data)
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
    return Configuration(data, centroids)
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

function init_centroid_pp(data::Matrix{Float64}, k::Int; ncandidates = nothing, w = nothing)
    ncandidates == 1 && return init_centroid_pp1(data, k; w)
    m, n = size(data)
    @assert n ≥ k

    ncandidates::Int = ncandidates ≡ nothing ? floor(Int, 2 + log(k)) : ncandidates

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
    return Configuration(m, k, n, c, costs, centr)
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

function init_centroid_pp1(data::Matrix{Float64}, k::Int; w = nothing)
    m, n = size(data)
    @assert n ≥ k

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
    return Configuration(m, k, n, c, costs, centr)
end

function init_centroid_maxmin(data::Matrix{Float64}, k::Int)
    m, n = size(data)
    @assert n ≥ k

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
    return Configuration(m, k, n, c, costs, centr)
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
    if k < tgt_k
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    k == tgt_k && return config

    # csizes′ = zeros(Int, k)
    # for i = 1:n
    #     csizes′[c[i]] += 1
    # end
    # @assert all(csizes′ .> 0)
    # @assert csizes == csizes′
    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))
    @inbounds for j = 1:k
        nns_costs[j], nns[j] = _get_nns(vs, j, k, centroids, csizes)
    end

    @inbounds while k > tgt_k
        jm = findmin(@view(nns_costs[1:k]))[2]
        js = nns[jm]
        @assert nns_costs[js] == nns_costs[jm]
        @assert jm < js

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

        k -= 1
    end

    config.k = k
    config.centroids = centroids[:,1:k]
    config.csizes = csizes[1:k]
    @assert all(config.csizes .> 0)
    config.active = trues(k)
    config.nonempty = trues(k)
    fill!(config.c, 0) # reset in order for partition_from_centroids! to work
    return
end

function recnninit(data::Matrix{Float64}, k::Int)
    m, n = size(data)
    if n ≤ 2k
        return init_centroid_nn(data, k)
    else
        return init_centroid_metann(data, k; init = recnninit)
    end
end

function init_centroid_metann(data::Matrix{Float64}, k::Int; init = init_centroid_pp1, ρ = 0.5)
    m, n = size(data)
    J = clamp(ceil(Int, √(ρ * n / k)), 1, n ÷ k)
    @assert J * k ≤ n
    # (J == 1 || J == n ÷ k) && @warn "edge case: J = $J"

    split = shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
    @assert all(sum(split .== a) ≥ k for a = 1:J)
    configs = Vector{Configuration}(undef, J)
    Threads.@threads for a = 1:J
        rdata = data[:,split .== a]
        config = init(rdata, k)
        lloyd!(config, rdata, 1_000, 1e-4, false)
        configs[a] = config
    end

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
    return mconfig
end

function init_centroid_nn(data::Matrix{Float64}, k::Int)
    m, n = size(data)

    centroids = copy(data)
    c = collect(1:n)
    costs = zeros(n)
    config = Configuration(m, n, n, c, costs, centroids)
    pairwise_nn!(config, k)
    partition_from_centroids!(config, data)
    return config
end

function init_centroid_refine(data::Matrix{Float64}, k::Int; init = init_centroid_pp, J = 10)
    m, n = size(data)
    @assert J * k ≤ n
    split = shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
    @assert all(sum(split .== a) ≥ k for a = 1:J)
    configs = Vector{Configuration}(undef, J)
    Threads.@threads for a = 1:J
        rdata = data[:,split .== a]
        config = init(rdata, k)
        lloyd!(config, rdata, 1_000, 1e-4, false)
        configs[a] = config
    end
    pool = hcat((configs[a].centroids for a in 1:J)...)
    for a = 1:J
        config = Configuration(pool, configs[a].centroids)
        lloyd!(config, pool, 1_000, 1e-4, false)
        configs[a] = config
    end
    a_best = argmin([configs[a].cost for a in 1:J])
    return Configuration(data, configs[a_best].centroids)
end

function init_centroid_scala(data::Matrix{Float64}, k::Int; rounds::Int = 5, ϕ::Float64 = 2.0)
    m, n = size(data)
    @assert n ≥ k

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
    cconfig = init_centroid_pp(centr, k; ncandidates=1, w=z)
    lloyd!(cconfig, centr, 1_000, 1e-4, false, z)
    return Configuration(data, cconfig.centroids)
end

function lloyd!(config::Configuration, data::Matrix{Float64}, max_it::Int, tol::Float64, verbose::Bool, w::Union{Vector{<:Real},Nothing} = nothing)
    cost0 = config.cost
    converged = false
    it = 0
    for outer it = 1:max_it
        centroids_from_partition!(config, data, w)
        old_cost = config.cost
        found_empty = check_empty!(config, data)
        partition_from_centroids!(config, data, w)
        new_cost = config.cost
        if new_cost ≥ old_cost * (1 - tol) && !found_empty
            verbose && println("converged cost = $new_cost")
            converged = true
            break
        end
        verbose && println("lloyd it = $it cost = $new_cost")
    end
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
It returns: 1) a vector of labels (`n` integers from 1 to `k`); 2) a `d`×`k` matrix of centroids;
3) the final cost; 4) whether it converged or not

It returns an object of type `Results`, which contains the following fields:
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take three
  values, `:collapsed`, `:maxgenerations` or `:noimprovement`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* cost: the final cost

The keyword arguments controlling the Lloyd's algorithm are:

* `max_it`: maximum number of iterations (default=1000). Normally the algorithm stops at fixed
  points.
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `init`: how to initialize (default=`"pnns"`). It can be a string or a `Matrix{Float64}`. If it's
  a matrix, it represents the initial centroids (by column). If it is a string, it can be one of the
  init algorithms described below.
* `tol`: a `Float64`, relative tolerance for detecting convergence (default=1e-5).
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.

The possible `init` methods are:

* `"unif"`: sample centroids uniformly at random from the dataset, without replacement
* `"++"`: kmeans++ by Arthur and Vassilvitskii (2006)
* `"maxmin"`: furthest-point heuristic, by Katsavounidis et al. (1994)
* `"scala"`: kmeans‖ also called "scalable kmeans++", by Bahmani et al. (2012)
* `"pnn"`: pairwise nearest-neighbor hierarchical clustering, by Equitz (1989)
* `"pnns"`: the PNN-smoothing meta-method
* `"refine"`: the refine meta-method by Bradely and Fayyad (1998)

The keyword arguments related to the `init` methods are:

* `init0`: the sub-initialization method to be used when `init="pnns"` or `init="refine"`; the
  argument can be any of the other methods listed above, or `"self"` for the fully-recursive
  version of `"pnns"`. If left empty (the default), then `"++"` with `ncandidates=1` will be used.
* `ρ`: a `Float64`, sets the number of sub-sets when `init="pnns"`. By default it is `0.5`.
  The formula is $⌈√(ρ N / k)⌉$ where $N$ is the number of data points and $k$ the number of
  clusters, but the result is clamped between `1` and `N÷k`.
* `J`: the number of sub-sets when `init="refine"`. By default it is `10`. Note that you must ensure
  that $J k ≤ N$.
* `ncandidates`: if init=="++" or init0="++", set the number of candidates for k-means++ (the default is
  `nothing`, which means that it is set automatically to `log(2+k)`)
* `rounds`: the number of rounds when `init="scala"` or `init0="scala"`.
"""
function kmeans(
        data::Matrix{Float64}, k::Integer;
        max_it::Integer = 1000,
        seed::Union{Integer,Nothing} = nothing,
        init::Union{AbstractString,Matrix{Float64}} = "pnns",
        verbose::Bool = true,
        tol::Float64 = 1e-5,
        ncandidates::Union{Nothing,Int} = nothing,
        ρ::Float64 = 0.5,
        J::Int = 10,
        rlevel::Int = 1,
        init0::AbstractString = "",
        rounds::Int = 5,
    )
    all_basic_methods = ["++", "unif", "pnn", "maxmin", "scala"]
    all_rec_methods = ["refine", "pnns"]
    all_methods = [all_basic_methods; all_rec_methods]
    if init isa AbstractString
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
    end

    if seed ≢ nothing
        Random.seed!(seed)
        if VERSION < v"1.7-"
            Threads.@threads for h = 1:Threads.nthreads()
                Random.seed!(seed + h)
            end
        end
    end
    m, n = size(data)

    if init isa AbstractString
        if init ∈ all_basic_methods
            if init == "++"
                config = init_centroid_pp(data, k; ncandidates)
            elseif init == "unif"
                config = init_centroid_unif(data, k)
            elseif init == "pnn"
                config = init_centroid_nn(data, k)
            elseif init == "maxmin"
                config = init_centroid_maxmin(data, k)
            elseif init == "scala"
                config = init_centroid_scala(data, k; rounds)
            else
                error("wat")
            end
        elseif init == "pnns" && init0 == "self"
            @assert rlevel == 0
            config = init_centroid_metann(data, k; ρ, init=recnninit)
        else
            @assert rlevel ≥ 1
            local metainit::Function
            if init == "refine"
                metainit = (data, k; kw...)->init_centroid_refine(data, k; J, kw...)
            elseif init == "pnns"
                metainit = (data, k; kw...)->init_centroid_metann(data, k; ρ, kw...)
            else
                error("wut")
            end
            local innerinit::Function
            if init0 == "++"
                innerinit = (data, k; kw...)->init_centroid_pp(data, k; ncandidates, kw...)
            elseif init0 == "unif"
                innerinit = (data, k; kw...)->init_centroid_unif(data, k; kw...)
            elseif init0 == "pnn"
                innerinit = (data, k; kw...)->init_centroid_nn(data, k; kw...)
            elseif init0 == "maxmin"
                innerinit = (data, k; kw...)->init_centroid_maxmin(data, k; kw...)
            elseif init0 == "scala"
                innerinit = (data, k; kw...)->init_centroid_scala(data, k; rounds, kw...)
            else
                error("wat")
            end
            wrappers = Vector{Function}(undef, rlevel)
            wrappers[1] = (data, k; kw...)->metainit(data, k; init=innerinit, kw...)
            for l in 2:rlevel
                wrappers[l] = (data, k; kw...)->metainit(data, k; init=wrappers[l-1], kw...)
            end
            config = wrappers[end](data, k)
        end
    else
        centroids = init
        size(centroids) == (m, k) || throw(ArgumentError("Incompatible init and data dimensions, data=$((m,k)) init=$(size(centroids))"))
        config = Configuration(data, centroids)
    end

    verbose && println("initial cost = $(config.cost)")
    converged = lloyd!(config, data, max_it, tol, verbose)

    exit_status = converged ? :converged : :maxiters

    return Results(exit_status, config)
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
