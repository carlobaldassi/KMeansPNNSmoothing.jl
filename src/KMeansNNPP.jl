# grouping method from Kaukoranta, Fränti, Nevlainen, "Reduced comparison for the exact GLA"

module KMeansNNPP

using Random
using Statistics
using StatsBase
using ExtractMacro

include("DataLogging.jl")
using .DataLogging

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

function Configuration(data::Matrix{Float64}, centroids::Matrix{Float64})
    m, n = size(data)
    k = size(centroids, 2)
    @assert size(centroids, 1) == m

    c = zeros(Int, n)
    costs = fill(Inf, n)
    config = Configuration(m, k, n, c, costs, centroids)
    partition_from_centroids!(config, data)
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

    config.k = k_new
    config.centroids = centroids
    config.csizes = csizes
    config.nonempty = nonempty

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


function partition_from_centroids!(config::Configuration, data::Matrix{Float64})
    @extract config: m k n c costs centroids active nonempty csizes
    @assert size(data) == (m, n)

    DataLogging.@push_prefix! "P_FROM_C"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    active_inds = findall(active)
    all_inds = collect(1:k)

    fill!(nonempty, false)
    fill!(csizes, 0)

    num_fullsearch = 0
    cost = 0.0
    t = @elapsed @inbounds for i in 1:n
        ci = c[i]
        if ci > 0 && active[ci]
            old_v′ = costs[i]
            @views new_v′ = _cost(data[:,i], centroids[:,ci])
            fullsearch = (new_v′ > old_v′)
        else
            fullsearch = (ci == 0)
        end
        num_fullsearch += fullsearch

        v, x, inds = fullsearch ? (Inf, 0, all_inds) : (costs[i], ci, active_inds)
        for j in inds
            @views v′ = _cost(data[:,i], centroids[:,j])
            if v′ < v
                v, x = v′, j
            end
        end
        costs[i], c[i] = v, x
        nonempty[x] = true
        csizes[x] += 1
        cost += v
    end

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost fullsearches: $num_fullsearch / $n"
    DataLogging.@pop_prefix!
    return config
end

let centroidsdict = Dict{NTuple{2,Int},Matrix{Float64}}()

    global function centroids_from_partition!(config::Configuration, data::Matrix{Float64})
        @extract config: m k n c costs centroids active nonempty csizes
        @assert size(data) == (m, n)

        new_centroids = get!(centroidsdict, (m,k)) do
            zeros(Float64, m, k)
        end
        fill!(new_centroids, 0.0)
        fill!(active, false)
        @inbounds for i = 1:n
            j = c[i]
            for l = 1:m
                new_centroids[l,j] += data[l,i]
            end
        end
        @inbounds for j = 1:k
            z = csizes[j]
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
    DataLogging.@push_prefix! "INIT_UNIF"
    m, n = size(data)
    DataLogging.@log "INPUTS m: $m n: $n k: $k"
    t = @elapsed config = begin
        centroids = zeros(m, k)
        for j = 1:k
            i = rand(1:n)
            centroids[:,j] .= data[:,i]
        end
        Configuration(data, centroids)
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function compute_costs_one!(costs::Vector{Float64}, data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    @inbounds for i = 1:n
        @views costs[i] = _cost(data[:,i], x)
    end
    return costs
end
compute_costs_one(data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64}) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, x)

function init_centroid_pp(data::Matrix{Float64}, k::Int; ncandidates = nothing)
    DataLogging.@push_prefix! "INIT_PP"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    ncandidates::Int = ncandidates ≡ nothing ? floor(Int, 2 + log(k)) : ncandidates

    DataLogging.@log "LOCAL_VARS n: $n ncandidates: $ncandidates"

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = rand(1:n)
        datay = data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)

        curr_cost = sum(costs)
        c = ones(Int, n)

        new_costs, new_c = similar(costs), similar(c)
        new_costs_best, new_c_best = similar(costs), similar(c)
        for j = 2:k
            pw = Weights(costs)
            candidates = sample(1:n, pw, min(ncandidates,n), replace = false)
            cost_best = Inf
            y_best = 0
            for y in candidates
                datay = data[:,y]
                compute_costs_one!(new_costs, data, datay)
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

function init_centroid_maxmin(data::Matrix{Float64}, k::Int)
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

        cost = sum(costs)
        c = ones(Int, n)

        for j = 2:k
            y = argmax(costs)
            datay = data[:,y]
            @inbounds for i = 1:n
                old_v = costs[i]
                new_v = _cost(@view(data[:,i]), datay)
                if new_v < old_v
                    costs[i] = new_v
                    c[i] = j
                    cost += new_v - old_v
                end
            end
            centr[:,j] .= datay
        end
        # returning config
        Configuration(m, k, n, c, costs, centr)
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost)"
    DataLogging.@pop_prefix!
    return config
end

function pairwise_nn!(config::Configuration, tgt_k::Int)
    @extract config : m k n c costs centroids csizes
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
    t_costs = @elapsed @inbounds for j = 1:k
        z = csizes[j]
        v, x = Inf, 0
        for j′ = 1:k
            j′ == j && continue
            z′ = csizes[j′]
            @views v1 = (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
            if v1 < v
                v = v1
                x = j′
            end
        end
        nns_costs[j], nns[j] = v, x
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
                z = csizes[j]
                v, x = Inf, 0
                for j′ = 1:(k-1)
                    j′ == j && continue
                    z′ = csizes[j′]
                    @views v1 = (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
                    if v1 < v
                        v, x = v1, j′
                    end
                end
                nns_costs[j], nns[j] = v, x
            # 2) clusters that did not point to jm or js
            #    only compare the old cost with the cost for the updated cluster
            else
                z = csizes[j]
                # note: what used to point at k now must point at js
                v, x = nns_costs[j], (nns[j] ≠ k ? nns[j] : js)
                j′ = jm
                z′ = csizes[j′]
                @views v′ = (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
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


function init_centroid_ppnn(data::Matrix{Float64}, k::Int; ncandidates = nothing, ρ = 0.5)
    DataLogging.@push_prefix! "INIT_PPNN"
    m, n = size(data)
    J = clamp(ceil(Int, √(ρ * n / k)), 1, n ÷ k)
    @assert J * k ≤ n
    (J == 1 || J == n ÷ k) && @warn "edge case: J = $J"
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"

    t = @elapsed mconfig = begin
        tpp = @elapsed configs = begin
            split = shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
            @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration}(undef, J)
            for a = 1:J
                rdata = data[:,split .== a]
                DataLogging.@push_prefix! "SPLIT=$a"
                config = init_centroid_pp(rdata, k; ncandidates)
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

function recnninit(data::Matrix{Float64}, k::Int)
    m, n = size(data)
    if n ≤ 2k
        return init_centroid_nn(data, k)
    else
        return init_centroid_metann(data, k; init = recnninit)
    end
end

function init_centroid_metann(data::Matrix{Float64}, k::Int; init = init_centroid_maxmin, ρ = 0.5)
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
            for a = 1:J
                rdata = data[:,split .== a]
                DataLogging.@push_prefix! "SPLIT=$a"
                config = init(rdata, k)
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

function init_centroid_hnn(data::Matrix{Float64}, k::Int)
    DataLogging.@push_prefix! "INIT_HNN"
    m, n = size(data)
    J = max(n ÷ 2k, 1)
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"

    t = @elapsed mconfig = begin
        split_inds = [Int[] for a = 1:J]
        split_data = Vector{Matrix{Float64}}(undef, J)
        t0 = @elapsed configs = begin
            ls = diff([round(Int,x) for x in range(1, n+1, length=J+1)])
            split = shuffle!(vcat((repeat([i], ls[i]) for i in 1:J)...))
            @assert length(split) == n
            for i = 1:n
                a = split[i]
                push!(split_inds[a], i)
            end
            @assert all(length(split_inds[a]) ≥ 2k for a = 1:J)
            for a = 1:J
                split_data[a] = data[:,split_inds[a]]
            end
            configs = Vector{Configuration}(undef, J)
            for a = 1:J
                rdata = split_data[a]
                DataLogging.@push_prefix! "SPLIT=$a"
                config = init_centroid_nn(rdata, k)
                lloyd!(config, rdata, 1_000, 1e-4, false)
                DataLogging.@pop_prefix!
                configs[a] = config
            end
            configs
        end
        # println(collect(config.cost for config in configs))
        DataLogging.@log "STEP0DONE time: $t0"

        tm = @elapsed fconfig = begin
            while J > 1
                # @assert sum(size(sd,2) for sd in split_data) == n
                J_new = J ÷ 2 + isodd(J)
                split_data_new = Vector{Matrix{Float64}}(undef, J_new)
                configs_new = Vector{Configuration}(undef, J_new)
                for b in 1:(J÷2)
                    a1, a2 = 2b-1, 2b
                    centroids_new = hcat(configs[a1].centroids, configs[a2].centroids)
                    n1, n2 = size(split_data[a1], 2), size(split_data[a2], 2)
                    nc = n1 + n2
                    c_new = zeros(Int, nc)
                    costs_new = zeros(nc)
                    for i = 1:n1
                        c_new[i] = configs[a1].c[i]
                        costs_new[i] = configs[a1].costs[i]
                    end
                    for i = 1:n2
                        c_new[n1+i] = configs[a2].c[i] + k
                        costs_new[n1+i] = configs[a2].costs[i]
                    end
                    rdata_new = hcat(split_data[a1], split_data[a2])
                    mconfig = Configuration(m, 2k, nc, c_new, costs_new, centroids_new)
                    pairwise_nn!(mconfig, k)
                    partition_from_centroids!(mconfig, rdata_new)
                    J_new > 1 && lloyd!(mconfig, rdata_new, 1_000, 1e-4, false)
                    split_data_new[b+isodd(J)] = rdata_new
                    configs_new[b+isodd(J)] = mconfig
                end
                if isodd(J)
                    split_data_new[1] = split_data[J]
                    configs_new[1] = configs[J]
                end
                J, split_data, configs = J_new, split_data_new, configs_new
                # println(collect(config.cost for config in configs))
            end
            # lloyd!(configs[1], split_data[1], 1_000, 0.0, true)
            fconfig = Configuration(data, configs[1].centroids)
        end
        DataLogging.@log "NNMRGDONE time: $tm"
        fconfig
    end
    DataLogging.@log "DONE time: $t cost: $(mconfig.cost)"
    DataLogging.@pop_prefix!
    return mconfig
end

function init_centroid_nn(data::Matrix{Float64}, k::Int)
    DataLogging.@push_prefix! "INIT_NN"
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

function init_centroid_refine(data::Matrix{Float64}, k::Int; init = init_centroid_pp, J = 10)
    DataLogging.@push_prefix! "INIT_REFINE"
    m, n = size(data)
    @assert J * k ≤ n
    # if init == "++"
    #     init_func = (data,k)->init_centroid_pp(data, k; ncandidates)
    # elseif init == "unif"
    #     init_func = init_centroid_unif
    # else
    #     error()
    # end
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"
    t = @elapsed mconfig = begin
        tinit = @elapsed configs = begin
            split = shuffle!(vcat((repeat([a], k) for a = 1:J)..., rand(1:J, (n - k*J))))
            @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration}(undef, J)
            for a = 1:J
                rdata = data[:,split .== a]
                DataLogging.@push_prefix! "SPLIT=$a"
                config = init(rdata, k)
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

function lloyd!(config::Configuration, data::Matrix{Float64}, max_it::Int, tol::Float64, verbose::Bool)
    DataLogging.@push_prefix! "LLOYD"
    DataLogging.@log "INPUTS max_it: $max_it tol: $tol"
    cost0 = config.cost
    converged = false
    it = 0
    t = @elapsed for outer it = 1:max_it
        centroids_from_partition!(config, data)
        old_cost = config.cost
        found_empty = check_empty!(config, data)
        partition_from_centroids!(config, data)
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

function kmeans(
        data::Matrix{Float64}, k::Integer;
        max_it::Integer = 1000,
        seed::Union{Integer,Nothing} = nothing,
        init::Union{String,Matrix{Float64}} = "++",
        verbose::Bool = true,
        tol::Float64 = 1e-5,
        ncandidates::Union{Nothing,Int} = nothing,
        ρ::Float64 = 0.5,
        logfile::AbstractString = "",
        J::Int = 10,
        rlevel::Int = 0,
        init0::String = "",
    )
    # allmethods = ["++", "unif", "++nn", "nn", "refine", "refine++", "hnn", "maxmin", "hnn2", "maxminnn", "maxminnnnn", "maxmi7n"]
    all_basic_methods = ["++", "unif", "nn", "maxmin", "hnn"]
    all_rec_methods = ["refine", "smoothnn"]
    all_methods = [all_basic_methods; all_rec_methods]
    if init isa String
        init ∈ all_methods || throw(ArgumentError("init should either be a matrix or one of: $all_methods"))
        if init ∈ all_rec_methods
            if init0 ∈ all_basic_methods
                rlevel ≤ 0 && (rlevel = 1)
            elseif init0 == "self"
                init == "smoothnn" || throw(ArgumentError("init0=$init0 unsupported with init=$init"))
                rlevel == 0 || @warn("Ignoring rlevel=$rlevel with init=$init and init0=$init0")
            else
                throw(ArgumentError("when init=$init, init0 should be \"self\" or one of: $all_basic_methods"))
            end
        else
            init0 == "" || @warn("Ignoring init0=$init0 with init=$init")
            rlevel == 0 || @warn("Ignoring rlevel=$rlevel with init=$init")
        end
    end

    logger = if !isempty(logfile)
        if DataLogging.logging_on
            init_logger(logfile, "w")
        else
            @warn "logging is off, ignoring logfile"
            nothing
        end
    end

    DataLogging.@push_prefix! "KMEANS"

    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)
    DataLogging.@log "INPUTS m: $m n: $n k: $k seed: $seed"

    if init isa String
        if init ∈ all_basic_methods
            if init == "++"
                config = init_centroid_pp(data, k; ncandidates)
            elseif init == "unif"
                config = init_centroid_unif(data, k)
            elseif init == "nn"
                config = init_centroid_nn(data, k)
            elseif init == "maxmin"
                config = init_centroid_maxmin(data, k)
            elseif init == "hnn"
                config = init_centroid_hnn(data, k)
            else
                error("wat")
            end
        elseif init == "smoothnn" && init0 == "self"
            @assert rlevel == 0
            config = init_centroid_metann(data, k; ρ, init=recnninit)
        else
            @assert rlevel ≥ 1
            local metainit::Function
            if init == "refine"
                metainit = (data, k; kw...)->init_centroid_refine(data, k; J, kw...)
            elseif init == "smoothnn"
                metainit = (data, k; kw...)->init_centroid_metann(data, k; ρ, kw...)
            else
                error("wut")
            end
            local innerinit::Function
            if init0 == "++"
                innerinit = (data, k; kw...)->init_centroid_pp(data, k; ncandidates, kw...)
            elseif init0 == "unif"
                innerinit = (data, k; kw...)->init_centroid_unif(data, k; kw...)
            elseif init0 == "nn"
                innerinit = (data, k; kw...)->init_centroid_nn(data, k; kw...)
            elseif init0 == "maxmin"
                innerinit = (data, k; kw...)->init_centroid_maxmin(data, k; kw...)
            elseif init0 == "hnn"
                innerinit = (data, k; kw...)->init_centroid_hnn(data, k; kw...)
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
    DataLogging.@log "INIT_COST" config.cost
    converged = lloyd!(config, data, max_it, tol, verbose)
    DataLogging.@log "FINAL_COST" config.cost

    DataLogging.@pop_prefix!
    logger ≢ nothing && pop_logger!()

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
