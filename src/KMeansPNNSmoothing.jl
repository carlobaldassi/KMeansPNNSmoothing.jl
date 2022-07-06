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
       KMAccel,
       KMeansSeeder, KMMetaSeeder,
       KMUnif, KMMaxMin, KMScala, KMPlusPlus, KMPNN,
       KMPNNS, KMPNNSR, KMRefine, KMAFKMC2

include("Cache.jl")
include("Utils.jl")
using .Utils

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

function update_csizes!(config::Configuration)
    @extract config: n c csizes
    fill!(csizes, 0)
    @inbounds for i in 1:n
        csizes[c[i]] += 1
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

include("accelerators.jl")

include("pairwise_nn.jl")

include("seeders.jl")

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

    Cache.clear!()

    return Results(exit_status, config)
end

## Centroid Index
## P. Fränti, M. Rezaei and Q. Zhao, Centroid index: cluster level similarity measure, Pattern Recognition, 2014
function CI(true_centroids::Matrix{Float64}, centroids::Matrix{Float64})
    m, tk = size(true_centroids)
    @assert size(centroids, 1) == m
    k = size(centroids, 2)

    matched = falses(tk)
    @inbounds for j = 1:k
        v, p = Inf, 0
        for tj = 1:tk
            @views v1 = _cost(true_centroids[:,tj], centroids[:,j])
            if v1 < v
                v, p = v1, tj
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

## Variation of Information
## M. Meilă, Comparing clusterings—an information based distance, Journal of multivariate analysis, 2007

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
