struct _Self <: Seeder
end

"""
    PNNS(init0=PlusPlus{1}(); ρ=1.0, rlevel=1)

The seeder for the PNN-smoothing algorithm. The first argument `init0` can be any
other seeder. The argument `ρ` sets the number of sub-sets, using the formula ``⌈√(ρ N / k)⌉``
where ``N`` is the number of data points and ``k`` the number of clusters, but the result is
clamped between `1` and `N÷k`. The argument `rlevel` sets the recursion level.

See `PNNSR` for the fully-recursive version.

See also: `kmeans`, `KMSeed`.
"""
struct PNNS{S<:Seeder} <: MetaSeeder{S}
    init0::S
    ρ::Float64
end
function PNNS(init0::S = PlusPlus{1}(); ρ = 1.0, rlevel::Int = 1) where {S <: Seeder}
    @assert rlevel ≥ 1
    kmseeder = init0
    for r = rlevel:-1:1
        kmseeder = PNNS{typeof(kmseeder)}(kmseeder, ρ)
    end
    return kmseeder
end

const PNNSR = PNNS{_Self}

"""
    PNNSR(;ρ=1.0)

The fully-recursive version of the `PNNS` seeder. It keeps splitting the dataset until the
number of points is ``≤2k``, at which point it uses `PNN`. The `ρ` option is documented in `PNNS`.
"""
PNNSR(;ρ = 1.0) = PNNS{_Self}(_Self(), ρ)


getJ(n, k, ρ) = clamp(ceil(Int, √(ρ * n / 2k)), 1, n ÷ k)

function inner_init(S::PNNSR, data::Mat64, k::Int, A::Type{<:Accelerator})
    @extract S : ρ
    m, n = size(data)
    if getJ(n, k, ρ) == 1
        return init_centroids(PNN(), data, k, A)
    else
        return init_centroids(S, data, k, A)
    end
end

inner_init(S::MetaSeeder{S0}, data::Mat64, k::Int, A::Type{<:Accelerator}) where S0 = init_centroids(S.init0, data, k, A)

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

function init_centroids(S::PNNS{S0}, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) where S0
    @extract S : ρ
    DataLogging.@push_prefix! "INIT_METANN"
    m, n = size(data)
    J = getJ(n, k, ρ)
    @assert J * k ≤ n
    S0 == _Self && J ≤ 1 && error("The PNNSR seeder requires that ρ * n > 2k, given ρ=$ρ n=$n k=$k")
    # (J == 1 || J == n ÷ k) && @warn "edge case: J = $J"
    DataLogging.@log "INPUTS m: $m n: $n k: $k J: $J"

    t = @elapsed mconfig = begin
        tpp = @elapsed configs = begin
            # split = gen_random_splits(n, J, k)
            split = gen_fair_splits(n, J)
            # @assert all(sum(split .== a) ≥ k for a = 1:J)
            configs = Vector{Configuration{A}}(undef, J)
            @bthreads for a = 1:J
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
            mconfig = m < 100 ?
                pairwise_nn(jconfig, k, data, A) :
                pairwise_nn_savedist(jconfig, k, data, A)
            mconfig
        end
        DataLogging.@log "NNDONE time: $tnn"
        mconfig
    end
    DataLogging.@log "DONE time: $t cost: $(mconfig.cost)"
    DataLogging.@pop_prefix!
    return mconfig
end
