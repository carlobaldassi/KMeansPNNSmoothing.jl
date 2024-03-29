"""
    Refine(init0=KMSeed.PlusPlus{1}(); J=10, rlevel=1)

The seeder for the "refine" algorithm. The first argument `init0` can be any
other seeder. The argument `J` sets the number of sub-sets. The argument `rlevel` sets the
recursion level.

See also: `kmeans`, `KMSeed`.
"""
struct Refine{S<:Seeder} <: MetaSeeder{S}
    init0::S
    J::Int
end
function Refine(init0::S = PlusPlus{1}(); J = 10, rlevel::Int = 1) where {S <: Seeder}
    @assert rlevel ≥ 1
    kmseeder = init0
    for r = rlevel:-1:1
        kmseeder = Refine{typeof(kmseeder)}(kmseeder, J)
    end
    return kmseeder
end


function init_centroids(S::Refine{S0}, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) where S0
    @extract S : J
    m, n = size(data)
    @assert J * k ≤ n
    # split = gen_random_splits(n, J, k)
    split = gen_fair_splits(n, J)
    # @assert all(sum(split .== a) ≥ k for a = 1:J)
    configs = Vector{Configuration{A}}(undef, J)
    @bthreads for a = 1:J
        rdata = KMMatrix(data.dmat[:,split .== a])
        config = inner_init(S, rdata, k, A)
        lloyd!(config, rdata, 1_000, 0.0, false)
        configs[a] = config
    end
    pool = KMMatrix(hcat((configs[a].centroids.dmat for a in 1:J)...))
    pconfigs = Vector{Configuration{A}}(undef, J)
    for a = 1:J
        config = Configuration{A}(pool, configs[a].centroids)
        lloyd!(config, pool, 1_000, 0.0, false)
        configs[a] = config
    end
    a_best = argmin([configs[a].cost for a in 1:J])
    mconfig = Configuration{A}(data, configs[a_best].centroids)
    return mconfig
end

