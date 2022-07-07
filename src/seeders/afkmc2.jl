"""
    AFKMC2{M}()

The seeder for Assumption-Free K-MC². The parameter `M` determines the
number of Monte Carlo steps. This algorithm is implemented in a way that is O(kND)
instead of O(k²M) because we still need to compute the partition by the end. So it
is provided only for testing; for practical purposes `KMSeed.PlusPlus` should be
preferred.

See also: `kmeans`, `KMSeed`.
"""
struct AFKMC2{M} <: Seeder
end

function init_centroids(::AFKMC2{L}, data::Mat64, k::Int, A::Type{<:Accelerator}; w = nothing) where L
    DataLogging.@push_prefix! "INIT_AFKMC2"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"

    @assert L isa Int

    DataLogging.@log "LOCAL_VARS n: $n L: $L"
    DataLogging.@exec dist_comp = 0

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = zeros(n)
        _costs_1_vs_all!(costs, data, y, data, w)
        DataLogging.@exec dist_comp += n
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
            DataLogging.@exec dist_comp += n
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
    DataLogging.@log "DONE time: $t cost: $(config.cost) dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end


