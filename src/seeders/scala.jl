"""
    Scala(;rounds=5, ϕ=2.0)

The seeder for "scalable kmeans++" or kmeans‖. The `rounds` option
determines the number of sampling rounds; the `ϕ` option determines the
oversampling factor, which is then computed as `ϕ * k`.

See also: `kmeans`, `KMSeed`.
"""
struct Scala <: Seeder
    rounds::Int
    ϕ::Float64
    Scala(;rounds = 5, ϕ = 2.0) = new(rounds, ϕ)
end

function init_centroids(S::Scala, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    @extract S : rounds ϕ
    DataLogging.@push_prefix! "INIT_SCALA"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"
    DataLogging.@exec dist_comp = 0

    t = @elapsed config = begin
        centr = zeros(m, 1)
        y = rand(1:n)
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)
        DataLogging.@exec dist_comp += n

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
                    DataLogging.@exec dist_comp += add_k
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
        cconfig = init_centroids(PlusPlus{1}(), centroids, k, ReducedComparison; w=z)
        lloyd!(cconfig, centroids, 1_000, 0.0, false, z)
        Configuration{A}(data, cconfig.centroids)
        # mconfig = Configuration{A}(m, k′, n, c, costs, centr)
        # pairwise_nn!(mconfig, k)
        # partition_from_centroids!(mconfig, data)
        # mconfig
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost) dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end

