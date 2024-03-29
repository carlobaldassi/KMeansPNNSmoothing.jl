"""
    PlusPlus()
    PlusPlus{NC}()

The seeder for kmeans++. The parameter `NC` determines the number of
candidates for the "greedy" version of the algorithm. By default, it is `nothing`,
meaning that it will use the value `log(2+k)`. Otherwise it must be an integer.
The value `1` corresponds to the original (non-greedy) algorithm, which has a
specialized implementation.

See also: `kmeans`, `KMSeed`.
"""
struct PlusPlus{NC} <: Seeder
end

PlusPlus() = PlusPlus{nothing}()


init_centroids(::PlusPlus{nothing}, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) =
    init_centroids(PlusPlus{floor(Int, 2 + log(k))}(), data, k, A; kw...)

function init_centroids(::PlusPlus{NC}, data::Mat64, k::Int, A::Type{<:Accelerator}; w = nothing) where NC
    m, n = size(data)
    @assert n ≥ k

    ncandidates::Int = NC

    @inbounds config = begin
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
    return config
end

function init_centroids(::PlusPlus{1}, data::Mat64, k::Int, A::Type{<:Accelerator}; w = nothing)
    m, n = size(data)
    @assert n ≥ k

    config = @inbounds begin
        centr = zeros(m, k)
        y = (w ≡ nothing ? rand(1:n) : sample(1:n, Weights(w)))
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay, w)

        c = ones(Int, n)

        for j = 2:k
            pw = Weights(w ≡ nothing ? costs : costs .* w)
            y = sample(1:n, pw)
            datay = @view data[:,y]

            update_costs_one!(costs, c, j, data, datay, w)

            centr[:,j] .= datay
        end
        # returning config
        Configuration{A}(data, c, costs, KMMatrix(centr))
    end
    return config
end

