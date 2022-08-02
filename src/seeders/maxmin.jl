"""
    MaxMin()

The seeder for the furthest-point heuristic, also called maxmin.

See also: `kmeans`, `KMSeed`.
"""
struct MaxMin <: Seeder
end

function init_centroids(::MaxMin, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    m, n = size(data)
    @assert n ≥ k

    config = @inbounds begin
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
    return config
end
