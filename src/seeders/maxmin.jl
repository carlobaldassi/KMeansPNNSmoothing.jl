"""
    KMMaxMin()

A `KMeansSeeder` for the furthest-point heuristic, also called maxmin.
"""
struct KMMaxMin <: KMeansSeeder
end

function init_centroids(::KMMaxMin, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...)
    DataLogging.@push_prefix! "INIT_MAXMIN"
    m, n = size(data)
    @assert n ≥ k

    DataLogging.@log "INPUTS m: $m n: $n k: $k"
    DataLogging.@exec dist_comp = 0

    t = @elapsed config = begin
        centr = zeros(m, k)
        y = rand(1:n)
        datay = @view data[:,y]
        centr[:,1] = datay

        costs = compute_costs_one(data, datay)
        DataLogging.@exec dist_comp += n

        c = ones(Int, n)

        for j = 2:k
            y = argmax(costs)
            datay = @view data[:,y]

            update_costs_one!(costs, c, j, data, datay)
            DataLogging.@exec dist_comp += n

            centr[:,j] .= datay
        end
        # returning config
        Configuration{A}(data, c, costs, KMMatrix(centr))
    end
    DataLogging.@log "DONE time: $t cost: $(config.cost) dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end
