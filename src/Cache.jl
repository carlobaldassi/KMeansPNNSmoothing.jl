module Cache

let new_centroids_dict = Dict{NTuple{3,Int},Matrix{Float64}}(),
    new_centroids_thr_dict = Dict{NTuple{3,Int},Vector{Matrix{Float64}}}(),
    zs_dict = Dict{NTuple{2,Int},Vector{Float64}}(),
    zs_thr_dict = Dict{NTuple{2,Int},Vector{Vector{Float64}}}()

    global function new_centroids(m, k)
        return get!(new_centroids_dict, (Threads.threadid(),m,k)) do
            zeros(Float64, m, k)
        end
    end

    global function new_centroids_thr(m, k)
        return get!(new_centroids_thr_dict, (Threads.threadid(),m,k)) do
            [zeros(Float64, m, k) for id in 1:Threads.nthreads()]
        end
    end

    global function zs(k)
        return get!(zs_dict, (Threads.threadid(),k)) do
            zeros(Float64, k)
        end
    end

    global function zs_thr(k)
        return get!(zs_thr_dict, (Threads.threadid(),k)) do
            [zeros(Float64, k) for id in 1:Threads.nthreads()]
        end
    end

    global function clear!()
        empty!(new_centroids_dict)
        empty!(new_centroids_thr_dict)
        empty!(zs_dict)
        empty!(zs_thr_dict)
    end
end

end
