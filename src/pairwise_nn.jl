Base.@propagate_inbounds function _merge_cost(centroids::AbstractMatrix{<:Float64}, z::Int, z′::Int, j::Int, j′::Int)
    return @views _merge_cost(centroids[:,j], centroids[:,j′], z, z′)
end

Base.@propagate_inbounds function _merge_cost(centr::AbstractVector{Float64}, centr′::AbstractVector{Float64}, z::Int, z′::Int)
    return _merge_cost(_cost(centr, centr′), z, z′)
end

Base.@propagate_inbounds function _merge_cost(cost::Float64, z::Int, z′::Int)
    return (z * z′) / (z + z′) * cost
end

function _get_all_nns(centroids::Matrix{Float64}, csizes::Vector{Int})
    nt = Threads.nthreads()
    k = length(csizes)
    @assert size(centroids, 2) == k
    (nt == 1 || k < 500) && return _get_all_nns_nothreads(centroids, csizes)
    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))
    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    @inbounds for j = 1:k
        nns_costs[j], nns[j] = _get_nns(vs, j, k, centroids, csizes)
        DataLogging.@exec dist_comp += k-1
    end
    return nns, nns_costs
end

function _get_nns(vs, j, k, centroids, csizes)
    k < 500 && return _get_nns(j, k, centroids, csizes)
    z = csizes[j]
    fill!(vs, (Inf, 0))
    Threads.@threads for j′ = 1:k
        j′ == j && continue
        @inbounds begin
            z′ = csizes[j′]
            v′ = _merge_cost(centroids, z, z′, j, j′)
            id = Threads.threadid()
            if v′ < vs[id][1]
                vs[id] = v′, j′
            end
        end
    end
    v, x = Inf, 0
    for id in 1:Threads.nthreads()
        if vs[id][1] < v
            v, x = vs[id]
        end
    end
    return v, x
end

function _get_nns(j, k, centroids, csizes)
    z = csizes[j]
    v, x = Inf, 0
    @inbounds for j′ = 1:k
        j′ == j && continue
        z′ = csizes[j′]
        v′ = _merge_cost(centroids, z, z′, j, j′)
        if v′ < v
            v, x = v′, j′
        end
    end
    return v, x
end

function _update_nns!(vs, nns_costs, nns, j, k, centroids, csizes)
    nt = Threads.nthreads()
    (nt == 1 || k < 500) && return _update_nns!(nns_costs, nns, j, k, centroids, csizes)
    z = csizes[j]
    fill!(vs, (Inf, 0))
    Threads.@threads for j′ = 1:k
        j′ == j && continue
        @inbounds begin
            z′ = csizes[j′]
            v′ = _merge_cost(centroids, z, z′, j, j′)
            id = Threads.threadid()
            if v′ < vs[id][1]
                vs[id] = v′, j′
            end
            if v′ < nns_costs[j′]
                nns_costs[j′], nns[j′] = v′, j
            end
        end
    end
    v, x = Inf, 0
    for id in 1:nt
        if vs[id][1] < v
            v, x = vs[id]
        end
    end
    nns_costs[j], nns[j] = v, x
end

function _update_nns!(nns_costs, nns, j, k, centroids, csizes)
    z = csizes[j]
    v, x = Inf, 0
    @inbounds for j′ = 1:k
        j′ == j && continue
        z′ = csizes[j′]
        v′ = _merge_cost(centroids, z, z′, j, j′)
        if v′ < v
            v, x = v′, j′
        end
        if v′ < nns_costs[j′]
            nns_costs[j′], nns[j′] = v′, j
        end
    end
    nns_costs[j], nns[j] = v, x
end


function pairwise_nn(config::Configuration, tgt_k::Int, data::Mat64, ::Type{A}) where {A<:Accelerator}
    @extract config : m k n centroids csizes
    @extract centroids : cmat=dmat
    DataLogging.@push_prefix! "PNN"
    DataLogging.@log "INPUTS k: $k tgt_k: $tgt_k"
    if k < tgt_k
        get_logger() ≢ nothing && pop_logger!()
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    if k == tgt_k
        DataLogging.@pop_prefix!
        return Configuration{A}(data, centroids)
    end
    DataLogging.@exec dist_comp = 0

    # csizes′ = zeros(Int, k)
    # for i = 1:n
    #     csizes′[c[i]] += 1
    # end
    # @assert all(csizes′ .> 0)
    # @assert csizes == csizes′
    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))

    t_costs = @elapsed nns, nns_costs = _get_all_nns(cmat, csizes)
    DataLogging.@exec dist_comp += (k * (k-1)) ÷ 2

    to_be_updated = Int[]

    t_fuse = @elapsed @inbounds while k > tgt_k
        jm = argmin(@view(nns_costs[1:k]))
        js = nns[jm]
        @assert nns_costs[js] == nns_costs[jm]
        @assert jm < js

        DataLogging.@push_prefix! "K=$k"
        DataLogging.@log "jm: $jm js: $js cost: $(nns_costs[jm])"

        # update centroid
        zm, zs = csizes[jm], csizes[js]
        for l = 1:m
            cmat[l,jm] = (zm * cmat[l,jm] + zs * cmat[l,js]) / (zm + zs)
            cmat[l,js] = cmat[l,k]
        end

        if k-1 == tgt_k
            # we're done, skip the last update
            k -= 1
            DataLogging.@pop_prefix!
            break
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

        # We need to keep track of which clusters used to point at jm or js
        # before we update the ones that pointed to k (now that cluster k
        # will become js)
        empty!(to_be_updated)
        for j = 1:(k-1)
            j == jm && continue
            j′ = nns[j]
            (j′ == jm || j′ == js) && push!(to_be_updated, j)
            j′ == k && (nns[j] = js)
        end

        # This performs a full update of the merged cluster jm
        # We also use the cost computations to update the min costs
        # of the other j′ in case neither jm nor js were their nearest neighbor
        # but now they are
        _update_nns!(vs, nns_costs, nns, jm, k-1, cmat, csizes)

        # Clusters which used to point at jm or js need to undergo a full update
        for j in to_be_updated
            nns_costs[j], nns[j] = _get_nns(vs, j, k-1, cmat, csizes)
        end
        num_fullupdates = 1 + length(to_be_updated)
        DataLogging.@exec dist_comp += num_fullupdates * (k-2)

        DataLogging.@log "fullupdates: $num_fullupdates"
        DataLogging.@pop_prefix!

        k -= 1
    end

    @assert all(csizes[1:k] .> 0)

    mconfig = Configuration{A}(data, KMMatrix(cmat[:,1:k]))

    DataLogging.@log "DONE t_costs: $t_costs t_fuse: $t_fuse dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return mconfig
end

