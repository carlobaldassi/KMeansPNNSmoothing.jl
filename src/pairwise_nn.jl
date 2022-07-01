Base.@propagate_inbounds function _merge_cost(centroids::AbstractMatrix{<:Float64}, z::Int, z′::Int, j::Int, j′::Int)
    return @views (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
end

function _get_nns(vs, j, k, centroids, csizes)
    k < 500 && return _get_nns(j, k, centroids, csizes)
    z = csizes[j]
    fill!(vs, (Inf, 0))
    Threads.@threads for j′ = 1:k
        j′ == j && continue
        @inbounds begin
            v1 = _merge_cost(centroids, z, csizes[j′], j, j′)
            id = Threads.threadid()
            if v1 < vs[id][1]
                vs[id] = v1, j′
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
        v1 = _merge_cost(centroids, z, csizes[j′], j, j′)
        if v1 < v
            v, x = v1, j′
        end
    end
    return v, x
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
    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))
    t_costs = @elapsed @inbounds for j = 1:k
        nns_costs[j], nns[j] = _get_nns(vs, j, k, cmat, csizes)
        DataLogging.@exec dist_comp += k-1
    end

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

        num_fullupdates = 0
        for j = 1:(k-1)
            # 1) merged cluster jm, or clusters which used to point to either jm or js
            #    perform a full update
            if j == jm || nns[j] == jm || nns[j] == js
                num_fullupdates += 1
                nns_costs[j], nns[j] = _get_nns(vs, j, k-1, cmat, csizes)
                DataLogging.@exec dist_comp += k-2
            # 2) clusters that did not point to jm or js
            #    only compare the old cost with the cost for the updated cluster
            else
                z = csizes[j]
                # note: what used to point at k now must point at js
                v, x = nns_costs[j], (nns[j] ≠ k ? nns[j] : js)
                j′ = jm
                z′ = csizes[j′]
                v′ = _merge_cost(cmat, z, z′, j, j′)
                DataLogging.@exec dist_comp += 1
                if v′ < v
                    v, x = v′, j′
                end
                nns_costs[j], nns[j] = v, x
                @assert nns[j] ≠ j
            end
        end
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

