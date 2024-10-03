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
    end
    return nns, nns_costs
end

function _get_all_nns_nothreads(centroids::Matrix{Float64}, csizes::Vector{Int})
    k = length(csizes)
    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    @inbounds for j = 1:k
        z = csizes[j]
        for j′ = 1:(j-1)
            z′ = csizes[j′]
            v′ = _merge_cost(centroids, z, z′, j, j′)
            if v′ < nns_costs[j]
                nns_costs[j], nns[j] = v′, j′
            end
            if v′ < nns_costs[j′]
                nns_costs[j′], nns[j′] = v′, j
            end
        end
    end
    return nns, nns_costs
end

function _get_nns(vs, j, k, centroids, csizes)
    nt = Threads.nthreads()
    (nt == 1 || k < 500) && return _get_nns(j, k, centroids, csizes)
    z = csizes[j]
    fill!(vs, (Inf, 0))
    Threads.@threads :static for j′ = 1:k
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
    for id in 1:nt
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
    Threads.@threads :static for j′ = 1:k
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
    if k < tgt_k
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    k == tgt_k && return Configuration{A}(data, centroids)

    # csizes′ = zeros(Int, k)
    # for i = 1:n
    #     csizes′[c[i]] += 1
    # end
    # @assert all(csizes′ .> 0)
    # @assert csizes == csizes′
    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))

    nns, nns_costs = _get_all_nns(cmat, csizes)

    to_be_updated = Int[]

    @inbounds while k > tgt_k
        jm = argmin(@view(nns_costs[1:k]))
        js = nns[jm]
        @assert nns_costs[js] == nns_costs[jm]
        @assert jm < js

        # update centroid
        zm, zs = csizes[jm], csizes[js]
        for l = 1:m
            cmat[l,jm] = (zm * cmat[l,jm] + zs * cmat[l,js]) / (zm + zs)
            cmat[l,js] = cmat[l,k]
        end

        if k-1 == tgt_k
            # we're done, skip the last update
            k -= 1
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

        k -= 1
    end

    @assert all(csizes[1:k] .> 0)

    mconfig = Configuration{A}(data, KMMatrix(cmat[:,1:k]))

    return mconfig
end

## Use triangle inequality to save on as many distance computations as we can
## (Keeps lower/upper bounds on distances)
## Avoids swapping row/columns of bounds matrices, instead keeps a vector of pointers
function pairwise_nn_savedist(config::Configuration, tgt_k::Int, data::Mat64, ::Type{A}) where {A<:Accelerator}
    @extract config : m k n centroids csizes
    @extract centroids : cmat=dmat
    if k < tgt_k
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    k == tgt_k && return Configuration{A}(data, centroids)

    # csizes′ = zeros(Int, k)
    # for i = 1:n
    #     csizes′[c[i]] += 1
    # end
    # @assert all(csizes′ .> 0)
    # @assert csizes == csizes′
    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    λ = zeros(k, k)
    ω = zeros(k, k)

    p = collect(1:k)

    vs = Threads.resize_nthreads!(Tuple{Float64,Int}[], (Inf, 0))

    @inbounds begin
        for j = 1:k
            z = csizes[j]
            centr = @view centroids[:,j]
            for j′ = 1:(j-1)
                z′ = csizes[j′]
                centr′ = @view centroids[:,j′]
                cost = _cost(centr, centr′)
                v′ = _merge_cost(cost, z, z′)
                d′ = √cost
                λ[j′,j] = d′
                ω[j′,j] = d′
                λ[j,j′] = d′
                ω[j,j′] = d′
                if v′ < nns_costs[j]
                    nns_costs[j], nns[j] = v′, j′
                end
                if v′ < nns_costs[j′]
                    nns_costs[j′], nns[j′] = v′, j
                end
            end
        end
        for j = 1:k
            z = csizes[j]
            zp = csizes[nns[j]]
            zm = z + zp
        end
    end

    to_be_updated = Set{Int}()
    did_cost = Set{Int}()
    lb = zeros(k)

    @inbounds while k > tgt_k
        # for j = 1:k
        #     v, d, x = Inf, Inf, 0
        #     for j′ = 1:k
        #         j′ == j && continue
        #         cost = @views _cost(cmat[:,j], cmat[:,j′])
        #         d′ = √cost
        #         v′ = _merge_cost(cost, csizes[j], csizes[j′])
        #         @assert λ[j′,j] ≤ d′ (j,j′,jm,js,k,d′,λ[j′,j],λ[j,j′])
        #         @assert ω[j′,j] ≥ d′
        #         if v′ < v
        #             v, d, x = v′, d′, j′
        #         end
        #     end
        #     @assert nns[j] == x
        #     @assert nns_costs[j] == v
        #     δd = csizes[x] / (csizes[x] + csizes[j]) * d
        #     @assert δds[j] == δd
        # end

        jm = argmin(@view(nns_costs[1:k]))
        js = nns[jm]
        @assert isapprox(nns_costs[js], nns_costs[jm]; atol=1e-20, rtol=√(eps(eltype(nns_costs)))) (nns_costs[js], nns_costs[jm])
        jm, js = jm < js ? (jm, js) : (js, jm)

        zm, zs = csizes[jm], csizes[js]

        if zm < zs
            if js == k # we can't assign jm to the cluster that we're about to remove, so we swap the data instead
                nns[jm], nns[js] = nns[js], nns[jm]
                nns_costs[jm], nns_costs[js] = nns_costs[js], nns_costs[jm]
                p[jm], p[js] = p[js], p[jm]
                zm, zs = zs, zm
                for l = 1:m
                    cmat[l,jm], cmat[l,js] = cmat[l,js], cmat[l,jm]
                end
            else
                jm, js = js, jm
                zm, zs = zs, zm
            end
        end
        zm′ = zm + zs

        # δd = zs / zm′ * √(zm′ / (zs * zm) * nns_costs[jm])
        # δd = √(zs / (zm * zm′) * nns_costs[jm])
        # @assert λ[p[jm],p[js]] == ω[p[jm],p[js]]
        δd = zs / zm′ * λ[p[jm],p[js]]

        # update centroid
        for l = 1:m
            cmat[l,jm] = (zm * cmat[l,jm] + zs * cmat[l,js]) / zm′
            cmat[l,js] = cmat[l,k]
        end

        if k-1 == tgt_k
            # we're done, skip the last update
            k -= 1
            break
        end

        # update csizes
        csizes[jm] = zm′
        csizes[js] = csizes[k]
        csizes[k] = 0

        # update nns
        nns[js], nns[k] = nns[k], 0
        nns_costs[js], nns_costs[k] = nns_costs[k], Inf

        # update rest
        p[js], p[k] = p[k], 0

        @assert p[jm] ≠ 0 k,jm,js,p,p[jm]

        empty!(to_be_updated)
        for j = 1:(k-1)
            j == jm && continue
            j′ = nns[j]
            (j′ == jm || j′ == js) && push!(to_be_updated, j)
            j′ == k && (nns[j] = js)
        end
        empty!(did_cost)

        # update merge-costs lower and upper bounds
        z = csizes[jm] # same as zm′
        um = Inf
        pjm = p[jm]
        for j′ = 1:k-1
            j′ == jm && continue
            z′ = csizes[j′]
            pj′ = p[j′]
            lb[j′] = _merge_cost((λ[pj′,pjm] - δd)^2, z, z′)
            u = _merge_cost((ω[pj′,pjm] + δd)^2, z, z′)
            if u < um
                um = u
            end
        end

        # tc, tx = _get_nns(vs, jm, k-1, cmat, csizes)
        # @assert lb[tx] ≤ tc lb[tx],tc

        ## Update the merged cluster's data
        centr = @view centroids[:,jm]
        v, x = Inf, 0
        for j′ = 1:k-1
            j′ == jm && continue
            l′ = lb[j′]
            pj′ = p[j′]
            if l′ ≤ max(um, nns_costs[j′])
                z′ = csizes[j′]
                centr′ = @view centroids[:,j′]
                cost = _cost(centr, centr′)
                v′ = _merge_cost(cost, z, z′)
                d = √cost
                λ[pj′,pjm] = d
                ω[pj′,pjm] = d
                λ[pjm,pj′] = d
                ω[pjm,pj′] = d
                if v′ < v
                    v, x = v′, j′
                end
                newbest′ = v′ < nns_costs[j′]
                tbu′ = j′ ∈ to_be_updated
                if newbest′ | tbu′
                    nns_costs[j′], nns[j′] = v′, jm
                    (newbest′ & tbu′) && delete!(to_be_updated, j′)
                    push!(did_cost, j′)
                end
            else
                λ[pj′,pjm] = max(λ[pj′,pjm] - δd, 0.0)
                λ[pjm,pj′] = max(λ[pjm,pj′] - δd, 0.0)
                ω[pj′,pjm] += δd
                ω[pjm,pj′] += δd
            end
        end
        nns_costs[jm], nns[jm] = v, x
        # @assert tx == x k,jm,js,tx,x,tc,v
        # @assert tc == v

        for j in to_be_updated
            # tc, tx = _get_nns(vs, j, k-1, cmat, csizes)

            z′ = csizes[j]
            pj = p[j]
            centr′ = @view centroids[:,j]
            if j ∉ did_cost
                cost = _cost(centr, centr′)
                v′ = _merge_cost(cost, z, z′)
                d = √cost
                λ[pj,pjm] = d
                ω[pj,pjm] = d
                λ[pjm,pj] = d
                ω[pjm,pj] = d
                x′ = jm
                old_v = nns_costs[j]
                nns_costs[j], nns[j] = v′, x′
                if v′ < old_v
                    # @assert tx == x′
                    # @assert tc == v′
                    continue
                end
            else
                v′, x′ = nns_costs[j], nns[j]
                # @assert x′ == jm
                # @assert λ[pj,pjm] == ω[pj,pjm]
            end
            # b2 = _merge_cost((ω[pj,pjm] + δd)^2, z, z′) # TODO?
            for j′ = 1:k-1 # TODO parallelize
                (j′ == jm || j′ == j) && continue
                z′′ = csizes[j′]
                pj′ = p[j′]
                cost = λ[pj′,pj]^2
                b1 = _merge_cost(cost, z′, z′′)
                b1 > v′ && continue
                if λ[pj′,pj] == ω[pj′,pj]
                    # centr′′ = @view centroids[:,j′]
                    # cost = _cost(centr′, centr′′)
                    # v′′ = _merge_cost(cost, z′, z′′)
                    # d′′ = √cost
                    v′′ = b1 # _merge_cost(cost, z′, z′′)
                    # @assert λ[pj′,pj] == √cost
                else
                    centr′′ = @view centroids[:,j′]
                    cost = _cost(centr′, centr′′)
                    v′′ = _merge_cost(cost, z′, z′′)
                    d = √cost
                    λ[pj′,pj] = d
                    ω[pj′,pj] = d
                    λ[pj,pj′] = d
                    ω[pj,pj′] = d
                end
                if v′′ < v′
                    v′, x′ = v′′, j′
                end
            end
            nns_costs[j], nns[j] = v′, x′

            # @assert nns_costs[j] == tc
            # @assert nns[j] == tx
        end

        # for j = 1:k-1
        #     v, d, x = Inf, Inf, 0
        #     for j′ = 1:k-1
        #         j′ == j && continue
        #         cost = @views _cost(cmat[:,j], cmat[:,j′])
        #         d′ = √cost
        #         v′ = _merge_cost(cost, csizes[j], csizes[j′])
        #         @assert λ[j′,j] ≤ d′ (j,j′,jm,js,k,d′,λ[j′,j],λ[j,j′])
        #         @assert ω[j′,j] ≥ d′
        #         if v′ < v
        #             v, d, x = v′, d′, j′
        #         end
        #     end
        #     @assert nns[j] == x
        #     @assert nns_costs[j] == v
        # end

        k -= 1
    end

    @assert all(csizes[1:k] .> 0)

    mconfig = Configuration{A}(data, KMMatrix(cmat[:,1:k]))

    return mconfig
end
