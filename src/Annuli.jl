module Annuli

using ExtractMacro

export SortedAnnuli, SimplerAnnuli, update!, get_inds

struct SortedAnnuli
    k::Int
    G::Int
    ws::Vector{BitSet}
    cws::Vector{BitSet} # cumulative ws
    inds::Vector{Int}
    es::Vector{Float64}
    dcache::Vector{Float64}
    icache::Vector{Int}
    jcache::Vector{Int}
    function SortedAnnuli(dist::AbstractVector{Float64}, iₓ::Int)
        @assert dist[iₓ] == 0.0
        k = length(dist) - 1
        G = ceil(Int, log2(k+2)-1) # ensures sum(2^f for f=1:G) == 2^(G+1)-2 ≥ k
        perm = sortperm(dist)
        @assert perm[1] == iₓ
        ws = [BitSet() for f = 1:G]
        cws = [BitSet() for f = 1:G]
        es = Vector{Float64}(undef, G)
        inds = zeros(Int, k+1)
        inds[iₓ] = typemin(Int) # sentinel
        i₀ = 0
        @inbounds for f = 1:G
            sz = 2^f
            i₁ = min(i₀+sz, k)
            for i in (i₀+1):i₁
                x = perm[i+1]
                push!(ws[f], x)
                inds[x] = f
            end
            es[f] = dist[perm[i₁]]
            i₀ = i₁
        end
        @assert i₀ == k
        copy!(cws[1], ws[1])
        for f = 2:G
            copy!(cws[f], cws[f-1])
            union!(cws[f], ws[f])
        end
        dcache = zeros(k)
        icache = zeros(Int, k)
        jcache = zeros(Int, k)
        return new(k, G, ws, cws, inds, es, dcache, icache, jcache)
    end
    function Base.copy(ann::SortedAnnuli)
        @extract ann : k G ws cws inds es
        dcache = zeros(k)
        icache = zeros(Int, k)
        jcache = zeros(Int, k)
        return new(k, G, copy.(ws), copy.(cws), copy(inds), copy(es), dcache, icache, jcache)
    end
end

function _check(ann::SortedAnnuli, dist::AbstractVector{Float64}, iₓ::Int)
    @extract ann : k G ws cws inds es
    @assert dist[iₓ] == 0.0
    @assert sum(2^f for f = 1:G) ≥ k
    for i = 1:k+1
        i == iₓ && continue
        @assert 1 ≤ inds[i] ≤ G
        @assert i ∈ ws[inds[i]]
    end
    @assert inds[iₓ] == typemin(Int)
    @assert collect(∪(ws...)) == deleteat!(collect(1:k+1), iₓ)
    @assert issorted(es)
    for f = 1:G
        if f < G
            @assert length(ws[f]) == 2^f
        else
            @assert length(ws[f]) == 2^G - (2^(G+1)-2-k)
        end
        for i in ws[f]
            @assert i ≠ iₓ
            @assert dist[i] ≤ es[f]
        end
        if f > 1
            for i in ws[f]
                @assert es[f-1] ≤ dist[i]
            end
        end
    end
    for i = 1:k+1
        i == iₓ && continue
        f = searchsortedfirst(es, dist[i])
        @assert inds[i] == f
    end
    for f = 1:G
        @assert cws[f] == union(ws[1:f]...)
    end
end

function update!(ann::SortedAnnuli, newdist::AbstractVector{Float64}, iₓ::Int, ignore::Union{BitVector,Vector{Bool},Nothing} = nothing)
    @extract ann : k G ws cws inds es dcache icache jcache
    @assert length(newdist) == k+1
    @assert newdist[iₓ] == 0.0
    @inbounds for i = 1:k+1
        i == iₓ && continue
        ignore ≢ nothing && ignore[i] && continue
        oldf = inds[i]
        ndi = newdist[i]
        # bet on not having changed first
        if (oldf == 1 || es[oldf-1] < ndi) && (oldf == G || es[oldf] ≥ ndi)
            if oldf == G && ndi > es[end]
                es[end] = ndi
            end
            continue
        end

        newf = searchsortedfirst(es, ndi)
        if newf > G
            @assert newf == G+1
            @assert ndi > es[end]
            es[end] = ndi
            newf = G
        end
        # @assert newf ≠ oldf
        # newf == oldf && continue
        delete!(ws[oldf], i)
        push!(ws[newf], i)
        inds[i] = newf
    end
    @inbounds for f = 1:(G-1)
        des_sz = 2^f
        sz = length(ws[f])
        sz == des_sz && continue
        if sz > des_sz
            resize!(dcache, sz)
            resize!(icache, sz)
            resize!(jcache, sz)
            for (j,i) in enumerate(ws[f])
                dcache[j] = newdist[i]
                jcache[j] = i
            end
            partialsortperm!(icache, dcache, 1:(sz-des_sz+1); rev=true)
            moveup = @view icache[1:sz-des_sz]
            for j in moveup
                i = jcache[j]
                delete!(ws[f], i)
                push!(ws[f+1], i)
                inds[i] = f+1
            end
            es[f] = dcache[icache[sz-des_sz+1]]
        else
            f′ = f+1
            add_sz = length(ws[f′])
            while sz + add_sz ≤ des_sz
                for i in ws[f′]
                    inds[i] = f
                end
                union!(ws[f], ws[f′])
                empty!(ws[f′])
                sz += add_sz
                f′ == G && @assert sz == des_sz
                f′ == G && break
                f′ += 1
                add_sz = length(ws[f′])
            end
            for f′′ = f:(f′-1)
                es[f′′] = es[f′-1]
            end
            if sz < des_sz
                resize!(dcache, add_sz)
                resize!(icache, add_sz)
                resize!(jcache, add_sz)
                for (j,i) in enumerate(ws[f′])
                    dcache[j] = newdist[i]
                    jcache[j] = i
                end
                movedown = partialsortperm!(icache, dcache, 1:(des_sz-sz))
                for j in movedown
                    i = jcache[j]
                    inds[i] = f
                    delete!(ws[f′], i)
                    push!(ws[f], i)
                end
                es[f] = dcache[movedown[end]]
            end
        end
    end
    copy!(cws[1], ws[1])
    for f = 2:G
        copy!(cws[f], cws[f-1])
        union!(cws[f], ws[f])
    end
    # _check(ann, newdist, iₓ)
    return ann
end

function get_inds(ann::SortedAnnuli, x::Float64)
    @extract ann : G es cws
    f = min(searchsortedfirst(es, x), G)
    return cws[f]
end

# function update_naive!(ann::SortedAnnuli, newdist::AbstractVector{Float64})
#     @extract ann : k G ws cws inds es
#     @assert length(newdist) == k
#     perm = sortperm(newdist)
#     foreach(empty!, ws)
#     i₀ = 0
#     @inbounds for f = 1:G
#         sz = 2^f
#         i₁ = min(i₀+sz, k)
#         for i in (i₀+1):i₁
#             x = perm[i]
#             push!(ws[f], x)
#             inds[x] = f
#         end
#         es[f] = newdist[perm[i₁]]
#         i₀ = i₁
#     end
#     @assert i₀ == k
#     copy!(cws[1], ws[1])
#     for f = 2:G
#         copy!(cws[f], union(cws[f-1], ws[f]))
#     end
#     return ann
# end



struct SimplerAnnuli
    k::Int
    G::Int
    es::Vector{Float64}
    pperm::Vector{Int}
    epos::Vector{Int}
    done::BitVector
    icache::Vector{Int}
    function SimplerAnnuli(k)
        G = ceil(Int, log2(k+2)-1) # ensures sum(2^f for f=1:G) == 2^(G+1)-2 ≥ k
        es = fill(Inf, G)
        pperm = zeros(Int, k)
        epos = zeros(Int, G)
        done = falses(k+1)
        icache = [1:k+1;]
        return new(k, G, es, pperm, epos, done, icache)
    end
    function Base.copy(ann::SimplerAnnuli)
        @extract ann : k G es pperm epos done icache
        return new(k, G, copy(es), copy(pperm), copy(epos), copy(done), copy(icache))
    end
end

function update!(ann::SimplerAnnuli, dist::AbstractVector{Float64}, iₓ::Int)
    @extract ann : k G es pperm epos done icache

    @assert length(dist) == k + 1
    @assert dist[iₓ] == 0.0
    dist[iₓ] = Inf # make it go to the end when sorting

    fill!(pperm, 0)
    fill!(done, false)
    done[iₓ] = true

    i = 0
    @inbounds for f = 1:G
        sz = 2^f
        i = min(i+sz, k)
        x = partialsortperm!(icache, dist, i)
        es[f] = dist[x]
        pperm[i] = x
        epos[f] = i
        done[x] = true
    end
    # @assert i == k
    # @show pperm
    # @show epos
    # @show es
    @inbounds for x = 1:(k+1)
        done[x] && continue
        f = searchsortedfirst(es, dist[x])
        # @assert 1 ≤ f ≤ G
        i = epos[f] - 1
        # @show ">>",x,f,i
        # @assert pperm[i] == 0 f,i,pperm
        pperm[i] = x
        epos[f] = i
        done[x] = true
        # @show pperm
        # @show epos
        # println("---")
    end
    dist[iₓ] = 0.0
    # i₀ = 0
    # for f = 1:G
    #     sz = 2^f
    #     i₁ = min(i₀+sz, k)
    #     sort!(@view pperm[(i₀+1):i₁])
    #     i₀ = i₁
    # end
    # _check(ann, dist, iₓ)
    return ann
end

function _check(ann::SimplerAnnuli, dist::AbstractVector{Float64}, iₓ::Int)
    @extract ann : k G es pperm epos done icache
    @assert dist[iₓ] == 0.0
    @assert sum(2^f for f = 1:G) ≥ k
    @assert issorted(es)
    @assert maximum(dist) == es[end]
    @assert all(done)
    @assert isperm([pperm; iₓ])
    @assert isperm(icache)
    i₀ = 0
    e₀ = 0.0
    for f = 1:G
        sz = 2^f
        i₁ = min(i₀+sz, k)
        e₁ = es[f]
        @assert all(e₀ .≤ dist[pperm[(i₀+1):i₁]] .≤ e₁)
        i₀ = i₁
    end
end

function get_inds(ann::SimplerAnnuli, x::Float64)
    @extract ann : k G es pperm
    f = min(searchsortedfirst(es, x), G)
    i₁ = min(2^(f+1)-2, k)
    return @view pperm[1:i₁]
end

end # module Annuli
