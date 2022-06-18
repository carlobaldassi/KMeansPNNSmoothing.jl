module Annuli

using ExtractMacro

export SortedAnnuli, update!

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
    function SortedAnnuli(dist::AbstractVector{Float64})
        k = length(dist)
        G = ceil(Int, log2(k+2)-1) # ensures sum(2^f for f=1:G) == 2^(G+1)-2 ≥ k
        perm = sortperm(dist)
        ws = [BitSet() for f = 1:G]
        cws = [BitSet() for f = 1:G]
        es = Vector{Float64}(undef, G)
        inds = zeros(Int, k)
        d₀ = 0.0
        i₀ = 0
        @inbounds for f = 1:G
            sz = 2^f
            i₁ = min(i₀+sz, k)
            for i in (i₀+1):i₁
                x = perm[i]
                push!(ws[f], x)
                inds[x] = f
            end
            es[f] = dist[perm[i₁]]
            i₀ = i₁
        end
        @assert i₀ == k
        cws[1] = copy(ws[1])
        for f = 2:G
            cws[f] = union(cws[f-1], ws[f])
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

function _check(ann::SortedAnnuli, dist::AbstractVector{Float64})
    @extract ann : k G ws cws inds es
    @assert sum(2^f for f = 1:G) ≥ k
    for i = 1:k
        @assert 1 ≤ inds[i] ≤ G
        @assert i ∈ ws[inds[i]]
    end
    @assert collect(∪(ws...)) == 1:k
    @assert issorted(es)
    for f = 1:G
        if f < G
            @assert length(ws[f]) == 2^f
        else
            @assert length(ws[f]) == 2^G - (2^(G+1)-2-k)
        end
        for i in ws[f]
            @assert dist[i] ≤ es[f]
        end
        if f > 1
            for i in ws[f]
                @assert es[f-1] ≤ dist[i]
            end
        end
    end
    for i = 1:k
        f = searchsortedfirst(es, dist[i])
        @assert inds[i] == f
    end
    for f = 1:G
        @assert cws[f] == union(ws[1:f]...)
    end
end

function update!(ann::SortedAnnuli, newdist::AbstractVector{Float64}, ignore::Union{BitVector,Nothing} = nothing)
    @extract ann : k G ws cws inds es dcache icache jcache
    @assert length(newdist) == k
    @inbounds for i = 1:k
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
    # _check(ann, newdist)
    return ann
end

function update_naive!(ann::SortedAnnuli, newdist::AbstractVector{Float64})
    @extract ann : k G ws cws inds es
    @assert length(newdist) == k
    perm = sortperm(newdist)
    foreach(empty!, ws)
    d₀ = 0.0
    i₀ = 0
    @inbounds for f = 1:G
        sz = 2^f
        i₁ = min(i₀+sz, k)
        for i in (i₀+1):i₁
            x = perm[i]
            push!(ws[f], x)
            inds[x] = f
        end
        es[f] = newdist[perm[i₁]]
        i₀ = i₁
    end
    @assert i₀ == k
    copy!(cws[1], ws[1])
    for f = 2:G
        copy!(cws[f], union(cws[f-1], ws[f]))
    end
    return ann
end

end # module Annuli
