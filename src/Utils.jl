module Utils

using LinearAlgebra
using ExtractMacro
using ..Cache

export KMMatrix, Mat64, update_quads!,
       Θ, ✓,
       @bthreads,
       _cost, _costs_1_vs_all!, _costs_1_vs_range!, _cost_1_vs_1,
       compute_costs_one!, compute_costs_one, update_costs_one!,
       findmin_and_2ndmin,
       _sum_clustered_data!, _update_centroids_δc!

import Base: size, copy, copy!, convert, view, maybeview

macro bthreads(ex)
    @assert ex.head == :for
    loop = ex.args[1]
    @assert loop.head == :(=)
    loop = Expr(:(=), esc.(loop.args)...)
    ex = Expr(:for, loop, esc.(ex.args[2:end])...)
    if Threads.nthreads() > 1
        return quote
            old_num_thr = BLAS.get_num_threads()
            old_num_thr ≠ 1 && BLAS.set_num_threads(1)
            try
                Threads.@threads :static $ex
            finally
                old_num_thr ≠ 1 && BLAS.set_num_threads(old_num_thr)
            end
        end
    else
        return ex
    end
end

const ColView{T} = SubArray{T, 1, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}

quad(v) = v ⋅ v

struct KMMatrix{T}
    dmat::Matrix{T}
    dquads::Vector{T}
    dviews::Vector{ColView{T}}
    function KMMatrix(dmat::Matrix{T}, dquads::Vector{T}) where {T}
        n = size(dmat, 2)
        dviews = [@view(dmat[:,i]) for i = 1:n]
        return new{T}(dmat, dquads, dviews)
    end
end

function update_quads!(kmat::KMMatrix{T}) where {T}
    @extract kmat: dviews dquads
    map!(quad, dquads, dviews)
    return kmat
end

function update_quads!(kmat::KMMatrix{T}, stable::AbstractVector{Bool}) where {T}
    @extract kmat: dviews dquads
    n = length(dquads)
    @assert length(stable) == n
    @inbounds for i = 1:n
        stable[i] && continue
        dquads[i] = quad(dviews[i])
    end
    return kmat
end


function KMMatrix(dmat::Matrix{T}) where {T}
    n = size(dmat, 2)
    dquads = Vector{T}(undef, n)
    kmat = KMMatrix(dmat, dquads)
    update_quads!(kmat)
    return kmat
end

function copy(kmat::KMMatrix{T}) where {T}
    @extract kmat: dmat dquads
    return KMMatrix(copy(dmat), copy(dquads))
end

function copy!(dest::KMMatrix{T}, src::KMMatrix{T}) where {T}
    size(dest) == size(src) || throw(ArgumentError("incompatible KMMatrix sizes dest=$(size(dest)) src=$(size(src))"))
    copy!(dest.dmat, src.dmat)
    copy!(dest.dquads, src.dquads)
    return dest
end

convert(::Type{Matrix{T}}, kmat::KMMatrix{T}) where {T} = kmat.dmat

size(kmat::KMMatrix, i...) = size(kmat.dmat, i...)

Base.@propagate_inbounds view(kmat::KMMatrix, ::Colon, i::Int) = kmat.dviews[i]
Base.@propagate_inbounds maybeview(kmat::KMMatrix, args...) = view(kmat, args...)

const Mat64 = KMMatrix{Float64}


## due to floating point approx, we may end up with tiny negative cost values
## we use this rectified square root for that
Θ(x) = ifelse(x > 0, x, 0.0)
✓(x) = √Θ(x)



## cost-related util functions

Base.@propagate_inbounds function _cost(d1, d2)
    v1 = 0.0
    @simd for l = 1:length(d1)
        v1 += abs2(d1[l] - d2[l])
    end
    return v1
end

Base.@propagate_inbounds function _costs_1_vs_all!(ret::AbstractVector{T}, m1::KMMatrix{T}, i::Int, m2::KMMatrix{T}, w::Nothing = nothing) where {T}
    @extract m1: d1=dviews[i] q1=dquads[i]
    @extract m2: d2=dmat q2=dquads
    mul!(ret, d2', d1)
    # ret .= q1 .+ q2 .- 2 .* ret
    @simd for j = 1:length(ret)
        ret[j] = q1 + q2[j] - 2 * ret[j]
    end
end

Base.@propagate_inbounds function _costs_1_vs_all!(ret::AbstractVector{T}, m1::KMMatrix{T}, i::Int, m2::KMMatrix{T}, w::AbstractVector{<:Real}) where {T}
    @extract m1: d1=dviews[i] q1=dquads[i]
    @extract m2: d2=dmat q2=dquads
    mul!(ret, d2', d1)
    @simd for j = 1:length(ret)
        ret[j] = w[j] * (q1 + q2[j] - 2 * ret[j])
    end
end

Base.@propagate_inbounds function _costs_1_vs_range!(ret::AbstractVector{T}, m1::KMMatrix{T}, i::Int, m2::KMMatrix{T}, r::UnitRange) where {T}
    @extract m1: d1=dviews[i] q1=dquads[i]
    @extract m2: d2=view(dmat,esc(:),esc(r)) q2=view(dquads,esc(r))
    rret = @view(ret[r])
    mul!(rret, d2', d1)
    # rret .= q1 .+ q2 .- 2 .* rret
    @simd for j = 1:length(rret)
        rret[j] = q1 + q2[j] - 2 * rret[j]
    end
end

Base.@propagate_inbounds function _cost_1_vs_1(m1::KMMatrix{T}, i::Int, m2::KMMatrix{T}, j::Int) where {T}
    @extract m1: d1=dviews[i] q1=dquads[i]
    @extract m2: d2=dviews[j] q2=dquads[j]
    n = length(d1)
    p = GC.@preserve d1 d2 BLAS.dot(n, pointer(d1, 1), 1, pointer(d2, 1), 1)
    return Θ(q1 + q2 - 2p)
end

function compute_costs_one!(costs::Vector{Float64}, data::Mat64, x::AbstractVector{Float64}, w::Nothing = nothing)
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    @bthreads for i = 1:n
        @inbounds costs[i] = _cost(@view(data[:,i]), x)
    end
    return costs
end

function compute_costs_one!(costs::Vector{Float64}, data::Mat64, x::AbstractVector{Float64}, w::AbstractVector{<:Real})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m
    @assert length(w) == n

    @bthreads for i = 1:n
        @inbounds costs[i] = w[i] * _cost(@view(data[:,i]), x)
    end
    return costs
end
compute_costs_one(data::Mat64, args...) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, args...)

function update_costs_one!(costs::Vector{Float64}, c::Vector{Int}, j::Int, data::Mat64, x::AbstractVector{Float64}, w::Nothing = nothing)
    m, n = size(data)
    @assert length(costs) == n
    @assert length(c) == n
    @assert length(x) == m

    @bthreads for i = 1:n
        @inbounds begin
            old_v = costs[i]
            new_v = _cost(@view(data[:,i]), x)
            if new_v < old_v
                costs[i] = new_v
                c[i] = j
            end
        end
    end
    return costs
end

function update_costs_one!(costs::Vector{Float64}, c::Vector{Int}, j::Int, data::Mat64, x::AbstractVector{Float64}, w::AbstractVector{<:Real})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(c) == n
    @assert length(x) == m

    @bthreads for i = 1:n
        @inbounds begin
            old_v = costs[i]
            new_v = w[i] * _cost(@view(data[:,i]), x)
            if new_v < old_v
                costs[i] = new_v
                c[i] = j
            end
        end
    end
    return costs
end


function findmin_and_2ndmin(a::AbstractVector{Float64})
    v1, v2, x1 = Inf, Inf, 0
    for (j,v) in enumerate(a)
        if v < v1
            v2 = v1
            v1, x1 = v, j
        elseif v < v2
            v2 = v
        end
    end
    return v1, v2, x1
end


## Auxiliary functions used in centroids_from_partition!

function _sum_clustered_data!(new_centroids, data, c, stable)
    m, k = size(new_centroids)
    @assert size(data, 1) == m
    n = size(data, 2)
    new_centroids_thr = Cache.new_centroids_thr(m, k)
    foreach(nc_thr->fill!(nc_thr, 0.0), new_centroids_thr)
    @bthreads for i = 1:n
        @inbounds begin
            j = c[i]
            stable[j] && continue
            id = Threads.threadid()
            ncj = @view new_centroids_thr[id][:,j]
            datai = @view data[:,i]
            @simd for l = 1:m
                ncj[l] += datai[l]
            end
        end
    end
    fill!(new_centroids, 0.0)
    for nc_thr in new_centroids_thr
        new_centroids .+= nc_thr
    end
    return new_centroids
end

function _sum_clustered_data!(new_centroids, zs, data, c, stable, w)
    m, k = size(new_centroids)
    @assert size(data, 1) == m
    n = size(data, 2)
    @assert length(zs) == k

    new_centroids_thr = Cache.new_centroids_thr(m, k)
    zs_thr = Cache.zs_thr(k)

    foreach(nc_thr->fill!(nc_thr, 0.0), new_centroids_thr)
    foreach(z->fill!(z, 0.0), zs_thr)
    @bthreads for i = 1:n
        @inbounds begin
            j = c[i]
            wi = w ≡ nothing ? 1 : w[i]
            stable ≢ nothing && stable[j] && continue
            id = Threads.threadid()
            ncj = @view new_centroids_thr[id][:,j]
            datai = @view data[:,i]
            @simd for l = 1:m
                ncj[l] += wi * datai[l]
            end
            zs_thr[id][j] += wi
        end
    end
    fill!(new_centroids, 0.0)
    for nc_thr in new_centroids_thr
        new_centroids .+= nc_thr
    end
    fill!(zs, 0.0)
    for zz in zs_thr
        zs .+= zz
    end
end

function _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
    m, k = size(centroids)
    fill!(δc, 0.0)
    @inbounds for j = 1:k
        stable[j] && continue
        z = csizes[j]
        z > 0 || continue
        centrj = @view centroids[:,j]
        ncentrj = @view new_centroids[:,j]
        ncentrj ./= z
        δc[j] = ✓(_cost(centrj, ncentrj))
        centrj[:] = ncentrj
    end
    update_quads!(centroids)
end


end # module Utils
