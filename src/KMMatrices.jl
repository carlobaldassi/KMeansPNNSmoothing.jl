module KMMatrices

using LinearAlgebra
using ExtractMacro

export KMMatrix, Mat64, update_quads!,
       _cost, _costs_1_vs_all!, _costs_1_vs_range!, _cost_1_vs_1,
       findmin_and_2ndmin

import Base: size, copy, copy!, convert, view, maybeview

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

Base.@propagate_inbounds function _cost(d1, d2)
    v1 = 0.0
    @simd for l = 1:length(d1)
        v1 += abs2(d1[l] - d2[l])
    end
    return v1
end

Base.@propagate_inbounds function _costs_1_vs_all!(ret::AbstractVector{T}, m1::KMMatrix{T}, i::Int, m2::KMMatrix{T}) where {T}
    @extract m1: d1=dviews[i] q1=dquads[i]
    @extract m2: d2=dmat q2=dquads
    mul!(ret, d2', d1)
    # ret .= q1 .+ q2 .- 2 .* ret
    @simd for j = 1:length(ret)
        ret[j] = q1 + q2[j] - 2 * ret[j]
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

end # module KMMatrices
