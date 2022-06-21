module KMMatrices

using LinearAlgebra
using ExtractMacro

export KMMatrix, Mat64, update_quads!

import Base: size, copy, copy!, convert, view, maybeview

const minibatchsize = 128

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

end # module KMMatrices
