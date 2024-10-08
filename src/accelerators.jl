## generic falbacks and reference implementations
_complete_initialization_none!(config::Configuration, data::Mat64) = config

function _complete_initialization_ub!(config::Configuration, data::Mat64)
    @extract config: n costs accel
    @extract accel: ub

    @inbounds @simd for i = 1:n
        ub[i] = ✓(costs[i])
    end

    return config
end

reset!(accel::Accelerator) = error("not implemented")

partition_from_centroids_from_scratch!(config::Configuration, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = error("not implemented")
partition_from_centroids!(config::Configuration, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = error("not implemented")

sync_costs!(config::Configuration, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = config

function _sync_costs_ub!(config::Configuration, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: ub
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly, Yinyang or Exponion accelerator method")

    @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            ub[i] = ✓(v)
        end
    end
    config.cost = sum(costs)
    return config
end

## Load accelerator files

const accel_dir = joinpath(@__DIR__, "accelerators")

macro include_accel(filename)
    quote
        include(joinpath(accel_dir, $(esc(filename))))
    end
end

@include_accel "naive.jl"
@include_accel "reduced_comparison.jl"
@include_accel "hamerly.jl"
@include_accel "elkan.jl"
@include_accel "exponion.jl"
@include_accel "yinyang.jl"
@include_accel "ball.jl"

## Create a module to contain just the accelerators
## to serve as a namespace that we can export

"""
The `KMAccel` module contains subtypes of `Accelerator` that can be passed
to [`kmeans`](@ref) to specify the method used to accelerate Lloyd's algorithm.

* [`Naive`](@ref): the standard method (only for reference, should be avoided)
* [`ReducedComparison`](@ref): method by Kaukoranta et al. (1999)
* [`Hamerly`](@ref) and [`SHam`](@ref): methods based on Hamerly (2010)
* [`SElk`](@ref) and [`RElk`](@ref): methods based on Elkan (2003)
* [`Exponion`](@ref): method from Newling and Fleuret (2016)
* [`Yinyang`](@ref) and [`Ryy`](@ref): methods based on Ding et al. (2015)
* [`Ball`](@ref): method from Xia et al. (2018)

See the documentation of each type for details.
"""
module KMAccel

import ..Naive,
       ..ReducedComparison,
       ..Hamerly, ..SHam,
       ..SElk, ..RElk,
       ..Exponion,
       ..Yinyang, ..Ryy,
       ..Ball

end # module KMAccel
