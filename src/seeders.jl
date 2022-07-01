"""
A `KMeansSeeder` object is used to specify the seeding algorithm.
Currently the following basic objects are defined:

* `KMUnif`: sample centroids uniformly at random from the dataset, without replacement
* `KMPlusPlus`: kmeans++ by Arthur and Vassilvitskii (2006)
* `KMMaxMin`: furthest-point heuristic, by Katsavounidis et al. (1994)
* `KMScala`: kmeans‖ also called "scalable kmeans++", by Bahmani et al. (2012)
* `KMPNN`: pairwise nearest-neighbor hierarchical clustering, by Equitz (1989)

There are also meta-methods, whose parent type is `KMMetaSeeder`, 

* `KMPNNS`: the PNN-smoothing meta-method
* `KMRefine`: the refine meta-method by Bradely and Fayyad (1998)

For each of these object, there is a corresponding implementation of
`init_centroids(::KMeansSeeder, data, k)`, which concretely performs the initialization.

The documentation of each method explains the arguments that can be passed to
control the initialization process.
"""
abstract type KMeansSeeder end

"""
A `KMMetaSeeder{S0<:KMeansSeeder}` object is a sub-type of `KMeansSeeder` representing a meta-method,
using an object of type `S0` as an internal seeder.
"""
abstract type KMMetaSeeder{S0<:KMeansSeeder} <: KMeansSeeder end



init_centroids(::KMeansSeeder, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) = error("not implemented")

## Load seeder files


const seeder_dir = joinpath(@__DIR__, "seeders")

macro include_seeder(filename)
    modname = Symbol(replace(filename, r"\.jl$" => ""))
    quote
        include(joinpath(seeder_dir, $(esc(filename))))
        # using .$modname
    end
end

@include_seeder "unif.jl"
@include_seeder "kmplusplus.jl"
@include_seeder "afkmc2.jl"
@include_seeder "maxmin.jl"
@include_seeder "scala.jl"
@include_seeder "pnn.jl"
@include_seeder "pnns.jl"
@include_seeder "refine.jl"


function gen_seeder(
        init::AbstractString = "pnns"
        ;
        init0::AbstractString = "",
        ρ::Float64 = 0.5,
        ncandidates::Union{Nothing,Int} = nothing,
        J::Int = 10,
        rlevel::Int = 1,
        rounds::Int = 5,
        ϕ::Float64 = 2.0,
    )
    all_basic_methods = ["++", "unif", "pnn", "maxmin", "scala"]
    all_rec_methods = ["refine", "pnns"]
    all_methods = [all_basic_methods; all_rec_methods]
    init ∈ all_methods || throw(ArgumentError("init should either be a matrix or one of: $all_methods"))
    if init ∈ all_rec_methods
        if init0 == ""
            init0="++"
            ncandidates ≡ nothing && (ncandidates = 1)
        end
        if init0 ∈ all_basic_methods
            init == "pnns" && rlevel < 1 && throw(ArgumentError("when init=$init and init0=$init0 rlevel must be ≥ 1"))
        elseif init0 == "self"
            init == "pnns" || throw(ArgumentError("init0=$init0 unsupported with init=$init"))
            rlevel = 0
        else
            throw(ArgumentError("when init=$init, init0 should be \"self\" or one of: $all_basic_methods"))
        end
    else
        init0 == "" || @warn("Ignoring init0=$init0 with init=$init")
    end

    if init ∈ all_basic_methods
        return init == "++"    ? KMPlusPlus{ncandidates}() :
               init == "unif"   ? KMUnif() :
               init == "pnn"    ? KMPNN() :
               init == "maxmin" ? KMMaxMin() :
               init == "scala"  ? KMScala(J, ϕ) :
               error("wat")
    elseif init == "pnns" && init0 == "self"
        @assert rlevel == 0
        return KMPNNSR(;ρ)
    else
        @assert rlevel ≥ 1
        kmseeder0 = init0 == "++"     ? KMPlusPlus{ncandidates}() :
                    init0 == "unif"   ? KMUnif() :
                    init0 == "pnn"    ? KMPNN() :
                    init0 == "maxmin" ? KMMaxMin() :
                    init0 == "scala"  ? KMScala(J, ϕ) :
                    error("wat")

        return init == "pnns"   ? KMPNNS(kmseeder0; ρ, rlevel) :
               init == "refine" ? KMRefine(kmseeder0; J, rlevel) :
               error("wut")
    end
end
