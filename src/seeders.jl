abstract type Seeder end

abstract type MetaSeeder{S0<:Seeder} <: Seeder end


init_centroids(::Seeder, data::Mat64, k::Int, A::Type{<:Accelerator}; kw...) = error("not implemented")

## Load seeder files


const seeder_dir = joinpath(@__DIR__, "seeders")

macro include_seeder(filename)
    # modname = Symbol(replace(filename, r"\.jl$" => ""))
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

## Create a module to contain just the seeders
## to serve as a namespace that we can export

"""
The `KMSeed` module contains objects of type `Seeder` that can be passed
to `kmeans` to control the initialization process:

* `Unif`: sample centroids uniformly at random from the dataset, without replacement
* `PlusPlus`: kmeans++ by Arthur and Vassilvitskii (2006)
* `MaxMin`: furthest-point heuristic, by Katsavounidis et al. (1994)
* `Scala`: kmeans‖ also called "scalable kmeans++", by Bahmani et al. (2012)
* `PNN`: pairwise nearest-neighbor hierarchical clustering, by Equitz (1989)
* `PNNS`: the PNN-smoothing meta-method
* `Refine`: the refine meta-method by Bradely and Fayyad (1998)

See the documentation of each method for details.
"""
module KMSeed

import ..Unif,
       ..PlusPlus,
       ..AFKMC2,
       ..MaxMin,
       ..Scala,
       ..PNN,
       ..PNNS, ..PNNSR,
       ..Refine

end # module KMSeed


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
        return init == "++"    ? PlusPlus{ncandidates}() :
               init == "unif"   ? Unif() :
               init == "pnn"    ? PNN() :
               init == "maxmin" ? MaxMin() :
               init == "scala"  ? Scala(J, ϕ) :
               error("wat")
    elseif init == "pnns" && init0 == "self"
        @assert rlevel == 0
        return KMPNNSR(;ρ)
    else
        @assert rlevel ≥ 1
        kmseeder0 = init0 == "++"     ? PlusPlus{ncandidates}() :
                    init0 == "unif"   ? Unif() :
                    init0 == "pnn"    ? PNN() :
                    init0 == "maxmin" ? MaxMin() :
                    init0 == "scala"  ? Scala(J, ϕ) :
                    error("wat")

        return init == "pnns"   ? PNNS(kmseeder0; ρ, rlevel) :
               init == "refine" ? Refine(kmseeder0; J, rlevel) :
               error("wut")
    end
end
