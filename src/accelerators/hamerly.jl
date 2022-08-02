"""
  Hamerly

The "simplified Hamerly" method as described in Newling and Fleuret (PMLR 2016),
which is a simplified version of the method by Hamerly (SDM 2010).

Note: during kmeans iteration with the `verbose` option, the intermediate costs that
get printed when using this method are not accurate unless "[synched]" is printed too
(the output cost is always correct though).

See also: [`kmeans`](@ref), [`KMAccel`](@ref), [`SHam`](@ref).
"""
struct Hamerly <: Accelerator
    config::Configuration{Hamerly}
    δc::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    s::Vector{Float64}
    stable::BitVector
    function Hamerly(config::Configuration{Hamerly})
        @extract config : n k
        centroids = config.centroids.dmat # XXX
        δc = zeros(k)
        lb = zeros(n)
        ub = fill(Inf, n)
        s = [@inbounds @views √(minimum(j′ ≠ j ? _cost(centroids[:,j], centroids[:,j′]) : Inf for j′ = 1:k)) for j = 1:k]
        stable = falses(k)
        return new(config, δc, lb, ub, s, stable)
    end
    function Base.copy(accel::Hamerly)
        @extract accel : config δc lb ub s stable
        return new(accel.config, copy(δc), copy(lb), copy(ub), copy(s), copy(stable))
    end
end

function reset!(accel::Hamerly)
    @extract accel : config δc lb ub s stable
    @extract config : k
    centroids = config.centroids.dmat
    fill!(δc, 0.0)
    fill!(lb, 0.0)
    fill!(ub, Inf)
    @inbounds for j = 1:k
        s[j] = @views √minimum(j′ ≠ j ? _cost(centroids[:,j], centroids[:,j′]) : Inf for j′ = 1:k)
    end
    fill!(stable, false)
    return accel
end

complete_initialization!(config::Configuration{Hamerly}, data::Mat64) = _complete_initialization_ub!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{Hamerly}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb ub stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly or Exponion accelerator method")

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    @bthreads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)
            costs[i], c[i] = v1, x1
            ub[i] = √̂(v1)
            lb[i] = √̂(v2)
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    return num_chgd
end

function partition_from_centroids!(config::Configuration{Hamerly}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub s stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly accelerator method")

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            lbi, ubi = lb[i], ub[i]
            hs = s[ci] / 2
            lbr = ifelse(lbi > hs, lbi, hs) # max(lbi, hs) # max accounts for NaN and signed zeros...
            lbr > ubi && continue
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            ub[i] = √̂(v)
            lbr > ub[i] && continue

            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)
            if x1 ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x1] = false
                    stable[ci] = false
                    csizes[x1] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i] = v1, x1
            ub[i], lb[i] = √̂(v1), √̂(v2)
        end
    end
    cost = sum(costs)

    config.cost = cost
    return num_chgd
end

sync_costs!(config::Configuration{Hamerly}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing) = _sync_costs_ub!(config, data, w)

function centroids_from_partition!(config::Configuration{Hamerly}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub s stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with Hamerly accelerator method")

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

    δcₘ, δcₛ, jₘ = 0.0, 0.0, 0
    @inbounds for j = 1:k
        δcj = δc[j]
        if δcj > δcₘ
            δcₛ = δcₘ
            δcₘ, jₘ = δcj, j
        elseif δcj > δcₛ
            δcₛ = δcj
        end
    end
    @inbounds for i = 1:n
        ci = c[i]
        lb[i] -= ifelse(ci == jₘ, δcₛ, δcₘ)
        ub[i] += δc[c[i]]
    end

    @inbounds for j = 1:k
        s[j] = Inf
        for j′ = 1:k
            j′ == j && continue
            @views cd = √̂(_cost(centroids[:,j′], centroids[:,j]))
            if cd < s[j]
                s[j] = cd
            end
        end
    end
    return config
end


"""
  SHam

Same as [`Hamerly`](@ref) but without the upper bound condition. It is often slightly worse and
provided mostly for reference. Its only tiny advantage is that the intermediate costs are always
accurate.

See also: [`kmeans`](@ref), [`KMAccel`](@ref), [`Hamerly`](@ref).

"""
struct SHam <: Accelerator
    config::Configuration{SHam}
    δc::Vector{Float64}
    lb::Vector{Float64}
    s::Vector{Float64}
    stable::BitVector
    function SHam(config::Configuration)
        @extract config : n k
        δc = zeros(k)
        lb = zeros(n)
        # s = [@inbounds @views √(minimum(j′ ≠ j ? _cost(centroids[:,j], centroids[:,j′]) : Inf for j′ = 1:k)) for j = 1:k]
        s = zeros(k)
        stable = falses(k)
        return new(config, δc, lb, s, stable)
    end
    function Base.copy(accel::SHam)
        @extract accel : config δc lb s stable
        return new(accel.config, copy(δc), copy(lb), copy(s), copy(stable))
    end
end

function reset!(accel::SHam)
    @extract accel : config δc lb s stable
    @extract config : k
    centroids = config.centroids.dmat # XXX
    fill!(δc, 0.0)
    fill!(lb, 0.0)
    @inbounds for j = 1:k
        s[j] = @views √minimum(j′ ≠ j ? _cost(centroids[:,j], centroids[:,j′]) : Inf for j′ = 1:k)
    end
    fill!(stable, false)
    return accel
end

complete_initialization!(config::Configuration{SHam}, data::Mat64) = _complete_initialization_none!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{SHam}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SHam accelerator method")

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    @bthreads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)
            costs[i], c[i] = v1, x1
            lb[i] = √̂(v2)
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    config.cost = cost
    return num_chgd
end

function partition_from_centroids!(config::Configuration{SHam}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb s stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SHam accelerator method")

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    num_chgd = 0
    lk = Threads.SpinLock()
    @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            lbi = lb[i]
            hs = s[ci] / 2
            lbr = ifelse(lbi > hs, lbi, hs) # max(lbi, hs) # max accounts for NaN and signed zeros...
            lbr > √̂(v) && continue

            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v1, v2, x1 = findmin_and_2ndmin(costsij)

            if x1 ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x1] = false
                    stable[ci] = false
                    csizes[x1] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i] = v1, x1
            lb[i] = √̂(v2)
        end
    end
    cost = sum(costs)

    config.cost = cost
    return num_chgd
end

function centroids_from_partition!(config::Configuration{SHam}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb s stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SHam accelerator method")

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)

    δcₘ, δcₛ, jₘ = 0.0, 0.0, 0
    @inbounds for j = 1:k
        δcj = δc[j]
        if δcj > δcₘ
            δcₛ = δcₘ
            δcₘ, jₘ = δcj, j
        elseif δcj > δcₛ
            δcₛ = δcj
        end
    end
    @inbounds for i = 1:n
        ci = c[i]
        lb[i] -= ifelse(ci == jₘ, δcₛ, δcₘ)
    end

    @inbounds for j = 1:k
        s[j] = Inf
        for j′ = 1:k
            j′ == j && continue
            @views cd = √̂(_cost(centroids[:,j′], centroids[:,j]))
            if cd < s[j]
                s[j] = cd
            end
        end
    end
    return config
end
