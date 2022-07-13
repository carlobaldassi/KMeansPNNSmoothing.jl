"""
  SElk

The "simplified Elkan" method as described in Newling and Fleuret (PMLR 2016),
which is a simplified version of the method by Elkan (ICML 2003). Ofen outperformed
by [`RElk`](@ref).

Note: during kmeans iteration with the `verbose` option, the intermediate costs that
get printed when using this method are not accurate unless "[synched]" is printed too
(the output cost is always correct though).

See also: [`kmeans`](@ref), [`KMAccel`](@ref), [`RElk`](@ref).
"""
struct SElk <: Accelerator
    config::Configuration{SElk}
    δc::Vector{Float64}
    lb::Matrix{Float64}
    ub::Vector{Float64}
    stable::BitVector
    function SElk(config::Configuration)
        @extract config : n k
        δc = zeros(k)
        lb = zeros(k, n)
        ub = fill(Inf, n)
        stable = falses(k)
        return new(config, δc, lb, ub, stable)
    end
    function Base.copy(accel::SElk)
        @extract accel : config δc ls ub stable
        return new(accel.config, copy(δc), copy(lb), copy(ub), copy(stable))
    end
end

function reset!(accel::SElk)
    @extract accel : δc lb ub stable
    fill!(δc, 0.0)
    fill!(lb, 0.0)
    fill!(ub, Inf)
    fill!(stable, false)
    return accel
end

complete_initialization!(config::Configuration{SElk}, data::Mat64) = _complete_initialization_ub!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{SElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: ub lb stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH[SELK]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            v, x = findmin(costsij)
            costs[i], c[i] = v, x
            lb[:,i] .= .√̂(costsij)
            ub[i] = lb[x,i]
        end
    end
    num_chgd = n
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    DataLogging.@exec dist_comp = n * k

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{SElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C[SELK]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    DataLogging.@exec dist_comp = 0

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            ubi = ub[i]
            lbi = @view lb[:,i]

            skip = true
            for j = 1:k
                lbi[j] ≤ ubi && j ≠ ci && (skip = false; break)
            end
            skip && continue

            datai = @view data[:,i]
            @views v = _cost(datai, centroids[:,ci])
            DataLogging.@exec dist_comp += 1
            x = ci
            sv = √̂(v)
            ubi = sv
            lbi[ci] = sv

            for j in 1:k
                (lbi[j] > ubi || j == ci) && continue
                @views v′ = _cost(datai, centroids[:,j])
                DataLogging.@exec dist_comp += 1
                sv′ = √̂(v′)
                lbi[j] = sv′
                if v′ < v
                    v, x = v′, j
                    ubi = sv′
                end
            end
            if x ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x] = false
                    stable[ci] = false
                    csizes[x] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i], ub[i] = v, x, ubi
        end
    end
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

function sync_costs!(config::Configuration{SElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb ub
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "SYNC"
    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            @views v = _cost(data[:,i], centroids[:,ci])
            costs[i] = v
            ub[i] = √̂(v)
            lb[ci,i] = √̂(v)
        end
    end
    config.cost = sum(costs)
    DataLogging.@log "DONE time: $t cost: $(config.cost) dist_comp: $n"
    DataLogging.@pop_prefix!
    return config
end

function centroids_from_partition!(config::Configuration{SElk}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb ub stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with SElk accelerator method")

    DataLogging.@push_prefix! "C_FROM_P[SELK]"
    DataLogging.@exec dist_comp = 0

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
    DataLogging.@exec dist_comp += k - count(stable)

    @inbounds for i = 1:n
        ci = c[i]
        lbi = @view lb[:,i]
        @simd for j in 1:k
            lbi[j] -= δc[j]
        end
    end

    @inbounds for i = 1:n
        ci = c[i]
        ub[i] += δc[ci]
    end

    DataLogging.@log "dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end


"""
  RElk

The "reduced-comparison simplified Elkan" method. This is the same as [`SElk`](@ref) except that it
doesn't use the upper bound, but it compensates by using the same technique as in [`ReducedComparison`](@ref),
often resulting in a better or equal performance.

See also: [`kmeans`](@ref), [`KMAccel`](@ref), [`SElk`](@ref), [`ReducedComparison`](@ref).
"""
struct RElk <: Accelerator
    config::Configuration{RElk}
    δc::Vector{Float64}
    lb::Matrix{Float64}
    active::BitVector
    stable::BitVector
    function RElk(config::Configuration)
        @extract config : n k
        δc = zeros(k)
        lb = zeros(k, n)
        active = trues(k)
        stable = falses(k)
        return new(config, δc, lb, active, stable)
    end
    function Base.copy(accel::RElk)
        @extract accel : config δc lb active stable
        return new(accel.config, copy(δc), copy(lb), copy(active), copy(stable))
    end
end

function reset!(accel::RElk)
    @extract accel : δc lb active stable
    fill!(δc, 0.0)
    fill!(lb, 0.0)
    fill!(active, true)
    fill!(stable, false)
    return accel
end

complete_initialization!(config::Configuration{RElk}, data::Mat64) = _complete_initialization_none!(config, data)

function partition_from_centroids_from_scratch!(config::Configuration{RElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids accel
    @extract accel: lb active stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with RElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C_SCRATCH[RELK]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    costsij_th = [zeros(k) for th = 1:Threads.nthreads()]

    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            costsij = costsij_th[Threads.threadid()]
            _costs_1_vs_all!(costsij, data, i, centroids)
            costs[i], c[i] = findmin(costsij)
            lb[:,i] .= .√̂(costsij)
        end
    end
    num_chgd = n
    fill!(active, true)
    fill!(stable, false)
    cost = sum(costs)
    update_csizes!(config)

    DataLogging.@exec dist_comp = n * k

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return num_chgd
end

function partition_from_centroids!(config::Configuration{RElk}, data::Mat64, w::Union{Vector{<:Real},Nothing} = nothing)
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb active stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with RElk accelerator method")

    DataLogging.@push_prefix! "P_FROM_C[RELK]"
    DataLogging.@log "INPUTS k: $k n: $n m: $m"

    DataLogging.@exec dist_comp = 0

    num_fullsearch_th = zeros(Int, Threads.nthreads())

    active_inds = findall(active)
    all_inds = collect(1:k)

    num_chgd = 0
    fill!(stable, true)
    lk = Threads.SpinLock()
    t = @elapsed @bthreads for i in 1:n
        @inbounds begin
            ci = c[i]
            datai = @view data[:,i]
            if active[ci]
                old_v = costs[i]
                @views v = _cost(datai, centroids[:,ci])
                DataLogging.@exec dist_comp += 1
                fullsearch = (v > old_v)
            else
                v = costs[i]
                fullsearch = false
            end
            lbi = @view lb[:,i]
            ubi = √̂(v)
            if active[ci]
                lbi[ci] = ubi
            end
            num_fullsearch_th[Threads.threadid()] += fullsearch

            x = ci
            inds = fullsearch ? all_inds : active_inds
            for j in inds
                (lbi[j] > ubi || j == ci) && continue
                @views v′ = _cost(datai, centroids[:,j])
                DataLogging.@exec dist_comp += 1
                sv′ = √̂(v′)
                lbi[j] = sv′
                if v′ < v
                    v, x = v′, j
                    ubi = sv′
                end
            end
            if x ≠ ci
                @lock lk begin
                    num_chgd += 1
                    stable[x] = false
                    stable[ci] = false
                    csizes[x] += 1
                    csizes[ci] -= 1
                end
            end
            costs[i], c[i] = v, x
        end
    end
    num_fullsearch = sum(num_fullsearch_th)
    cost = sum(costs)

    config.cost = cost
    DataLogging.@log "DONE time: $t cost: $cost dist_comp: $dist_comp fullsearches: $num_fullsearch / $n"
    DataLogging.@pop_prefix!
    return num_chgd
end

function centroids_from_partition!(config::Configuration{RElk}, data::Mat64, w::Union{AbstractVector{<:Real},Nothing})
    @extract config: m k n c costs centroids csizes accel
    @extract accel: δc lb active stable
    @assert size(data) == (m, n)

    w ≡ nothing || error("w unsupported with RElk accelerator method")

    DataLogging.@push_prefix! "C_FROM_P[RELK]"
    DataLogging.@exec dist_comp = 0

    new_centroids = Cache.new_centroids(m, k)

    _sum_clustered_data!(new_centroids, data, c, stable)

    _update_centroids_δc!(centroids, new_centroids, δc, csizes, stable)
    DataLogging.@exec dist_comp += k - count(stable)

    active .= .~stable

    @inbounds for i = 1:n
        lbi = @view lb[:,i]
        @simd for j in 1:k
            lbi[j] -= δc[j]
        end
    end

    DataLogging.@log "dist_comp: $dist_comp"
    DataLogging.@pop_prefix!
    return config
end
