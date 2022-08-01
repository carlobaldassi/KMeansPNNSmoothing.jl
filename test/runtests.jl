using KMeansPNNSmoothing
using Test
using DelimitedFiles

# The a3 dataset was downloaded from here:
#
#  http://cs.uef.fi/sipu/datasets/
#
#  Clustering basic benchmark
#  P. Fränti and S. Sieranoja
#  K-means properties on six clustering benchmark datasets
#  Applied Intelligence, 48 (12), 4743-4759, December 2018
#  https://doi.org/10.1007/s10489-018-1238-7

a3 = Matrix(readdlm(joinpath(@__DIR__, "a3.txt"))')
a3 ./= maximum(a3)
n, m = size(a3)
k = 50

seed = 1029384756

all_accel = [
             KMAccel.ReducedComparison,
             KMAccel.Hamerly, KMAccel.SHam,
             KMAccel.SElk, KMAccel.RElk,
             KMAccel.Exponion,
             KMAccel.Yinyang, KMAccel.Ryy,
             KMAccel.Ball
            ]

function checkresult(result, status = :converged)
    return length(result.labels) == m &&
           all(∈(1:k), result.labels) &&
           size(result.centroids) == (2,k) &&
           result.exit_status == status
end

function naivematch(v1, v2)
    @assert length(v1) == length(v2)
    n = length(v1)

    k = maximum(v1)
    @assert maximum(v2) == k
    @assert sort(unique(v1)) == 1:k
    @assert sort(unique(v2)) == 1:k

    p = zeros(Int, k)
    for i in 1:n
        x1, x2 = v1[i], v2[i]
        if p[x1] == 0
            p[x1] = x2
        else
            @assert p[x1] == x2
        end
    end
    @assert sort(p) == 1:k
    return p[v1] == v2
end

function checkaccels(result, kmseeder)
    for accel in all_accel
        result_accel = kmeans(a3, k; kmseeder, seed, accel, verbose=false)
        @test checkresult(result_accel)
        @test naivematch(result_accel.labels, result.labels)
        @test result_accel.cost ≈ result.cost
    end
end

@testset "kmeans uniform" begin
    kmseeder = KMSeed.Unif()
    result = kmeans(a3, k; kmseeder, seed, accel=KMAccel.Naive, verbose=false)
    @test checkresult(result)
    @test 7 < result.cost < 25
    checkaccels(result, kmseeder)
end

@testset "kmeans++" begin
    kmseeder = KMSeed.PlusPlus()
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans PNN" begin
    kmseeder = KMSeed.PNN()
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans||" begin
    kmseeder = KMSeed.Scala()
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 15
    checkaccels(result, kmseeder)
end


@testset "kmeans PNNS(UNIF)" begin
    kmseeder = KMSeed.PNNS(KMSeed.Unif())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.Unif(); rlevel=2)
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.Unif(); rlevel=3)
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans PNNS([G]KM++)" begin
    kmseeder = KMSeed.PNNS(KMSeed.PlusPlus{1}())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.PlusPlus{1}(); rlevel=2)
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.PlusPlus())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.PlusPlus(); rlevel=2)
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans PNNS(MAXMIN)" begin
    kmseeder = KMSeed.PNNS(KMSeed.MaxMin())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.MaxMin(); rlevel=2)
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans PNNS(SCALA)" begin
    kmseeder = KMSeed.PNNS(KMSeed.Scala())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)

    kmseeder = KMSeed.PNNS(KMSeed.Scala(); rlevel=2)
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans PNNSR" begin
    kmseeder = KMSeed.PNNSR()
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end



@testset "kmeans REFINE(UNIF)" begin
    kmseeder = KMSeed.Refine(KMSeed.Unif())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans REFINE(++)" begin
    kmseeder = KMSeed.Refine(KMSeed.PlusPlus{1}())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans REFINE(MAXMIN)" begin
    kmseeder = KMSeed.Refine(KMSeed.MaxMin())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end

@testset "kmeans REFINE(SCALA)" begin
    kmseeder = KMSeed.Refine(KMSeed.Scala())
    result = kmeans(a3, k; kmseeder, seed, verbose=false)
    @test checkresult(result)
    @test 6.7 < result.cost < 11
    checkaccels(result, kmseeder)
end
