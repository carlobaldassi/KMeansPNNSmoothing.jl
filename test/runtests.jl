using KMeansPNNSmoothing
using Test
using DelimitedFiles
using Random

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

Random.seed!(472632723)

@testset "kmeans uniform" begin
    kmseeder = KMeansPNNSmoothing.KMUnif()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 7 < result.cost < 25
    @test result.exit_status == :converged
end

@testset "kmeans++" begin
    kmseeder = KMeansPNNSmoothing.KMPlusPlus()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPlusPlus{1}()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans maxmin" begin
    kmseeder = KMeansPNNSmoothing.KMMaxMin()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNN" begin
    kmseeder = KMeansPNNSmoothing.KMPNN()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans||" begin
    kmseeder = KMeansPNNSmoothing.KMScala()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end


@testset "kmeans PNNS(UNIF)" begin
    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMUnif())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMUnif(); rlevel=2)
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMUnif(); rlevel=3)
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNS([G]KM++)" begin
    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMPlusPlus{1}())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMPlusPlus{1}(); rlevel=2)
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMPlusPlus())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMPlusPlus(); rlevel=2)
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNS(MAXMIN)" begin
    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMMaxMin())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMMaxMin(); rlevel=2)
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNS(SCALA)" begin
    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMScala())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    kmseeder = KMeansPNNSmoothing.KMPNNS(KMeansPNNSmoothing.KMScala(); rlevel=2)
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNSR" begin
    kmseeder = KMeansPNNSmoothing.KMPNNSR()
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end



@testset "kmeans REFINE(UNIF)" begin
    kmseeder = KMeansPNNSmoothing.KMRefine(KMeansPNNSmoothing.KMUnif())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans REFINE(++)" begin
    kmseeder = KMeansPNNSmoothing.KMRefine(KMeansPNNSmoothing.KMPlusPlus{1}())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans REFINE(MAXMIN)" begin
    kmseeder = KMeansPNNSmoothing.KMRefine(KMeansPNNSmoothing.KMMaxMin())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans REFINE(SCALA)" begin
    kmseeder = KMeansPNNSmoothing.KMRefine(KMeansPNNSmoothing.KMScala())
    result = kmeans(a3, k; kmseeder, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end
