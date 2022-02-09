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
    result = kmeans(a3, k, init="unif", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 7 < result.cost < 25
    @test result.exit_status == :converged
end

@testset "kmeans++" begin
    result = kmeans(a3, k, init="++", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="++", ncandidates=1, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans maxmin" begin
    result = kmeans(a3, k, init="maxmin", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNN" begin
    result = kmeans(a3, k, init="pnn", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans||" begin
    result = kmeans(a3, k, init="scala", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end


@testset "kmeans PNNS(UNIF)" begin
    result = kmeans(a3, k, init="smoothnn", init0="unif", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="unif", rlevel=2, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="unif", rlevel=3, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNS([G]KM++)" begin
    result = kmeans(a3, k, init="smoothnn", init0="++", ncandidates=1, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="++", ncandidates=1, rlevel=2, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="++", ncandidates=nothing, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="++", ncandidates=nothing, rlevel=2, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNS(MAXMIN)" begin
    result = kmeans(a3, k, init="smoothnn", init0="maxmin", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="maxmin", rlevel=2, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNS(SCALA)" begin
    result = kmeans(a3, k, init="smoothnn", init0="scala", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged

    result = kmeans(a3, k, init="smoothnn", init0="scala", rlevel=2, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans PNNSR" begin
    result = kmeans(a3, k, init="smoothnn", init0="self", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end



@testset "kmeans REFINE(UNIF)" begin
    result = kmeans(a3, k, init="refine", init0="unif", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans REFINE(++)" begin
    result = kmeans(a3, k, init="refine", init0="++", ncandidates=1, verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans REFINE(MAXMIN)" begin
    result = kmeans(a3, k, init="refine", init0="maxmin", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeans REFINE(SCALA)" begin
    result = kmeans(a3, k, init="refine", init0="scala", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end
