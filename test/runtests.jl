using KMeansNNPP
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
end

@testset "kmeans++NN" begin
    result = kmeans(a3, k, init="++nn", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end

@testset "kmeansNN" begin
    result = kmeans(a3, k, init="nn", verbose=false)
    @test length(result.labels) == m
    @test all(∈(1:k), result.labels)
    @test size(result.centroids) == (2,k)
    @test 6.7 < result.cost < 11
    @test result.exit_status == :converged
end
