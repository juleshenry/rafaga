using Test
using VIXModels

@testset "VIXModels.jl" begin
    # Table 7.4 (Oct 2011) Parameters
    κ, θ, σ = 11.05, 3.38, 1.97
    VIX0 = 20.0
    
    mrlr = MRLR(κ, θ, σ, VIX0)
    
    T_expiry = 0.25
    r = 0.01
    K = 22.0
    
    future_price = vix_future(mrlr, 0.0, T_expiry, VIX0)
    @test future_price > 0
    
    call_price = vix_option(mrlr, 0.0, T_expiry, K, VIX0, r)
    @test call_price > 0
    
    # MRLRJ
    mrlrj = MRLRJ(κ, θ, σ, 1.0, 5.0, VIX0)
    future_price_j = vix_future(mrlrj, 0.0, T_expiry, VIX0)
    @test future_price_j > 0
    
    call_price_j = vix_option(mrlrj, 0.0, T_expiry, K, VIX0, r)
    @test call_price_j > 0
end
