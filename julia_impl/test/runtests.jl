using Test
using VIXModels
using QuadGK
using Random
using Statistics

const V = 20.0
const τ = 0.25
const Ks = collect(12.0:2.0:40.0)

@testset "VIXModels" begin

    @testset "MRLR closed form == Fourier pricer" begin
        m = MRLR(11.05, 3.38, 1.97)                    # Table 7.4, Oct-2011
        @test vix_calls(m, V, τ, Ks) ≈ vix_calls_fourier(m, V, τ, Ks) atol = 1e-8
    end

    @testset "future == ψ(-i)" begin
        for m in (MRLR(5.0, 3.0, 1.2), MRLRJ(5.0, 3.0, 1.2, 3.0, 6.0),
                  MRLRSV(4.0, 3.0, 0.5, 1.7, 1.1, 1.0, 1.5))
            @test vix_future(m, V, τ) ≈ real(cf(m, V, τ, complex(0.0, -1.0))) rtol = 1e-8
        end
    end

    @testset "nested models reduce to MRLR" begin
        base = MRLR(5.0, 3.0, 1.2)
        @test vix_calls(MRLRJ(5.0, 3.0, 1.2, 1e-10, 6.0), V, τ, Ks) ≈
              vix_calls(base, V, τ, Ks) atol = 1e-8
        # σv → 0 with V0 = θv: variance of variance vanishes, √V = σ
        @test vix_calls(MRLRSV(5.0, 3.0, 0.3, 2.0, 1.2^2, 1e-8, 1.2^2), V, τ, Ks) ≈
              vix_calls(base, V, τ, Ks) atol = 1e-6
    end

    @testset "MRLRSV ODE step size converged" begin
        for m in (MRLRSV(4.27, 3.14, 1.00, 1.68, 1.11, 1.98, 1.81),   # Table 7.6
                  MRLRSV(4.0, 3.0, -0.5, 1.7, 1.1, 3.0, 0.5))
            for (Vx, τx) in ((42.3, 22 / 365), (16.0, 0.3))
                c = vix_calls(m, Vx, τx, Ks)
                cfine = vix_calls(m, Vx, τx, Ks; hscale = 0.1)
                @test c ≈ cfine atol = 1e-4
            end
        end
    end

    @testset "no-arbitrage shape" begin
        for m in (MRLRJ(20.0, 3.0, 1.5, 100.0, 9.0), MRLRSV(4.0, 3.1, 1.0, 1.7, 1.1, 2.0, 1.8))
            F = vix_future(m, V, τ)
            c = vix_calls(m, V, τ, Ks)
            @test all(diff(c) .< 0)                              # decreasing in K
            @test all(diff(diff(c)) .> -1e-10)                   # convex in K
            @test all(max.(F .- Ks, 0) .- 1e-8 .<= c .<= F)
        end
    end

    @testset "Monte Carlo agreement" begin
        rng = Xoshiro(42)
        for m in (MRLRJ(8.0, 2.9, 1.0, 10.0, 5.0), MRLRSV(4.0, 3.0, 0.6, 1.7, 1.1, 1.0, 1.5))
            X = simulate_terminal(m, V, τ, 200_000, 500; rng)
            @test mean(X) ≈ vix_future(m, V, τ) rtol = 0.01
            for K in (16.0, 22.0, 30.0)
                pay = max.(X .- K, 0)
                se = std(pay) / sqrt(length(pay))
                # 4 standard errors plus a small allowance for Euler bias
                @test abs(mean(pay) - vix_call(m, V, τ, K)) < 4se + 0.01
            end
        end
    end

    @testset "Float64 is enough (BigFloat cross-check)" begin
        m = MRLRJ(29.84, 3.0, 1.46, 169.45, 9.94)               # Table 7.5, Oct-2011
        Vb, τb = big(42.3), big(22) / 365
        mb = MRLRJ(big.(VIXModels.params(m))...)
        F = vix_future(mb, Vb, τb)
        for K in (30.0, 45.0, 70.0)
            k = log(big(K))
            I1 = quadgk(u -> imag(cis(-u * k) * cf(mb, Vb, τb, complex(u, -1))) / u,
                        big(0), big(Inf); rtol = big(1e-25))[1]
            I2 = quadgk(u -> imag(cis(-u * k) * cf(mb, Vb, τb, complex(u))) / u,
                        big(0), big(Inf); rtol = big(1e-25))[1]
            ref = F / 2 + I1 / π - K * (big(0.5) + I2 / π)
            @test vix_call(m, 42.3, 22 / 365, K) ≈ Float64(ref) atol = 1e-9
        end
    end

    @testset "Black implied vol round trip" begin
        for K in (15.0, 20.0, 30.0)
            c = black_call(21.0, K, τ, 0.9; D = 0.99)
            @test black_iv(c, 21.0, K, τ; D = 0.99) ≈ 0.9 atol = 1e-8
        end
        # MRLR implies a flat skew against the future (it is lognormal)
        m = MRLR(5.0, 3.0, 1.2)
        F = vix_future(m, V, τ)
        ivs = [black_iv(c, F, K, τ) for (c, K) in zip(vix_calls(m, V, τ, Ks), Ks)]
        @test maximum(ivs) - minimum(ivs) < 1e-6
    end

    @testset "θ anchoring reproduces the market future" begin
        s = OptionSlice(VIXModels.Date(2026, 10, 21), τ, 0.99, 22.5, Ks, Ks, Ks, Ks)
        for m in (MRLR(5.0, 0.0, 1.2), MRLRJ(5.0, 0.0, 1.2, 3.0, 6.0),
                  MRLRSV(4.0, 0.0, 0.5, 1.7, 1.1, 1.0, 1.5))
            @test vix_future(anchor(m, V, s), V, τ) ≈ 22.5 rtol = 1e-9
        end
    end
end
