module VIXModels

export MeanRevertingLogModel, MRLR, MRLRJ, MRLRSV, MRLRSVJ
export simulate, vix_future, vix_option

using Distributions, Statistics, SpecialFunctions, QuadGK, Optim

abstract type MeanRevertingLogModel end

# ===========================================
# Model 1: MRLR (Pure Diffusion)
# ===========================================
struct MRLR <: MeanRevertingLogModel
    κ::BigFloat      # mean reversion speed
    θ::BigFloat      # long-term mean (log scale)
    σ::BigFloat      # volatility of volatility
    VIX0::BigFloat   # initial VIX level
end

# Constructor for standard Float64 that converts to BigFloat
MRLR(κ::Float64, θ::Float64, σ::Float64, VIX0::Float64) = 
    MRLR(BigFloat(κ), BigFloat(θ), BigFloat(σ), BigFloat(VIX0))

function simulate(m::MRLR, T::Float64, dt::Float64, n_paths::Int=1)
    n_steps = Int(round(T/dt))
    paths = zeros(BigFloat, n_paths, n_steps+1)
    paths[:, 1] .= log(m.VIX0)
    
    for i in 1:n_paths
        for t in 1:n_steps
            dW = randn() * sqrt(dt)
            dlnVIX = m.κ * (m.θ - paths[i, t]) * dt + m.σ * dW
            paths[i, t+1] = paths[i, t] + dlnVIX
        end
    end
    
    return exp.(paths)
end

function vix_future(m::MRLR, t::Float64, T::Float64, VIX_t::Float64)
    τ = BigFloat(T - t)
    VIX_t_bf = BigFloat(VIX_t)
    ϕ = exp(-m.κ * τ)
    M = exp(m.θ * (1 - ϕ) + (m.σ^2/(4*m.κ)) * (1 - exp(-2*m.κ*τ)))
    return Float64(VIX_t_bf^ϕ * M)
end

function vix_option(m::MRLR, t::Float64, T::Float64, K::Float64, VIX_t::Float64, r::Float64=0.0)
    τ = BigFloat(T - t)
    F = BigFloat(vix_future(m, t, T, VIX_t))
    K_bf = BigFloat(K)
    
    total_var = (m.σ^2/(2*m.κ)) * (1 - exp(-2*m.κ*τ))
    σ_eff = sqrt(total_var)
    
    d1 = (log(F/K_bf) + 0.5*total_var) / σ_eff
    d2 = d1 - σ_eff
    
    # Distributions.jl works with Float64, so we cast for the CDF
    dist = Normal(0.0, 1.0)
    call_val = exp(-BigFloat(r)*τ) * (F * BigFloat(cdf(dist, Float64(d1))) - K_bf * BigFloat(cdf(dist, Float64(d2))))
    return Float64(call_val)
end

# ===========================================
# Model 2: MRLRJ (Jump Diffusion)
# ===========================================
struct MRLRJ <: MeanRevertingLogModel
    κ::BigFloat      # mean reversion speed
    θ::BigFloat      # long-term mean (log scale)
    σ::BigFloat      # diffusion volatility
    λ::BigFloat      # jump intensity
    η::BigFloat      # exponential jump size parameter (η > 0)
    VIX0::BigFloat   # initial VIX level
end

MRLRJ(κ::Float64, θ::Float64, σ::Float64, λ::Float64, η::Float64, VIX0::Float64) = 
    MRLRJ(BigFloat(κ), BigFloat(θ), BigFloat(σ), BigFloat(λ), BigFloat(η), BigFloat(VIX0))

function characteristic_function(m::MRLRJ, t::Float64, T::Float64, s::ComplexF64, VIX_t::Float64)
    τ = BigFloat(T - t)
    ϕ = exp(-m.κ * τ)
    s_bf = Complex{BigFloat}(s)
    
    # Main components
    term1 = im * s_bf * ϕ * log(BigFloat(VIX_t))
    term2 = im * s_bf * m.θ * (1 - ϕ)
    term3 = -s_bf^2 * m.σ^2 * (1 - exp(-2*m.κ*τ))/(4*m.κ)
    term4 = (m.λ/m.κ) * log((m.η - im*s_bf*ϕ)/(m.η - im*s_bf))
    
    return ComplexF64(exp(term1 + term2 + term3 + term4))
end

function vix_future(m::MRLRJ, t::Float64, T::Float64, VIX_t::Float64)
    τ = BigFloat(T - t)
    ϕ = exp(-m.κ * τ)
    M = exp(m.θ * (1 - ϕ) + (m.σ^2/(4*m.κ)) * (1 - exp(-2*m.κ*τ)) + 
            (m.λ/m.κ) * log((m.η - ϕ)/(m.η - 1)))
    return Float64(BigFloat(VIX_t)^ϕ * M)
end

function vix_option(m::MRLRJ, t::Float64, T::Float64, K::Float64, VIX_t::Float64, r::Float64=0.0)
    τ = T - t
    F = vix_future(m, t, T, VIX_t)
    
    ψ(s) = characteristic_function(m, t, T, ComplexF64(s), VIX_t)
    ψ1(s) = characteristic_function(m, t, T, ComplexF64(s) - im, VIX_t) / characteristic_function(m, t, T, ComplexF64(0.0, -1.0), VIX_t)
    ψ2(s) = ψ(s)
    
    function gil_pelaez(ψ_func, k)
        integrand(x) = imag(ψ_func(x) * exp(-im*x*k)) / x
        integral, err = quadgk(integrand, 1e-8, 100.0, rtol=1e-6)
        return 0.5 + (1/π) * integral
    end
    
    Π1 = gil_pelaez(ψ1, log(K))
    Π2 = gil_pelaez(ψ2, log(K))
    
    return exp(-r*τ) * (F * Π1 - K * Π2)
end

end # module
