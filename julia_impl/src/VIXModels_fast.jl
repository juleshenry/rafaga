module VIXModelsFast

export MRLR, MRLRJ
export vix_future, vix_option

using Distributions, Statistics, SpecialFunctions, QuadGK

# Fast versions using Float64 instead of BigFloat for calibration speed
struct MRLR
    κ::Float64
    θ::Float64
    σ::Float64
    VIX0::Float64
end

function vix_future(m::MRLR, t::Float64, T::Float64, VIX_t::Float64)
    τ = T - t
    ϕ = exp(-m.κ * τ)
    M = exp(m.θ * (1 - ϕ) + (m.σ^2/(4*m.κ)) * (1 - exp(-2*m.κ*τ)))
    return VIX_t^ϕ * M
end

function vix_option(m::MRLR, t::Float64, T::Float64, K::Float64, VIX_t::Float64, r::Float64=0.0)
    τ = T - t
    F = vix_future(m, t, T, VIX_t)
    total_var = (m.σ^2/(2*m.κ)) * (1 - exp(-2*m.κ*τ))
    σ_eff = sqrt(total_var)
    d1 = (log(F/K) + 0.5*total_var) / σ_eff
    d2 = d1 - σ_eff
    dist = Normal(0.0, 1.0)
    return exp(-r*τ) * (F * cdf(dist, d1) - K * cdf(dist, d2))
end

struct MRLRJ
    κ::Float64
    θ::Float64
    σ::Float64
    λ::Float64
    η::Float64
    VIX0::Float64
end

function characteristic_function(m::MRLRJ, t::Float64, T::Float64, s::ComplexF64, VIX_t::Float64)
    τ = T - t
    ϕ = exp(-m.κ * τ)
    term1 = im * s * ϕ * log(VIX_t)
    term2 = im * s * m.θ * (1 - ϕ)
    term3 = -s^2 * m.σ^2 * (1 - exp(-2*m.κ*τ))/(4*m.κ)
    term4 = (m.λ/m.κ) * log((m.η - im*s*ϕ)/(m.η - im*s))
    return exp(term1 + term2 + term3 + term4)
end

function vix_future(m::MRLRJ, t::Float64, T::Float64, VIX_t::Float64)
    τ = T - t
    ϕ = exp(-m.κ * τ)
    M = exp(m.θ * (1 - ϕ) + (m.σ^2/(4*m.κ)) * (1 - exp(-2*m.κ*τ)) + 
            (m.λ/m.κ) * log((m.η - ϕ)/(m.η - 1)))
    return VIX_t^ϕ * M
end

function vix_option(m::MRLRJ, t::Float64, T::Float64, K::Float64, VIX_t::Float64, r::Float64=0.0)
    τ = T - t
    F = vix_future(m, t, T, VIX_t)
    
    ψ(s) = characteristic_function(m, t, T, ComplexF64(s), VIX_t)
    ψ1(s) = characteristic_function(m, t, T, ComplexF64(s) - im, VIX_t) / characteristic_function(m, t, T, ComplexF64(0.0, -1.0), VIX_t)
    ψ2(s) = ψ(s)
    
    function gil_pelaez(ψ_func, k)
        integrand(x) = imag(ψ_func(x) * exp(-im*x*k)) / x
        integral, err = quadgk(integrand, 1e-6, 50.0, rtol=1e-4)
        return 0.5 + (1/π) * integral
    end
    
    Π1 = gil_pelaez(ψ1, log(K))
    Π2 = gil_pelaez(ψ2, log(K))
    
    return exp(-r*τ) * (F * Π1 - K * Π2)
end

end # module
