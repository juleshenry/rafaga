# Model definitions and conditional characteristic functions of ln VIX_T.
#
# Notation follows Bao (2013). All parameters are constant (the paper's
# chapter-7 calibration uses constant parameters per maturity). τ = T - t in
# years, V = spot VIX. cf(m, V, τ, s) = E_t[exp(i s ln VIX_T)].

abstract type VIXModel end

"""
    MRLR(κ, θ, σ)

d ln VIX = κ(θ - ln VIX) dt + σ dW          (Bao 2013, ch. 3)
"""
struct MRLR{T<:Real} <: VIXModel
    κ::T
    θ::T
    σ::T
end
MRLR(κ, θ, σ) = MRLR(promote(κ, θ, σ)...)

"""
    MRLRJ(κ, θ, σ, λ, η)

d ln VIX = κ(θ - ln VIX) dt + σ dW + J dN,   N ~ Poisson(λ), J ~ Exp(η)
(Bao 2013, ch. 4). The jump is not compensated, so η > 1 is needed for the
VIX future (the first moment of VIX_T) to be finite.
"""
struct MRLRJ{T<:Real} <: VIXModel
    κ::T
    θ::T
    σ::T
    λ::T
    η::T
end
MRLRJ(κ, θ, σ, λ, η) = MRLRJ(promote(κ, θ, σ, λ, η)...)

"""
    MRLRSV(κ, θ, ρ, κv, θv, σv, V0)

d ln VIX = κ(θ - ln VIX) dt + √V dW
dV       = κv(θv - V) dt + σv √V dZ,   d⟨W,Z⟩ = ρ dt      (Bao 2013, ch. 5)
"""
struct MRLRSV{T<:Real} <: VIXModel
    κ::T
    θ::T
    ρ::T
    κv::T
    θv::T
    σv::T
    V0::T
end
MRLRSV(κ, θ, ρ, κv, θv, σv, V0) = MRLRSV(promote(κ, θ, ρ, κv, θv, σv, V0)...)

paramnames(::Type{<:MRLR}) = (:κ, :θ, :σ)
paramnames(::Type{<:MRLRJ}) = (:κ, :θ, :σ, :λ, :η)
paramnames(::Type{<:MRLRSV}) = (:κ, :θ, :ρ, :κv, :θv, :σv, :V0)
paramnames(m::VIXModel) = paramnames(typeof(m))
params(m::VIXModel) = [getfield(m, p) for p in paramnames(m)]
modelname(m::VIXModel) = modelname(typeof(m))
modelname(::Type{<:MRLR}) = "MRLR"
modelname(::Type{<:MRLRJ}) = "MRLRJ"
modelname(::Type{<:MRLRSV}) = "MRLRSV"

"Return a copy of `m` with the long-term mean replaced by `θ`."
with_theta(m::MRLR, θ) = MRLR(m.κ, θ, m.σ)
with_theta(m::MRLRJ, θ) = MRLRJ(m.κ, θ, m.σ, m.λ, m.η)
with_theta(m::MRLRSV, θ) = MRLRSV(m.κ, θ, m.ρ, m.κv, m.θv, m.σv, m.V0)

# ---------------------------------------------------------------------------
# Characteristic functions
# ---------------------------------------------------------------------------

# Gaussian (OU) part shared by MRLR and MRLRJ: mean and variance of ln VIX_T.
function ou_moments(κ, θ, σ, V, τ)
    ϕ = exp(-κ * τ)
    μ = ϕ * log(V) + θ * (1 - ϕ)
    v = σ^2 * (1 - ϕ^2) / (2κ)
    return μ, v, ϕ
end

function cf(m::MRLR, V, τ, s::Number)
    μ, v, _ = ou_moments(m.κ, m.θ, m.σ, V, τ)
    return exp(im * s * μ - s^2 * v / 2)
end

# Jump contribution: λ ∫_0^τ (E[exp(i s e^{-κh} J)] - 1) dh
#                  = (λ/κ) log((η - i s ϕ) / (η - i s))
function cf(m::MRLRJ, V, τ, s::Number)
    μ, v, ϕ = ou_moments(m.κ, m.θ, m.σ, V, τ)
    jump = (m.λ / m.κ) * log((m.η - im * s * ϕ) / (m.η - im * s))
    return exp(im * s * μ - s^2 * v / 2 + jump)
end

cf(m::Union{MRLR,MRLRJ}, V, τ, s::AbstractVector) = [cf(m, V, τ, si) for si in s]

# MRLRSV: ψ = exp(a(τ) + b(τ) V0 + i s e^{-κτ} ln V), where with u(h) = i s e^{-κh}
#   b' = u²/2 + ρ σv u b + σv² b²/2 - κv b,     b(0) = 0
#   a  = i s θ (1 - e^{-κτ}) + κv θv ∫_0^τ b
# This is the Riccati system (5.4) of the paper, solved by RK4 as the paper
# recommends over the Kummer-function closed form.
function cf(m::MRLRSV, V, τ, s::AbstractVector; hscale = 1.0)
    T = float(real(eltype(s)))
    b = zeros(Complex{T}, length(s))
    I = zeros(Complex{T}, length(s))
    f(hh, bb, si) = begin
        u = im * si * exp(-m.κ * hh)
        u^2 / 2 + m.ρ * m.σv * u * bb + m.σv^2 * bb^2 / 2 - m.κv * bb
    end
    @inbounds for j in eachindex(s)
        si = s[j]
        # RK4 step: resolve the e^{-κh} decay, and keep h·|∂f/∂b| ≈ h·(σv|s| + κv)
        # inside the RK4 stability region (the Riccati is stiff for large |s|).
        h_target = hscale * min(0.1 / max(m.κ, 1), 1 / (m.σv * abs(si) + m.κv + 1))
        n = max(20, ceil(Int, τ / h_target))
        h = τ / n
        bj = zero(Complex{T})
        Ij = zero(Complex{T})
        for k in 0:n-1
            t0 = k * h
            k1 = f(t0, bj, si)
            k2 = f(t0 + h / 2, bj + h / 2 * k1, si)
            k3 = f(t0 + h / 2, bj + h / 2 * k2, si)
            k4 = f(t0 + h, bj + h * k3, si)
            Ij += h / 6 * (bj + 2(bj + h / 2 * k1) + 2(bj + h / 2 * k2) + (bj + h * k3))
            bj += h / 6 * (k1 + 2k2 + 2k3 + k4)
        end
        b[j] = bj
        I[j] = Ij
    end
    ϕ = exp(-m.κ * τ)
    return @. exp(im * s * (ϕ * log(V) + m.θ * (1 - ϕ)) + m.κv * m.θv * I + b * m.V0)
end

cf(m::MRLRSV, V, τ, s::Number) = cf(m, V, τ, [complex(s)])[1]
