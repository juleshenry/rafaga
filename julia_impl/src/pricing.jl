# VIX futures and European VIX call prices.
#
# Calls are priced with the Heston-style formula of Bao (2013), Theorem 4.3:
#   C = D (F Π1 - K Π2),
#   Π2 = 1/2 + 1/π ∫_0^∞ Im(e^{-iuk} ψ(u)) / u du
#   Π1 = 1/2 + 1/π ∫_0^∞ Im(e^{-iuk} ψ(u - i) / ψ(-i)) / u du,   k = ln K
# where F = ψ(-i) = E[VIX_T] is the model VIX future and D the discount factor.
# Every strike of a maturity shares one characteristic-function grid.

const GL16 = gauss(16)

"""
    vix_future(m, V, τ)

Model VIX future price E_t[VIX_T] given spot VIX `V`.
"""
vix_future(m::VIXModel, V, τ) = real(cf(m, V, τ, complex(0.0, -1.0)))

function vix_future(m::MRLR, V, τ)
    μ, v, _ = ou_moments(m.κ, m.θ, m.σ, V, τ)
    return exp(μ + v / 2)
end

function vix_future(m::MRLRJ, V, τ)
    m.η > 1 || return Inf
    μ, v, ϕ = ou_moments(m.κ, m.θ, m.σ, V, τ)
    return exp(μ + v / 2 + (m.λ / m.κ) * log((m.η - ϕ) / (m.η - 1)))
end

# Upper integration limit: grow until the characteristic function is negligible.
# Capped at 400: |ψ(400)| > 1e-12 means sd(ln VIX_T) < ~2%, far from any VIX
# market, and only reached by degenerate parameters during calibration (e.g.
# σ → 0 in MRLRJ, where ψ stops decaying).
# MRLRSV uses 150 (sd > 5%) because each node costs an ODE solve whose step
# count grows with u.
cf_umax(::VIXModel) = 400.0
cf_umax(::MRLRSV) = 150.0
function cf_cutoff(m::VIXModel, V, τ; tol = 1e-12, umax = cf_umax(m))
    U = 8.0
    while U < umax && abs(cf(m, V, τ, U)) > tol
        U *= 1.5
    end
    return min(U, umax)
end

# Composite 16-point Gauss-Legendre nodes/weights on [0, U].
function fourier_grid(U)
    panels = max(16, ceil(Int, U / 3))
    x0, w0 = GL16
    edges = range(0, U; length = panels + 1)
    x = Float64[]
    w = Float64[]
    for p in 1:panels
        a, b = edges[p], edges[p+1]
        append!(x, (b - a) / 2 .* x0 .+ (a + b) / 2)
        append!(w, (b - a) / 2 .* w0)
    end
    return x, w
end

"""
    vix_calls(m, V, τ, K; D=1.0)

Model prices of VIX calls with strikes `K` (vector), spot VIX `V`, time to
expiry `τ` (years) and discount factor `D`.
"""
function vix_calls(m::VIXModel, V, τ, K::AbstractVector; D = 1.0, cfkw...)
    F = vix_future(m, V, τ)
    isfinite(F) || return fill(NaN, length(K))
    x, w = fourier_grid(cf_cutoff(m, V, τ))
    n = length(x)
    ψall = cf(m, V, τ, vcat(complex.(x), complex.(x, -1.0)); cfkw...)
    ψ = @view ψall[1:n]
    ψs = @view ψall[n+1:2n]
    prices = similar(K, Float64)
    for (j, Kj) in pairs(K)
        k = log(Kj)
        I1 = 0.0
        I2 = 0.0
        @inbounds for i in 1:n
            e = cis(-x[i] * k)
            I1 += w[i] * imag(e * ψs[i]) / x[i]
            I2 += w[i] * imag(e * ψ[i]) / x[i]
        end
        FΠ1 = F / 2 + I1 / π
        Π2 = 0.5 + I2 / π
        prices[j] = D * (FΠ1 - Kj * Π2)
    end
    return prices
end

# MRLR is lognormal: closed-form Black formula on the model future.
function vix_calls(m::MRLR, V, τ, K::AbstractVector; D = 1.0)
    F = vix_future(m, V, τ)
    _, v, _ = ou_moments(m.κ, m.θ, m.σ, V, τ)
    return [black_call_var(F, Kj, v, D) for Kj in K]
end

"Fourier pricer applied to any model (used to test the MRLR closed form)."
function vix_calls_fourier(m::MRLR, V, τ, K; D = 1.0)
    invoke(vix_calls, Tuple{VIXModel,Any,Any,AbstractVector}, m, V, τ, K; D = D)
end

vix_call(m::VIXModel, V, τ, K::Real; D = 1.0) = vix_calls(m, V, τ, [K]; D = D)[1]

# ---------------------------------------------------------------------------
# Black (1976) on the VIX future: eq. (7.11) of the paper, used to quote
# implied volatilities with the future (not spot VIX) as the underlying.
# ---------------------------------------------------------------------------

const N01 = Normal()

function black_call_var(F, K, v, D)
    v <= 0 && return D * max(F - K, 0.0)
    sv = sqrt(v)
    d1 = (log(F / K) + v / 2) / sv
    return D * (F * cdf(N01, d1) - K * cdf(N01, d1 - sv))
end

black_call(F, K, τ, σ; D = 1.0) = black_call_var(F, K, σ^2 * τ, D)

"""
    black_iv(C, F, K, τ; D=1.0)

Implied Black volatility of a VIX call using the VIX future `F` as underlying.
Returns NaN when `C` is outside the no-arbitrage bounds.
"""
function black_iv(C, F, K, τ; D = 1.0)
    lo, hi = 1e-4, 20.0
    (C <= black_call(F, K, τ, lo; D) || C >= black_call(F, K, τ, hi; D)) && return NaN
    for _ in 1:200
        mid = (lo + hi) / 2
        black_call(F, K, τ, mid; D) < C ? (lo = mid) : (hi = mid)
        hi - lo < 1e-10 && break
    end
    return (lo + hi) / 2
end

# ---------------------------------------------------------------------------
# Monte Carlo (Euler) simulation of VIX_T, used only for validation.
# ---------------------------------------------------------------------------

function simulate_terminal(m::MRLR, V, τ, npaths, nsteps; rng = Random.default_rng())
    dt = τ / nsteps
    x = fill(log(V), npaths)
    for _ in 1:nsteps, p in 1:npaths
        x[p] += m.κ * (m.θ - x[p]) * dt + m.σ * sqrt(dt) * randn(rng)
    end
    return exp.(x)
end

function simulate_terminal(m::MRLRJ, V, τ, npaths, nsteps; rng = Random.default_rng())
    dt = τ / nsteps
    x = fill(log(V), npaths)
    for _ in 1:nsteps, p in 1:npaths
        x[p] += m.κ * (m.θ - x[p]) * dt + m.σ * sqrt(dt) * randn(rng)
        for _ in 1:rand(rng, Poisson(m.λ * dt))
            x[p] += randexp(rng) / m.η
        end
    end
    return exp.(x)
end

function simulate_terminal(m::MRLRSV, V, τ, npaths, nsteps; rng = Random.default_rng())
    dt = τ / nsteps
    x = fill(log(V), npaths)
    v = fill(m.V0, npaths)
    ρ̄ = sqrt(1 - m.ρ^2)
    for _ in 1:nsteps, p in 1:npaths
        z1, z2 = randn(rng), randn(rng)
        vp = max(v[p], 0.0)                     # full truncation
        x[p] += m.κ * (m.θ - x[p]) * dt + sqrt(vp * dt) * z1
        v[p] += m.κv * (m.θv - vp) * dt + m.σv * sqrt(vp * dt) * (m.ρ * z1 + ρ̄ * z2)
    end
    return exp.(x)
end
