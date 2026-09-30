# Calibration following Bao (2013), chapter 7:
#   * mid quotes of calls with positive bid and positive open interest,
#   * loss MMLSE = Σ (C_mod - C_mkt)² + α Σ (ln C_mod - ln C_mkt)²,  α = 8,
#   * fitting quality reported as PE (mean relative error) and MAE.

"One maturity of market VIX call quotes."
struct OptionSlice
    expiry::Date
    τ::Float64
    D::Float64          # discount factor e^{-rτ}
    F::Float64          # VIX future implied by put-call parity
    K::Vector{Float64}
    bid::Vector{Float64}
    ask::Vector{Float64}
    oi::Vector{Float64}
end

mid(s::OptionSlice) = (s.bid .+ s.ask) ./ 2

"Sub-slice keeping only the strikes selected by `idx`."
subslice(s::OptionSlice, idx) =
    OptionSlice(s.expiry, s.τ, s.D, s.F, s.K[idx], s.bid[idx], s.ask[idx], s.oi[idx])

function mmlse(model, market; α = 8.0)
    any(!isfinite, model) && return Inf
    lm = log.(max.(model, 1e-8))
    return sum(abs2, model .- market) + α * sum(abs2, lm .- log.(market))
end

"PE, MAE and share of model prices inside the bid-ask band."
function fit_stats(model, s::OptionSlice)
    c = mid(s)
    return (PE = mean(abs.(model .- c) ./ c),
            MAE = mean(abs.(model .- c)),
            inband = mean(s.bid .- 1e-9 .<= model .<= s.ask .+ 1e-9),
            n = length(c))
end

# Unconstrained parameterisations: positivity via exp, ρ via tanh, η > 1.
to_model(::Type{<:MRLR}, x) = MRLR(exp(x[1]), x[2], exp(x[3]))
to_model(::Type{<:MRLRJ}, x) = MRLRJ(exp(x[1]), x[2], exp(x[3]), exp(x[4]), 1 + exp(x[5]))
# MRLRSV parameters are box-bounded (logistic) to keep the optimiser out of
# regions with extreme vol-of-vol, where the Riccati ODE becomes very stiff.
# The bounds are 10-25x the paper's Table 7.6 values (2026 short-dated quotes
# need vol-of-vol far above the 2011 values); κ ≥ 0.1 and
# θ ∈ [1, 5] rule out the κ → 0, θ → -∞ ridge (only κθ is identified there).
bounded(x, hi) = hi / (1 + exp(-x))
unbounded(y, hi) = log(y / (hi - y))
to_model(::Type{<:MRLRSV}, x) =
    MRLRSV(0.1 + bounded(x[1], 50), 1 + bounded(x[2], 4), tanh(x[3]), bounded(x[4], 50),
           bounded(x[5], 50), bounded(x[6], 30), bounded(x[7], 30))

# Random starting points (log-uniform over wide, plausible ranges).
function random_start(::Type{<:MRLR}, θ0, rng)
    [log(rand(rng) * 30 + 0.5), θ0, log(rand(rng) * 3 + 0.2)]
end
function random_start(::Type{<:MRLRJ}, θ0, rng)
    [log(exp(rand(rng) * log(60)) + 0.5), θ0 - rand(rng) * 0.5, log(rand(rng) * 3 + 0.2),
     log(exp(rand(rng) * log(200)) + 0.1), log(rand(rng) * 15 + 0.5)]
end
function random_start(::Type{<:MRLRSV}, θ0, rng)
    logu(lo, hi) = exp(log(lo) + rand(rng) * log(hi / lo))
    [unbounded(logu(0.5, 20), 50), unbounded(θ0 - 1, 4), atanh(1.9rand(rng) - 0.95),
     unbounded(logu(0.2, 20), 50), unbounded(logu(0.1, 10), 50),
     unbounded(logu(0.3, 20), 30), unbounded(logu(0.1, 5), 30)]
end

"""
    anchor(m, V, s)

Replace θ so that the model VIX future matches the market (parity-implied)
future of slice `s`. θ enters ln F linearly with coefficient (1 - e^{-κτ}),
so this is exact: θ = (ln F_mkt - ln F_model(θ=0)) / (1 - e^{-κτ}).
This is the second calibration stage in Bao's Theorems 3.3/4.4/5.4.
"""
function anchor(m::VIXModel, V, s::OptionSlice)
    m0 = with_theta(m, zero(m.θ))
    F0 = vix_future(m0, V, s.τ)
    isfinite(F0) && F0 > 0 || return m
    return with_theta(m, (log(s.F) - log(F0)) / (1 - exp(-m.κ * s.τ)))
end

function slice_prices(m, V, s; anchored)
    mm = anchored ? anchor(m, V, s) : m
    return vix_calls(mm, V, s.τ, s.K; D = s.D), mm
end

"""
    calibrate(M, V, slices; anchored=false, nstarts=10, seed=1)

Fit model type `M` (MRLR, MRLRJ or MRLRSV) to the call quotes in `slices`
with spot VIX `V`.

* One slice, `anchored=false`: the paper's chapter-7 procedure — every
  parameter (including θ) free, spot VIX as the underlying.
* `anchored=true`: θ is set per slice so that the model reproduces the
  market VIX future; remaining parameters are fitted. With several slices this
  is a joint term-structure fit with one set of dynamics parameters.

Each Nelder-Mead run stops after `iterations` or `time_limit` seconds.
Returns `(models, loss)` where `models[i]` is the fitted model for slice `i`.
"""
function calibrate(::Type{M}, V, slices::Vector{OptionSlice};
                   anchored = false, nstarts = 10, seed = 1, iterations = 3000,
                   time_limit = 300.0) where {M<:VIXModel}
    !anchored && length(slices) > 1 &&
        error("joint calibration needs anchored=true (θ differs per maturity)")
    mkt = [mid(s) for s in slices]
    obj(x) = begin
        m = to_model(M, x)
        total = 0.0
        for (s, c) in zip(slices, mkt)
            p, _ = slice_prices(m, V, s; anchored)
            total += mmlse(p, c)
        end
        isfinite(total) ? total : 1e12
    end
    θ0 = log(slices[1].F)
    # independent multi-start: each start has its own RNG stream and runs on its own task
    runs = map(1:nstarts) do k
        Threads.@spawn begin
            x0 = random_start(M, θ0, Random.Xoshiro(seed + k))
            if obj(x0) < 1e12
                r = optimize(obj, x0, NelderMead(), Optim.Options(; iterations, time_limit))
                # one restart from the optimum helps Nelder-Mead escape collapsed simplices
                r = optimize(obj, Optim.minimizer(r), NelderMead(), Optim.Options(; iterations, time_limit))
                (Optim.minimizer(r), Optim.minimum(r))
            else
                (nothing, Inf)
            end
        end
    end
    best_x, best_f = argmin(last, fetch.(runs))
    best_x === nothing && error("no finite starting point found for $M")
    m = to_model(M, best_x)
    models = [anchored ? anchor(m, V, s) : m for s in slices]
    return models, best_f
end
