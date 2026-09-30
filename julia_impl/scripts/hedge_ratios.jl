# Futures hedge ratios of VIX calls: MRLRJ (Oct-2026 fit) vs Black.
#
# In the MRLR family VIX_T = VIX_t^φ · G with G independent of VIX_t, so
# dC/dF = (dC/dV)/(dF/dV) = D·Π1 exactly. The Black delta D·N(d1) is taken at
# the Black vol that reproduces the same model price on the model future.
#
# Usage: julia --project=. scripts/hedge_ratios.jl

using VIXModels, Distributions, Printf

m = MRLRJ(9.29, 2.76, 0.353, 5.23, 3.29)          # 21-Oct-2026 fit, results/calibration_2026-09-29.md
V, τ = 16.04, 0.0595
D = exp(-0.0446τ)
h = 1e-3
F = vix_future(m, V, τ)
dF = (vix_future(m, V + h, τ) - vix_future(m, V - h, τ)) / 2h
println("| K | call | MRLRJ hedge ratio | Black delta | ratio |\n|---|---|---|---|---|")
for K in (16.0, 18.0, 20.0, 25.0, 30.0, 40.0)
    C = vix_call(m, V, τ, K; D)
    hj = (vix_call(m, V + h, τ, K; D) - vix_call(m, V - h, τ, K; D)) / 2h / dF
    σ = black_iv(C, F, K, τ; D)
    hb = D * cdf(Normal(), (log(F / K) + σ^2 * τ / 2) / (σ * sqrt(τ)))
    @printf("| %.0f | %.3f | %.3f | %.3f | %.2f |\n", K, C, hj, hb, hj / hb)
end
