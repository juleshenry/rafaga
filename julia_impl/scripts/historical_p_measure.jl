# Real-world (P-measure) dynamics of ln VIX from daily closes, to compare with
# the option-implied (Q) parameters of calibrate_2026.jl.
#
#   * exact AR(1) discretisation of the OU process:
#       x_{t+1} = θ(1-β) + β x_t + ε,  β = e^{-κΔ},  Var ε = σ²(1-β²)/(2κ)
#   * frequency of large one-day upward moves in ln VIX
#
# Usage: julia --project=. scripts/historical_p_measure.jl

using Dates, Printf, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
lines = readlines(joinpath(ROOT, "data", "vix_historical.csv"))[4:end]   # 3 header rows
dates = [Date(split(l, ",")[1]) for l in lines]
x = [log(parse(Float64, split(l, ",")[2])) for l in lines]
Δ = 1 / 252
y, z = x[2:end], x[1:end-1]
β = cov(z, y) / var(z)
α = mean(y) - β * mean(z)
ε = y .- α .- β .* z
κ = -log(β) / Δ
θ = α / (1 - β)
σ = sqrt(var(ε) * 2κ / (1 - β^2))
years = Dates.value(dates[end] - dates[1]) / 365.25

@printf("sample %s to %s, %d daily closes (%.1f years)\n", dates[1], dates[end], length(x), years)
@printf("AR(1): beta = %.5f  ->  kappa = %.2f /yr, half-life = %.1f trading days\n", β, κ, log(2) / -log(β))
@printf("theta = %.3f (long-run VIX e^theta = %.1f), sigma = %.3f /sqrt(yr)\n", θ, exp(θ), σ)
@printf("sd of daily ln-change = %.4f\n", std(diff(x)))
for c in (0.15, 0.20, 0.30)
    n = count(>(c), diff(x))
    @printf("one-day ln-VIX rises > %.2f (VIX +%.0f%%): %d days, %.2f per year\n", c, 100(exp(c) - 1), n, n / years)
end
println("largest one-day rises:")
d = diff(x)
for i in sortperm(d; rev = true)[1:5]
    @printf("  %s  %.2f -> %.2f  (ln-change %.3f)\n", dates[i+1], exp(x[i]), exp(x[i+1]), d[i])
end
