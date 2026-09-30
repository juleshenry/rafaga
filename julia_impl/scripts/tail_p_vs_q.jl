# Model-free comparison of option-implied (Q) and historical (P) tail
# probabilities for the 21-Oct-2026 VIX expiry.
#
#   Q(VIX_T > K): from the call-price slope, -dC/dK / D, using the mid quotes of
#                 the two neighbouring strikes (and, as a check, from the
#                 calibrated MRLRJ model).
#   P(VIX_T > K): share of historical days with VIX in [14, 18] (spot 16.04)
#                 on which the VIX closed above K 16 trading days later
#                 (29 Sep -> 21 Oct 2026 is 16 trading days).
#
# Usage: julia --project=. scripts/tail_p_vs_q.jl

using VIXModels, Dates, Printf, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
q, spot, valuation = parse_cboe(read(joinpath(ROOT, "data", "cboe", "VIX_options_2026-09-29.json"), String))
slices, r = build_slices(q, valuation; expiries = sort(unique(q.expiry[q.root.=="VIX"]))[1:1])
s = slices[1]
c = mid(s)
m = MRLRJ(9.29, 2.76, 0.353, 5.23, 3.29)                 # 21-Oct-2026 fit

lines = readlines(joinpath(ROOT, "data", "vix_historical.csv"))[4:end]
v = [parse(Float64, split(l, ",")[2]) for l in lines]
h = 16
start = findall(i -> 14 <= v[i] <= 18 && i + h <= length(v), eachindex(v))
@printf("historical starts with VIX in [14,18]: %d days\n", length(start))

println("| K | Q from call spread | Q from MRLRJ | P historical | Q/P |")
println("|---|---|---|---|---|")
for K in (20.0, 25.0, 30.0)
    i = findfirst(==(K), s.K)
    lo, hi = i - 1, i + 1
    qs = -(c[hi] - c[lo]) / (s.K[hi] - s.K[lo]) / s.D
    # model: Π2 = Q(VIX_T > K) from a digital built by a tight call spread
    ε = 0.01
    qm = -(vix_call(m, spot, s.τ, K + ε; D = s.D) - vix_call(m, spot, s.τ, K - ε; D = s.D)) / (2ε) / s.D
    p = mean(v[start.+h] .> K)
    @printf("| %.0f | %.3f | %.3f | %.3f | %.1f |\n", K, qs, qm, p, qs / p)
end

# Expected payoff of short-dated calls under history (P) vs the price received
# for selling them at the bid, discounted with the chain's rate.
println("\n| K | bid | ask | historical E[(VIX_T-K)+]·D | bid / P-value |")
println("|---|---|---|---|---|")
for K in (20.0, 25.0, 30.0, 40.0)
    i = findfirst(==(K), s.K)
    pv = s.D * mean(max.(v[start.+h] .- K, 0))
    @printf("| %.0f | %.2f | %.2f | %.3f | %.1f |\n", K, s.bid[i], s.ask[i], pv, s.bid[i] / pv)
end
