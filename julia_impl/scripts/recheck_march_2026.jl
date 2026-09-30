# Re-check the March 2026 calibration claimed in the original blog post
# ("sub-3% MAPE") with the thesis' procedure.
#
# Data: live_vix_calls.csv, VIXW calls expiring Wed 25 Mar 2026, scraped on the
# weekend of 7-8 Mar 2026; quotes are the Friday 6 Mar close (spot VIX 29.49).
# No puts were saved, so the future cannot be implied from parity; r = 4.5% as
# in the original script.
#
# Usage: julia -t auto --project=. scripts/recheck_march_2026.jl

using VIXModels, Dates, Printf, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
rows = [split(l, ",") for l in readlines(joinpath(ROOT, "live_vix_calls.csv"))[2:end]]
col(name) = findfirst(==(name), split(readline(joinpath(ROOT, "live_vix_calls.csv")), ","))
K = [parse(Float64, r[col("strike")]) for r in rows]
bid = [parse(Float64, r[col("bid")]) for r in rows]
ask = [parse(Float64, r[col("ask")]) for r in rows]
oi = [parse(Float64, r[col("openInterest")]) for r in rows]
spot = 29.49
valuation = DateTime(2026, 3, 6, 16, 15)
τ = year_fraction(valuation, Date(2026, 3, 25))
D = exp(-0.045τ)

keep = findall(i -> bid[i] > 0 && oi[i] > 0, eachindex(K))
@printf("%d calls, %d with positive bid and OI, strikes %.1f-%.1f, tau = %.4f\n",
        length(K), length(keep), minimum(K[keep]), maximum(K[keep]), τ)
# F is only used for the starting θ; spot is the paper's underlying anyway
s = OptionSlice(Date(2026, 3, 25), τ, D, spot, K[keep], bid[keep], ask[keep], oi[keep])

println("| Model | PE | MAE | in bid-ask | model future | parameters |")
println("|---|---|---|---|---|---|")
results = [Threads.@spawn begin
               (ms, _) = calibrate(M, spot, [s]; nstarts = M === MRLRSV ? 6 : 10)
               ms[1]
           end for M in (MRLR, MRLRJ, MRLRSV)]
for m in fetch.(results)
    st = fit_stats(vix_calls(m, spot, τ, s.K; D), s)
    @printf("| %s | %.2f%% | %.3f | %.0f%% | %.2f | %s |\n", modelname(m), 100st.PE, st.MAE,
            100st.inband, vix_future(m, spot, τ),
            join((@sprintf("%s=%.3g", n, v) for (n, v) in zip(paramnames(m), params(m))), ", "))
end
