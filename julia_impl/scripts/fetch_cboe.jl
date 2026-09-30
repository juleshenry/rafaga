# Save a snapshot of CBOE's delayed VIX option chain to data/cboe/.
# Run after the close (16:15 ET) to capture closing quotes.
#
# Usage: julia --project=. scripts/fetch_cboe.jl

using VIXModels, Dates, Downloads

raw = String(take!(Downloads.download(VIXModels.CBOE_VIX_URL, IOBuffer())))
q, spot, valuation = parse_cboe(raw)
dir = joinpath(@__DIR__, "..", "..", "data", "cboe")
mkpath(dir)
path = joinpath(dir, "VIX_options_$(Date(valuation)).json")
write(path, raw)
println("saved $(size(q, 1)) quotes, spot VIX $spot at $valuation -> $(normpath(path))")
