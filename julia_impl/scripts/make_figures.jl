# Build the paper's figures as plain TikZ (no pgfplots needed) from the
# per-strike fits written by calibrate_2026.jl.
#
# Usage: julia --project=. scripts/make_figures.jl [snapshot.json]

using VIXModels, Dates, Printf, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
snapshot = length(ARGS) > 0 ? ARGS[1] :
    last(sort(filter(endswith(".json"), readdir(joinpath(ROOT, "data", "cboe"); join = true))))
q, spot, valuation = parse_cboe(read(snapshot, String))
tag = Dates.format(Date(valuation), "yyyy-mm-dd")
expiries = sort(unique(q.expiry[q.root.=="VIX"]))[1:4]
slices, r = build_slices(q, valuation; expiries)

# curves CSV: model,expiry,strike,bid,ask,mid,fit
rows = [split(l, ",") for l in readlines(joinpath(ROOT, "results", "calibration_$(tag)_curves.csv"))[2:end]]
curve(model, e) = sort([(parse(Float64, x[3]), parse(Float64, x[7])) for x in rows
                        if x[1] == model && Date(x[2]) == e])

const COLORS = Dict("MRLR" => "black!55", "MRLRJ" => "red!75!black", "MRLRSV" => "blue!70!black")
const DASH = Dict("MRLR" => "densely dotted", "MRLRJ" => "solid", "MRLRSV" => "dashed")

"""
Implied Black volatility (VIX future as underlying) vs strike: market bid-ask
as vertical bars, the three calibrated models as lines.
"""
function skew_figure(s::OptionSlice; kmax, W = 7.0, H = 4.6)
    iv(c, K) = black_iv(c, s.F, K, s.τ; D = s.D)
    sel = findall(K -> K <= kmax, s.K)
    bars = [(s.K[i], iv(s.bid[i], s.K[i]), iv(s.ask[i], s.K[i])) for i in sel]
    lines = Dict(m => [(K, iv(c, K)) for (K, c) in curve(m, s.expiry) if K <= kmax]
                 for m in ("MRLR", "MRLRJ", "MRLRSV"))
    vals = vcat([b[2] for b in bars], [b[3] for b in bars],
                [p[2] for m in values(lines) for p in m])
    vals = filter(isfinite, vals)
    ylo, yhi = floor(quantile(vals, 0.0) * 10) / 10, ceil(quantile(vals, 1.0) * 10) / 10
    xlo, xhi = floor(minimum(s.K[sel]) / 5) * 5, kmax
    X(k) = (k - xlo) / (xhi - xlo) * W
    Y(v) = (v - ylo) / (yhi - ylo) * H
    io = IOBuffer()
    println(io, "\\begin{tikzpicture}[font=\\scriptsize]")
    @printf(io, "\\draw[black!40] (0,0) rectangle (%.3f,%.3f);\n", W, H)
    for k in xlo:(xhi - xlo > 40 ? 10 : 5):xhi
        @printf(io, "\\draw (%.3f,0) -- (%.3f,-0.08) node[below] {%d};\n", X(k), X(k), k)
    end
    ystep = (yhi - ylo) > 1.2 ? 0.4 : 0.2
    for v in ylo:ystep:yhi+1e-9
        @printf(io, "\\draw (0,%.3f) -- (-0.08,%.3f) node[left] {%.1f};\n", Y(v), Y(v), v)
    end
    @printf(io, "\\node at (%.3f,-0.55) {strike};\n", W / 2)
    @printf(io, "\\node[rotate=90] at (-0.75,%.3f) {Black implied vol.};\n", H / 2)
    @printf(io, "\\node[anchor=north west] at (0.05,%.3f) {\\textbf{%s} ($\\tau$=%.3f, $F$=%.2f)};\n",
            H - 0.02, Dates.format(s.expiry, "d u yyyy"), s.τ, s.F)
    println(io, "\\begin{scope}\\clip (0,0) rectangle ($W,$H);")
    for (k, lo, hi) in bars
        isfinite(lo) && isfinite(hi) || continue
        @printf(io, "\\draw[black!45, line width=1.6pt] (%.3f,%.3f) -- (%.3f,%.3f);\n", X(k), Y(lo), X(k), Y(hi))
    end
    for m in ("MRLR", "MRLRJ", "MRLRSV")
        pts = filter(p -> isfinite(p[2]), lines[m])
        isempty(pts) && continue
        print(io, "\\draw[$(COLORS[m]), $(DASH[m]), thick] ")
        println(io, join((@sprintf("(%.3f,%.3f)", X(k), Y(v)) for (k, v) in pts), " -- "), ";")
    end
    println(io, "\\end{scope}")
    # legend
    lx, ly = W - 2.1, 0.95
    @printf(io, "\\draw[black!45, line width=1.6pt] (%.3f,%.3f) -- (%.3f,%.3f) node[right, black] {bid--ask};\n",
            lx + 0.2, ly + 0.02, lx + 0.2, ly + 0.2)
    for (j, m) in enumerate(("MRLR", "MRLRJ", "MRLRSV"))
        y = ly - 0.28j + 0.1
        @printf(io, "\\draw[%s, %s, thick] (%.3f,%.3f) -- (%.3f,%.3f) node[right, black] {%s};\n",
                COLORS[m], DASH[m], lx, y, lx + 0.4, y, m)
    end
    println(io, "\\end{tikzpicture}")
    return String(take!(io))
end

outdir = joinpath(ROOT, "paper", "figures")
mkpath(outdir)
for (s, kmax) in ((slices[1], 50.0), (slices[4], 60.0))
    path = joinpath(outdir, "skew_$(s.expiry).tex")
    write(path, skew_figure(s; kmax))
    println("wrote ", relpath(path, ROOT))
end
