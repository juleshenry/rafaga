# CBOE delayed-quote snapshots of the VIX option chain.
#
# Source: https://cdn.cboe.com/api/global/delayed_quotes/options/_VIX.json
# (free, 15-minute delayed; after the close it holds the closing quotes).

const CBOE_VIX_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_VIX.json"
const OCC_RE = r"^([A-Z]+)(\d{6})([CP])(\d{8})$"

"""
    parse_cboe(json_string) -> (quotes::DataFrame, spot, valuation::DateTime)

`valuation` is the time of the last VIX index print (US/Eastern).
"""
function parse_cboe(str::AbstractString)
    d = JSON3.read(str)[:data]
    rows = NamedTuple[]
    for o in d[:options]
        mt = match(OCC_RE, String(o[:option]))
        mt === nothing && continue
        root, ymd, cp, k = mt.captures
        push!(rows, (root = String(root),
                     expiry = Date(ymd, dateformat"yymmdd") + Year(2000),
                     type = cp == "C" ? "call" : "put",
                     strike = parse(Int, k) / 1000,
                     bid = Float64(o[:bid]), ask = Float64(o[:ask]),
                     last = Float64(something(o[:last_trade_price], NaN)),
                     volume = Float64(something(o[:volume], 0.0)),
                     open_interest = Float64(something(o[:open_interest], 0.0)),
                     cboe_iv = Float64(something(o[:iv], NaN))))
    end
    valuation = DateTime(String(d[:last_trade_time])[1:19])
    return DataFrame(rows), Float64(d[:current_price]), valuation
end

"Download the current CBOE VIX option chain."
fetch_cboe() = parse_cboe(String(take!(Downloads.download(CBOE_VIX_URL, IOBuffer()))))

"""
Year fraction from `valuation` to the VIX special opening quotation (09:30 ET)
on the expiry date, ACT/365.
"""
year_fraction(valuation::DateTime, expiry::Date) =
    Dates.value(DateTime(expiry) + Hour(9) + Minute(30) - valuation) / (1000 * 86400 * 365)

# Call/put mid pairs with positive bids on both sides.
function parity_pairs(q::DataFrame, expiry)
    c = filter(r -> r.expiry == expiry && r.type == "call" && r.bid > 0, q)
    p = filter(r -> r.expiry == expiry && r.type == "put" && r.bid > 0, q)
    j = innerjoin(select(c, :strike, [:bid, :ask] => ((b, a) -> (b .+ a) ./ 2) => :c),
                  select(p, :strike, [:bid, :ask] => ((b, a) -> (b .+ a) ./ 2) => :p),
                  on = :strike)
    return j.strike, j.c .- j.p
end

"""
    implied_forwards(q, expiries, τs) -> (F::Vector, r)

Put-call parity for VIX options holds against the VIX future of the same
expiry: C - P = e^{-rτ}(F - K). One rate `r` is fitted jointly across
expiries (least squares), and each F is then the least-squares level.
"""
function implied_forwards(q::DataFrame, expiries, τs)
    data = [parity_pairs(q, e) for e in expiries]
    fwd(r, (K, y), τ) = mean(y ./ exp(-r * τ) .+ K)
    sse(r) = sum(sum(abs2, y .- exp(-r * τ) .* (fwd(r, (K, y), τ) .- K))
                 for ((K, y), τ) in zip(data, τs))
    r = Optim.minimizer(optimize(sse, -0.05, 0.20))
    return [fwd(r, d, τ) for (d, τ) in zip(data, τs)], r
end

"""
    build_slices(q, valuation; expiries) -> (slices, r)

Apply the paper's filters (calls, positive bid, positive open interest) to the
listed expiries and attach the parity-implied future and discount factor.
"""
function build_slices(q::DataFrame, valuation::DateTime; expiries)
    τs = [year_fraction(valuation, e) for e in expiries]
    F, r = implied_forwards(q, expiries, τs)
    slices = OptionSlice[]
    for (e, τ, Fe) in zip(expiries, τs, F)
        c = sort(filter(x -> x.expiry == e && x.type == "call" && x.bid > 0 &&
                             x.open_interest > 0, q), :strike)
        push!(slices, OptionSlice(e, τ, exp(-r * τ), Fe, c.strike, c.bid, c.ask,
                                  c.open_interest))
    end
    return slices, r
end
