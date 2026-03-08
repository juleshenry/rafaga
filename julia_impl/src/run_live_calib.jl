using Pkg
Pkg.activate(".")

include("VIXModels_fast.jl")
using .VIXModelsFast
using CSV, DataFrames, Optim, Statistics, Printf, Dates

# Load the live options data we just scraped
df = CSV.read("../../live_vix_calls.csv", DataFrame)

# Calculate Mid price (our market price to fit) if not already done, just to be sure
df.Mid = (df.bid .+ df.ask) ./ 2.0
filter!(row -> row.Mid > 0.0, df)

# We are at 2026-03-08, target is 2026-03-25 -> 17 days
days_to_exp = 17.0
T = days_to_exp / 365.0

# Spot VIX
meta = readlines("../../live_vix_meta.txt")[1]
target_exp, spot_vix_str = split(meta, ",")
VIX_t = parse(Float64, spot_vix_str)

r = 0.045 # roughly 4.5% risk free rate right now

strikes = df.strike
market_prices = df.Mid

println("Calibrating on $(length(strikes)) strikes for expiration $target_exp")
println("Spot VIX: $VIX_t, T: $T")

# ---- MRLR Calibration (Baseline) ----
function obj_MRLR(params)
    κ = abs(params[1])
    θ = params[2]
    σ = abs(params[3])
    if κ <= 0.01 || σ <= 0.01; return 1e6; end
    
    model = MRLR(κ, θ, σ, VIX_t)
    err = 0.0
    for i in 1:length(strikes)
        try
            price = vix_option(model, 0.0, T, strikes[i], VIX_t, r)
            err += abs(price - market_prices[i]) / market_prices[i]
        catch; return 1e6; end
    end
    return err / length(strikes)
end

res_mrlr = optimize(obj_MRLR, [5.0, log(VIX_t), 1.0], NelderMead(), Optim.Options(iterations=500))
pe_mrlr = Optim.minimum(res_mrlr)

# ---- MRLRJ Calibration (Jump Diffusion) ----
function obj_MRLRJ(params)
    κ = abs(params[1])
    θ = params[2]
    σ = abs(params[3])
    λ = abs(params[4])
    η = abs(params[5])
    
    # constraint
    if η <= 1.01 || κ <= 0.01 || σ <= 0.01 || λ <= 0.01; return 1e6; end
    
    model = MRLRJ(κ, θ, σ, λ, η, VIX_t)
    err = 0.0
    for i in 1:length(strikes)
        try
            price = vix_option(model, 0.0, T, strikes[i], VIX_t, r)
            if isnan(price) || isinf(price)
                return 1e6
            end
            err += abs(price - market_prices[i]) / market_prices[i]
        catch
            return 1e6
        end
    end
    return err / length(strikes)
end

init_guess_jump = [1.5, log(VIX_t), 1.2, 2.5, 3.0]
res_mrlrj = optimize(obj_MRLRJ, init_guess_jump, NelderMead(), Optim.Options(iterations=1000))
pe_mrlrj = Optim.minimum(res_mrlrj)

println("--- Calibration Results (Percentage Error) ---")
@printf("MRLR (Baseline) PE: %.2f%%\n", pe_mrlr * 100)
@printf("MRLRJ (Jump) PE: %.2f%% (Implied Accuracy: %.2f%%)\n", pe_mrlrj * 100, 100 - pe_mrlrj * 100)

opt_params = Optim.minimizer(res_mrlrj)
@printf("\nOptimal MRLRJ parameters:\nKappa: %.4f\nTheta: %.4f\nSigma: %.4f\nLambda: %.4f\nEta: %.4f\n",
    abs(opt_params[1]), opt_params[2], abs(opt_params[3]), abs(opt_params[4]), abs(opt_params[5]))
