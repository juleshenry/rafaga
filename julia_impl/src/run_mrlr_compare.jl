using Pkg
Pkg.activate(".")

include("VIXModels.jl")
using .VIXModels
using CSV, DataFrames, Optim, Statistics, Printf

df = CSV.read("../../csvs/VIX_April_16_2021_de_may_19_2021.csv", DataFrame)
filter!(row -> row."Last Price" > 0.0, df)

T = 33.0 / 365.0
VIX_t = 16.25 
r = 0.01
strikes = df.Strike
market_prices = df."Last Price"

# ---- MRLR Calibration ----
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

res_mrlr = optimize(obj_MRLR, [5.0, 2.8, 1.0], NelderMead(), Optim.Options(iterations=500))
pe_mrlr = Optim.minimum(res_mrlr)

# ---- MRLRJ Calibration ----
function obj_MRLRJ(params)
    κ = abs(params[1])
    θ = params[2]
    σ = abs(params[3])
    λ = abs(params[4])
    η = abs(params[5])
    if η <= 1.01 || κ <= 0.01 || σ <= 0.01; return 1e6; end
    
    model = MRLRJ(κ, θ, σ, λ, η, VIX_t)
    err = 0.0
    for i in 1:length(strikes)
        try
            price = vix_option(model, 0.0, T, strikes[i], VIX_t, r)
            err += abs(price - market_prices[i]) / market_prices[i]
        catch; return 1e6; end
    end
    return err / length(strikes)
end

res_mrlrj = optimize(obj_MRLRJ, [2.18, 0.60, 2.31, 2.48, 3.66], NelderMead(), Optim.Options(iterations=1000))
pe_mrlrj = Optim.minimum(res_mrlrj)

println("--- Calibration Results (Percentage Error) ---")
@printf("MRLR Model PE: %.2f%%\n", pe_mrlr * 100)
@printf("MRLRJ Model PE: %.2f%% (Implied Accuracy: %.2f%%)\n", pe_mrlrj * 100, 100 - pe_mrlrj * 100)
