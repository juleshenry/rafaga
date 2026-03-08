using Pkg
Pkg.activate(".")

include("VIXModels.jl")
using .VIXModels
using CSV, DataFrames, Optim, Statistics, Printf

df = CSV.read("../../csvs/VIX_April_16_2021_de_may_19_2021.csv", DataFrame)
filter!(row -> row."Last Price" > 0.0, df)

T = 33.0 / 365.0
VIX_t = 16.25 # spot VIX roughly on April 16, 2021
r = 0.01

strikes = df.Strike
market_prices = df."Last Price"

# Objective function for MRLRJ
function obj_MRLRJ(params)
    κ = abs(params[1])
    θ = params[2]
    σ = abs(params[3])
    λ = abs(params[4])
    η = abs(params[5])
    
    if η <= 1.01 
        return 1e6
    end
    if κ <= 0.01 || σ <= 0.01
        return 1e6
    end
    
    model = MRLRJ(κ, θ, σ, λ, η, VIX_t)
    
    err = 0.0
    for i in 1:length(strikes)
        try
            price = vix_option(model, 0.0, T, strikes[i], VIX_t, r)
            err += abs(price - market_prices[i]) / market_prices[i]
        catch
            return 1e6
        end
    end
    return err / length(strikes)
end

# Starting from better guess
init_guess = [5.0, 2.8, 1.0, 2.0, 2.0]
println("Initial MAPE: ", obj_MRLRJ(init_guess) * 100, "%")

res = optimize(obj_MRLRJ, init_guess, NelderMead(), Optim.Options(iterations=500, g_tol=1e-4, show_trace=false))
opt_params = Optim.minimizer(res)

println("--- MRLRJ Calibration Results ---")
@printf("Kappa: %.4f, Theta: %.4f, Sigma: %.4f, Lambda: %.4f, Eta: %.4f\n", 
    abs(opt_params[1]), opt_params[2], abs(opt_params[3]), abs(opt_params[4]), abs(opt_params[5]))

pe = Optim.minimum(res)
@printf("Mean Absolute Percentage Error (MAPE): %.2f%%\n", pe * 100)
@printf("Implied Accuracy: %.2f%%\n", 100 - pe * 100)
