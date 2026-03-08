using Pkg
Pkg.activate(".")

include("VIXModels.jl")
using .VIXModels
using CSV, DataFrames, Optim, Statistics, Printf

df = CSV.read("../../csvs/VIX_April_16_2021_de_may_19_2021.csv", DataFrame)

# Filter valid options
filter!(row -> row."Last Price" > 0.0, df)

# We are on 2021-04-16 and maturity is 2021-05-19 -> ~33 days -> T = 33/365
T = 33.0 / 365.0
VIX_t = 16.25 # spot VIX roughly on April 16, 2021
r = 0.01

strikes = df.Strike
market_prices = df."Last Price"

# Test evaluation of the initial guess
model = MRLRJ(5.0, 2.8, 1.0, 2.0, 2.0, VIX_t)

println("Testing VIX option pricing...")
for i in 1:length(strikes)
    try
        price = vix_option(model, 0.0, T, strikes[i], VIX_t, r)
        println("Strike $(strikes[i]): Market $(market_prices[i]), Model $price")
    catch e
        println("Error on strike $(strikes[i]): $e")
    end
end
