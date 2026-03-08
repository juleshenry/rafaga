using CSV
using DataFrames
using Statistics
using Dates
using Printf

# Load the historical VIX data
df = CSV.read("../../data/vix_historical.csv", DataFrame, header=3)
# Clean up missing or renaming
rename!(df, :1 => :Date, :2 => :Close)

# Filter out rows with missing dates or close
dropmissing!(df, [:Date, :Close])

# We want to model ln(VIX)
vix = Float64.(df.Close)
log_vix = log.(vix)

n = length(vix)
y = log_vix[2:n]
x = log_vix[1:n-1]

# Simple AR(1) regression: y = alpha + beta * x
x_mean = mean(x)
y_mean = mean(y)

beta = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
alpha = y_mean - beta * x_mean

# Predictions
y_pred = alpha .+ beta .* x

# Calculate R^2
ss_tot = sum((y .- y_mean).^2)
ss_res = sum((y .- y_pred).^2)
r2 = 1.0 - (ss_res / ss_tot)

dt = 1.0 / 252.0
kappa = (1.0 - beta) / dt
theta = alpha / (kappa * dt)
sigma = std(y .- y_pred) / sqrt(dt)

# Also check correlation of simple levels if the user meant VIX (not log)
vix_y = vix[2:n]
vix_x = vix[1:n-1]
vix_pred = exp.(y_pred)

ss_tot_vix = sum((vix_y .- mean(vix_y)).^2)
ss_res_vix = sum((vix_y .- vix_pred).^2)
r2_vix = 1.0 - (ss_res_vix / ss_tot_vix)

# Calculate Percentage Error as in the paper (mean absolute percentage error for prices)
pe = mean(abs.(vix_y .- vix_pred) ./ vix_y)

println("--- MRLR Model Calibration (Time Series) ---")
@printf("Beta: %.6f\n", beta)
@printf("Alpha: %.6f\n", alpha)
@printf("Implied Kappa: %.6f\n", kappa)
@printf("Implied Theta (log): %.6f\n", theta)
@printf("Implied Sigma: %.6f\n", sigma)
@printf("Implied Long-term Mean VIX: %.2f\n", exp(theta))
println("------------------------------")
@printf("R^2 (Log VIX): %.2f%%\n", r2 * 100)
@printf("R^2 (VIX Levels): %.2f%%\n", r2_vix * 100)
@printf("MAPE (Percentage Error): %.2f%%\n", pe * 100)

# Check directional accuracy
dir_actual = sign.(y .- x)
dir_pred = sign.(y_pred .- x)
acc = mean(dir_actual .== dir_pred)
@printf("Directional Accuracy: %.2f%%\n", acc * 100)
