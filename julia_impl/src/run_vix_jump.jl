using CSV
using DataFrames
using Statistics
using Dates
using Printf

df = CSV.read("../../data/vix_historical.csv", DataFrame, header=3)
rename!(df, :1 => :Date, :2 => :Close)
dropmissing!(df, [:Date, :Close])
vix = Float64.(df.Close)
log_vix = log.(vix)

n = length(vix)
y = log_vix[2:n]
x = log_vix[1:n-1]

# MRLRJ: log-VIX follows a mean-reverting jump diffusion
# We can identify jumps as large residuals from the AR1 model
x_mean = mean(x)
y_mean = mean(y)
beta = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
alpha = y_mean - beta * x_mean

y_pred_base = alpha .+ beta .* x
res = y .- y_pred_base

# Threshold for jumps (e.g. 2 or 3 standard deviations)
sigma_res = std(res)
jump_idx = abs.(res) .> 2.5 * sigma_res

# Fit basic diffusion on non-jump parts
beta_no_jump = sum((x[.!jump_idx] .- mean(x[.!jump_idx])) .* (y[.!jump_idx] .- mean(y[.!jump_idx]))) / sum((x[.!jump_idx] .- mean(x[.!jump_idx])).^2)
alpha_no_jump = mean(y[.!jump_idx]) - beta_no_jump * mean(x[.!jump_idx])

# Predictions with jump included (perfect hindsight for jump size, just to see upper bound, or just AR1?)
# If we do out-of-sample or just 1-day ahead expected value, the expected jump size is lambda * mean_jump
# But we only predict expected value, which doesn't know jump happens.
# Wait, if we just use the fitted expectations:
y_pred_jump = alpha_no_jump .+ beta_no_jump .* x
vix_y = vix[2:n]
vix_pred = exp.(y_pred_jump) # ignoring jump compensator for simplicity

pe = mean(abs.(vix_y .- vix_pred) ./ vix_y)

println("--- MRLR Jump Diffusion Model Calibration (Time Series) ---")
@printf("MAPE (Percentage Error): %.2f%%\n", pe * 100)
@printf("Accuracy: %.2f%%\n", 100 - pe * 100)
