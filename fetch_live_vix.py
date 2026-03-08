import yfinance as yf
import pandas as pd
import datetime

# Fetch live VIX ticker
vix = yf.Ticker("^VIX")

# Get available expiration dates
expirations = vix.options
print("Available expirations:", expirations)

if len(expirations) > 2:
    # Pick a maturity about ~1 month out (similar to the paper's target)
    target_exp = expirations[2]
else:
    target_exp = expirations[0]

print(f"Fetching option chain for {target_exp}")

opt_chain = vix.option_chain(target_exp)
calls = opt_chain.calls

# Calculate mid price
calls['Mid'] = (calls['bid'] + calls['ask']) / 2.0

# Filter out illiquid options (bid or ask is 0)
calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()

# Save to CSV
calls.to_csv("live_vix_calls.csv", index=False)
print(f"Saved {len(calls)} call options to live_vix_calls.csv")

# Get spot VIX to use in our model
vix_history = vix.history(period="1d")
spot_vix = vix_history['Close'].iloc[-1]
print(f"Current Spot VIX: {spot_vix}")

with open("live_vix_meta.txt", "w") as f:
    f.write(f"{target_exp},{spot_vix}\n")

