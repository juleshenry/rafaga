"""
Mean-reverting logarithmic models of the VIX from Bao (2013), *Mean-Reverting
Logarithmic Modeling of VIX* (MPRA paper 46413): MRLR, MRLRJ and MRLRSV,
with VIX future / option pricing and the chapter-7 calibration procedure.
"""
module VIXModels

using Dates
using DataFrames
using Distributions: Normal, Poisson, cdf
using Downloads
using JSON3
using Optim
using QuadGK: gauss
using Random
using Statistics

export VIXModel, MRLR, MRLRJ, MRLRSV, paramnames, params, modelname, with_theta
export cf, vix_future, vix_calls, vix_call, vix_calls_fourier
export black_call, black_iv, simulate_terminal
export OptionSlice, mid, subslice, mmlse, fit_stats, anchor, calibrate
export parse_cboe, fetch_cboe, year_fraction, implied_forwards, build_slices

include("models.jl")
include("pricing.jl")
include("calibration.jl")
include("marketdata.jl")

end # module
