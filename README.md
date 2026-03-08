# Mean-Reverting Logarithmic Modeling of VIX (Bao, 2013)

This repository contains the implementation of the algorithms presented in the paper *Mean-Reverting Logarithmic Modeling of VIX* by Qunfang Bao (2013).

## Motivation

The initial attempt to model the characteristic functions and complex integration for the VIX pricing options was done in Python. However, due to precision issues when handling exponential jumps and mean-reverting properties inherent to the equations (especially high powers and logs), **Julia is chosen as the primary backend for precision computation**. The Julia implementation natively leverages `BigFloat` where precision bottlenecks were occurring in Python (`float64` underflow/overflow).

The Python code has been retained, refactored, and organized for utility (e.g., historical data fetching, basic diffusion models, Black-Scholes approximations), while the core pricing and option algorithms using the high-precision characteristic functions and jump-diffusion models (MRLR, MRLRJ, MRLRSV) reside in Julia.

## Repository Structure

- `documents/`: Contains the original MPRA paper.
- `python_impl/`: Refactored and consolidated Python code.
  - `models/`: Basic MRLR implementations and SciPy optimization logic.
  - `utils/`: Option pricing utilities, including Black-Scholes.
  - `scripts/`: Data grabbing scripts for VIX using Selenium.
  - `tests/`: Original unit tests.
- `julia_impl/`: The core high-precision implementation of the paper's models using Julia.
  - `src/VIXModels.jl`: Core implementations for `MRLR` and `MRLRJ` taking advantage of `BigFloat`.
  - `test/runtests.jl`: Unit tests for the models.

## How to Install and Run Julia

### 1. Installation

The easiest way to install Julia is through `juliaup`, the official Julia version manager.

**Windows / Linux / macOS (via curl):**
```bash
curl -fsSL https://install.julialang.org | sh
```
Follow the on-screen instructions. Once installed, restart your terminal.

Alternatively, you can download the installer from the [official Julia website](https://julialang.org/downloads/).

### 2. Environment Setup

To run the Julia code, you need to instantiate the project environment. This ensures all the required packages (e.g., `Distributions`, `QuadGK`) are downloaded.

```bash
cd julia_impl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 3. Running Tests

To verify that the high-precision VIX models are working correctly:

```bash
cd julia_impl
julia --project=. -e 'using Pkg; Pkg.test()'
```
