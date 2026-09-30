# Mean-Reverting Logarithmic Modeling of VIX (Bao, 2013), revisited

An open Julia implementation of the VIX models in Qunfang Bao's thesis
*Mean-Reverting Logarithmic Modeling of VIX* (Zhejiang University, 2013;
[MPRA 46413](https://mpra.ub.uni-muenchen.de/46413/)), a check of the thesis'
published calibration, and an out-of-sample test on the 2026 VIX option chain.

Authored by Julian Philip Henry.

**Paper:** [`paper/rafaga.pdf`](paper/rafaga.pdf) (5 pages): models and motivation,
verification, calibration protocol, results and full parameter disclosure.

## Models

| Model | Dynamics of ln VIX | Why it exists |
|---|---|---|
| MRLR | mean-reverting OU | positivity, mean reversion, level-proportional vol; log-normal, so **no skew** |
| MRLRJ | + upward exponential jumps | VIX spikes → positive skew, strongest at short maturities |
| MRLRSV | + square-root stochastic vol-of-vol, correlated | vol-of-vol rises with the VIX → skew that builds with maturity |

Futures are ψ(−i) of the characteristic function; calls use the Gil-Pelaez
(Heston-style) formula of the thesis. MRLRSV's Riccati ODE is solved by RK4, as the
thesis recommends. MRLRSVJ is not implemented (the thesis finds it adds nothing over MRLRSV).

## Main results

**The implementation matches the thesis.** The 2011 quotes were never published, so
the thesis' errors cannot be recomputed. Its MRLRJ and MRLRSV parameter tables were
fitted separately to the same market; priced with this code they agree to 2–3.5
cents, and reproduce the skews of the thesis' Figure 7.3
([`results/paper_2011_check.md`](results/paper_2011_check.md)).

**2026 (Cboe chain, 29 Sep 2026, spot VIX 16.04), the thesis' own procedure**
([`results/calibration_2026-09-29.md`](results/calibration_2026-09-29.md)):

| | MRLR | MRLRJ | MRLRSV | thesis 2011 (J / SV) |
|---|---|---|---|---|
| pricing error (PE), 4 maturities | 28–38% | 1.9–2.7% | 1.6–2.8% | 3.1–5.2% / 3.0–5.1% |
| model prices inside bid–ask | 18–22% | 93–100% | 92–100% | – |
| out-of-sample PE (held-out strikes) | 29–40% | 1.9–4.7% | 2.2–3.5% | – |
| one parameter set for all maturities | 69–95% | 5–13% | 12–31%* | – |

\*time-capped optimisation, an upper bound.

- **Confirmed:** MRLR cannot price the VIX skew; jumps *or* stochastic vol-of-vol fix it.
- **Not identified:** in MRLR, κ has no effect on a single-maturity fit (identical loss
  for κ = 0.5…32), so the thesis' MRLR κ values are arbitrary. The 2026 MRLRJ fits
  drift to κ ≈ 0.01, θ ≈ −20. The thesis' θ = 3.00 (MRLRJ) and ρ = 1.00 (MRLRSV)
  sit on undisclosed bounds.
- **New regime:** 2026 prices rare, large jumps (λ ≈ 2–5/yr, mean log-jump 0.3)
  where 2011 had λ = 60–170/yr of small ones; MRLRSV needs vol-of-vol 3–7× the 2011 values.
- **No term-structure model:** no constant-parameter model fits all four maturities.
  The family works as a per-maturity smile model, not as a description of VIX dynamics.

## Repository

```
julia_impl/
  src/            VIXModels package: models, pricing, calibration, Cboe data
  test/           34 tests (closed form vs Fourier, nested limits, arbitrage
                  bounds, Monte Carlo, RK4 convergence, BigFloat cross-check)
  scripts/
    fetch_cboe.jl        save today's Cboe VIX chain to data/cboe/
    reproduce_paper.jl   2011 consistency check  -> results/paper_2011_check.md
    calibrate_2026.jl    2026 study              -> results/calibration_<date>.md
    make_figures.jl      TikZ figures            -> paper/figures/
data/cboe/        raw Cboe snapshots (JSON)
results/          generated reports and per-strike fits
paper/            LaTeX source and PDF
documents/        the thesis and related papers
docs/             trading notes (ideas, not results)
```

## Running it

Requires Julia ≥ 1.10 ([juliaup](https://julialang.org/downloads/)).

```bash
cd julia_impl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. scripts/reproduce_paper.jl                # seconds
julia -t auto --project=. scripts/calibrate_2026.jl         # ~25 min on 8 threads
julia --project=. scripts/make_figures.jl
cd ../paper && pdflatex rafaga.tex && pdflatex rafaga.tex
```

To apply the study to a new day, run `scripts/fetch_cboe.jl` after the 16:15 ET
close; `calibrate_2026.jl` uses the newest snapshot in `data/cboe/` (or pass a path).

Calibration follows the thesis (ch. 7): calls with positive bid and open interest,
mid quotes, loss Σ(ΔC)² + 8·Σ(Δ ln C)², four monthly maturities. The optimiser details
the thesis omits (transforms, bounds, starts, seeds, limits) are in
`src/calibration.jl` and Section 5 of the paper.

## Notes

- Float64 is sufficient: a 25-digit BigFloat quadrature agrees with the pricer to 1e-9.
- The older `csvs/` (2021 SPX/AAPL/VIX chains) and `data/vix_historical.csv` are kept
  but not used by the current scripts.
- Trading ideas previously in this README are in [`docs/trading_notes.md`](docs/trading_notes.md);
  they are speculative and untested here.
