# Trading notes and speculative extensions

> Moved from the README. These are ideas, not results: nothing here is implemented or backtested in this repository. See the README and `results/` for what has actually been verified.

## How to Exploit the MRLRJ Model for Volatility Trading

Fitting the market accurately (to ~97% accuracy with the MRLRJ model) doesn’t mean you have a crystal ball to predict where the VIX is going. Instead, it means you have a highly precise mathematical map of how the market *currently* values risk across different strikes and expirations. 

Here are the four primary ways quantitative hedge funds and volatility traders exploit this kind of mathematical edge:

### 1. Statistical Arbitrage: Harvesting the Volatility Risk Premium ($\mathbb{P}$ vs. $\mathbb{Q}$)
This is a common proprietary trading strategy for volatility.
* **The Concept:** There are two "worlds" of probability. The real world (historical time-series, known as the $\mathbb{P}$-measure) and the options market world (implied by prices, known as the $\mathbb{Q}$-measure). Investors are terrified of VIX spikes, so they overpay for VIX call options as insurance. 
* **The Trade:** You use the historical time-series to calculate the historical jump frequency and size ($\mathbb{P}$). Then you run the options calibration to find the implied jump parameters ($\mathbb{Q}$). If the options market's implied jump intensity ($\lambda$) is wildly higher than the historical baseline, the options are statistically overpriced. You sell out-of-the-money (OTM) VIX calls, dynamically hedge your directional risk, and collect the premium as the expected jumps fail to materialize.

### 2. Relative Value (Dispersion) Trading on the Skew
Since the MRLRJ model fits the "fair" curve of the volatility skew with high accuracy, the remaining error is where the money is.
* **The Concept:** Market makers occasionally misprice specific strikes due to supply/demand imbalances (e.g., a massive fund buys a huge block of VIX 40 calls, temporarily driving the price of that specific strike up).
* **The Trade:** You sweep the live option chain using the MRLRJ model. If the model says a VIX 35 Call should be \$0.60 but it is trading at \$0.80, while the VIX 40 Call is perfectly priced at \$0.48, you have found a local dislocation. You sell the expensive 35 Call and buy the fairly priced 40 Call (a bear call spread) to isolate the mispricing. You are betting the specific strike will revert to the smooth mathematical curve.

### 3. Superior Hedging and Market Making
Because you cannot buy or short the "Spot VIX" (it is an uninvestable index), you *must* hedge VIX options using VIX futures.
* **The Concept:** If you sell a VIX option to collect the premium, you take on directional risk. To neutralize this, you buy/sell VIX futures (Delta hedging).
* **The Trade:** A basic Black-Scholes model gives you the wrong "Delta" because it doesn't account for jumps or mean-reversion. If a jump occurs, your Black-Scholes hedge breaks down and you lose money. By using the MRLRJ model, you calculate a highly precise mathematically-linked Delta. This allows you to act as a market maker—safely capturing the Bid/Ask spread while remaining perfectly neutral to market movements because your hedge accounts for the jump risk.

### 4. Calendar Spreads via Mean-Reversion ($\kappa$)
The MRLRJ model explicitly solves for $\kappa$ (the speed of mean reversion) and $\theta$ (the long-term mean).
* **The Concept:** VIX term structures (the price of next month's options vs. 3 months out) often get distorted during panics.
* **The Trade:** If the VIX spikes to 40 today, short-term options will be incredibly expensive. However, if your model calculates that the historical mean reversion speed ($\kappa$) is very high, it mathematically dictates that the 3-month options are underpricing how fast the VIX will collapse back to its long-term mean ($\theta$). You can construct calendar spreads (e.g., selling near-term volatility and buying long-term volatility) optimally weighted by the model's exact reversion decay curve.

*(Standard Disclaimer: Volatility trading is highly complex, involves extreme tail risks, and historically VIX spikes can wipe out improperly capitalized short-volatility strategies.)*

## Future Work: Modernizing MRLRJ for Maximized Alpha

The MRLRJ model is a classic framework, but modern algorithmic trading and market microstructure require significant extensions to maximize profitability and adaptability in today's markets. Future research and development should focus on the following speculative extensions:

### 1. Rough Volatility Integration ("Rough MRLRJ")
*   **The Theory:** Recent quantitative finance literature demonstrates that volatility is "rough," scaling locally like a fractional Brownian motion with a Hurst parameter $H < 0.5$, rather than the standard $H = 0.5$ of classic MRLRJ.
*   **The Extension:** Replace the standard Brownian driver in the MRLRJ model with a fractional Brownian motion.
*   **Trading Edge:** Standard MRLRJ misprices short-dated VIX options because it cannot generate the steep volatility skew observed in the market without relying on unrealistic jump sizes. A Rough-MRLRJ model accurately predicts the extreme short-term skew, enabling systematic selling of overpriced short-dated VIX OTM options.

### 2. Self-Exciting Jumps via Hawkes Processes
*   **The Theory:** In the standard MRLRJ, jumps arrive randomly and independently via a Poisson process. In reality, financial panics cluster.
*   **The Extension:** Upgrade the "J" in MRLRJ to a **Hawkes Process** (a self-exciting point process).
*   **Trading Edge:** This allows quantification of the "decay rate of panic." A Hawkes-MRLRJ model calculates the exact half-life of volatility clustering, allowing for precise timing of calendar spreads (buying longer-dated VIX puts and selling near-term VIX calls) when the "contagion" phase has peaked.

### 3. Neural SDEs and Deep Calibration
*   **The Theory:** Calibrating MRLRJ using traditional methods is too slow for intraday trading and assumes static parameters.
*   **The Extension:** Use **Universal Differential Equations (Neural SDEs)** to replace the static drift and volatility functions of the MRLRJ with deep neural networks trained continuously on limit order book (LOB) data and SPX/VIX cross-asset signals.
*   **Trading Edge:** The Neural-MRLRJ can detect micro-regime shifts intraday. If the neural drift component detects an impending collapse in mean-reversion speed, traders can front-run the market by buying VIX call butterflies before the broader market prices in the elevated jump risk.

### 4. Maximizing Profit via "Deep Hedging" (Reinforcement Learning)
*   **The Theory:** Traditional MRLRJ trading relies on Black-Scholes-style Greeks, assuming frictionless markets. In reality, VIX options have wide bid-ask spreads, meaning strict delta-hedging bleeds alpha through transaction costs.
*   **The Extension:** Wrap the extended MRLRJ model inside a **Deep Reinforcement Learning (RL)** environment.
*   **Trading Edge:** Train an RL agent to maximize the P&L of a VIX options portfolio where the underlying VIX path is simulated by the Rough-Hawkes-MRLRJ model. The agent learns to hedge only when risk exceeds a certain threshold, optimizing the trade-off between transaction costs and gamma risk.

### 5. Joint SPX-VIX Variance Risk Premium (VRP) Extraction
*   **The Theory:** The VIX is a derivative of SPX options, but MRLRJ models the VIX in isolation. Modern arbitrage requires modeling the joint dynamics.
*   **The Extension:** A coupled model where the SPX follows a local-stochastic volatility model with jumps (LSV-J), and the VIX is deterministically derived from it, but with an added stochastic Variance Risk Premium (VRP) factor modeled via MRLRJ.
*   **Trading Edge:** By dynamically calculating the theoretical VIX and comparing it to the market VIX, traders can isolate the pure VRP. When the VRP is historically stretched, this enables relative value dispersion trades (e.g., shorting VIX futures/options while longing SPX straddles).

## Practical Profit Generation from Modern Extensions

To practically profit from these advanced MRLRJ extensions, a trading desk must transition from static end-of-day calibration to real-time, tick-by-tick ingestion of live options data. 

Here is exactly how a quantitative volatility desk would operationalize these theoretical extensions using live VIX data to generate alpha:

### 1. Rough Volatility: Monetizing the Short-Term Skew
*   **The Live Data Feed:** You stream the live bid/ask prices of VIX options expiring in less than 14 days. 
*   **The Signal:** Live data frequently shows an aggressively steep "volatility skew" for short-dated OTM VIX calls (e.g., 30-delta calls are vastly more expensive than at-the-money options). A standard MRLRJ model struggles to justify this without assuming impossibly large jumps. Your Rough-MRLRJ model, however, accurately maps this skew using the fractional Hurst parameter ($H < 0.5$).
*   **The Practical Trade:** The Rough-MRLRJ model calculates the mathematically "fair" probability of a short-term explosion. If the live market price of a 1-week VIX 25 Call exceeds your Rough-MRLRJ theoretical value, the market is overpaying for short-term panic insurance. You **sell short-dated OTM call credit spreads** (e.g., sell the 25 Call, buy the 30 Call for protection) and collect the structurally overpriced premium, letting the options decay rapidly.

### 2. Hawkes Processes: Fading the Volatility Clustering
*   **The Live Data Feed:** You stream tick-by-tick VIX index levels and VIX futures prices during a market sell-off.
*   **The Signal:** The VIX spikes from 15 to 25 in an hour. Standard models assume this new high level will simply mean-revert linearly. The Hawkes-MRLRJ model continuously calculates the *intensity* ($\lambda_t$) of the "contagion." It looks at the live frequency of the jumps. When the live data shows the time between micro-jumps is lengthening, the Hawkes formula signals that the "self-excitation" phase has mathematically exhausted itself, even though the VIX is still optically high.
*   **The Practical Trade:** The moment the Hawkes intensity function peaks and begins to decay, you execute a **mean-reversion calendar spread**. You sell massively inflated near-term VIX calls (which are pricing in continued contagion) and buy cheaper deferred-month VIX options. You profit as the near-term implied volatility collapses faster than the broader market expects.

### 3. Neural SDEs: Front-Running Micro-Regime Shifts
*   **The Live Data Feed:** You ingest high-frequency Limit Order Book (LOB) data for SPX futures and VIX options (including Level 2 depth, not just top-of-book).
*   **The Signal:** Static models wait for the VIX to move before updating parameters. The Neural SDE is continuously evaluating the LOB imbalances. For example, it detects live liquidity vanishing on the bid side of SPX futures while VIX call open interest suddenly spikes. The neural network's internal drift and volatility functions instantaneously adjust, predicting a massive increase in jump probability *before* the VIX index actually moves.
*   **The Practical Trade:** Because the Neural SDE predicts the parameter shift milliseconds before the broader market reprices the options chain, you aggressively buy **VIX call butterflies** or outright OTM VIX calls at "stale" low prices. When the anticipated jump occurs and the market makers widen their quotes to reflect the new regime, you instantly sell your positions back to the market at a premium.

### 4. Deep Hedging (RL): Minimizing Transaction Friction
*   **The Live Data Feed:** You maintain a portfolio of VIX options. The live feed provides the continuous, real-time bid/ask spreads of the VIX options and VIX futures (which are notoriously wide and illiquid).
*   **The Signal:** Your standard MRLRJ delta-hedging algorithm calculates that your portfolio's delta just shifted by 2%. Traditional quant theory dictates you must immediately buy VIX futures to neutralize the risk.
*   **The Practical Trade:** If you delta-hedge every time the market twitches, crossing the wide VIX bid/ask spread will bleed your account dry. Instead, you feed the live portfolio Greeks and live bid/ask spreads into your trained RL agent. The agent looks at the Rough-Hawkes-MRLRJ path simulations and decides: *"The risk of a jump in the next 10 minutes is low, and the spread is currently too wide. Do not hedge yet."* The RL agent optimizes the exact moment to execute the hedge, drastically reducing transaction costs and preserving the alpha generated by your other strategies.

### 5. Joint SPX-VIX Extraction: Arbitraging the Variance Risk Premium (VRP)
*   **The Live Data Feed:** You simultaneously stream the live options chains for *both* the S&P 500 (SPX) and the VIX.
*   **The Signal:** The VIX is mathematically derived from a strip of SPX options. You use your live SPX options data to calculate the exact theoretical VIX. You then compare this to the live market price of VIX futures/options. The difference between the theoretical VIX (from SPX) and the actual traded VIX is the Variance Risk Premium (VRP). 
*   **The Practical Trade (Relative Value):** If live data shows the market is bidding up VIX options to a level 15% higher than what the underlying SPX options math dictates (a massively stretched VRP driven by pure fear), you execute a **dispersion trade**. You short the overpriced VIX options and buy a delta-neutral SPX straddle. You are perfectly hedged against the actual market movement; you simply profit when the structural pricing mismatch between the SPX chain and the VIX chain collapses back to mathematical parity.
