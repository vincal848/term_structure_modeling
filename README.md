# Term-Structure Models on the US Treasury Curve

**Question.** How well do the classic one-factor short-rate models reproduce the actual US Treasury curve, and what does it take to fit it exactly?

**Data.** FRED H.15 constant-maturity Treasury yields (1M–30Y) and the 3-month T-bill, weekly from 1990, on three curve shapes: normal (2017-06-30), steep near zero (2021-06-30), and inverted (2023-06-30).

**Methods.**
- Zero curves bootstrapped from par yields, with Nelson-Siegel fits.
- One simulator for the general one-factor SDE, validated against closed-form Vasicek and CIR bond prices.
- Calibration two ways: to the T-bill history (measure $P$; exact AR(1) and CIR likelihoods) and to each day's curve (measure $Q$).
- Hull-White with its drift taken from the fitted forward curve.

**Headline results**

| RMSE vs market zero yields (bp) | 2017 normal | 2021 steep | 2023 inverted |
|---|---|---|---|
| Vasicek, historical parameters (P) | 120.1 | 101.7 | 45.1 |
| CIR, historical parameters (P) | 100.4 | 66.3 | 49.1 |
| Vasicek, drift fitted to curve (Q) | 7.5 | 11.2 | 24.4 |
| CIR, drift fitted to curve (Q) | 7.5 | 10.9 | 24.4 |
| Hull-White (a, σ historical; θ(t) from curve) | 5.8 | 3.3 | 11.1 |

- **Historical parameters don't price the curve.** The long-run rate implied by the curve is 1.7–2.6 percentage points above the historical estimate. That gap is the market price of interest-rate risk.
- **One-factor, time-homogeneous models have one curve shape.** With the drift fitted to the curve they come within about 8 bp on a normal curve, but miss humped and inverted curves by 11–24 bp.
- **Hull-White reprices the fitted curve exactly.** Monte Carlo agrees within 0.4 bp. Its remaining error is the Nelson-Siegel fit itself, largest in 2023 because of the cheap 20-year bond.
- **Mean reversion is weakly identified.** The half-life is about 6 years, with a standard error of about half of κ. On 10-year windows it ranges from 0.5 years to no mean reversion at all.
- **Discretisation matters.** Simulating CIR with Euler and flooring at zero, as the original R version did, overstates the 10-year yield by 12.7 bp.

## Contents

1. [How to run](#how-to-run)
2. [Background](#background)
3. [Bootstrapping and yield curves](#bootstrapping-and-yield-curves)
4. [Term structure](#term-structure)
5. [One-factor term-structure models](#one-factor-term-structure-models)
6. [The affine class](#the-affine-class)
7. [Time-varying coefficients: Hull-White](#time-varying-coefficients-hull-white)
8. [Results](#results)
9. [Model comparison, limitations, extensions](#model-comparison)

## How to run

```bash
pip install -r requirements.txt
pytest                               # 26 tests: closed forms, calibration recovery, Monte Carlo agreement
python scripts/run_analysis.py       # regenerates every figure (figures/) and table (results/)
python scripts/fetch_data.py         # optional: refresh the cached FRED data (no API key)
```

The FRED data is cached in `data/`, so the analysis runs offline. Every number and figure in this README is produced by `scripts/run_analysis.py`.

```
termstructure/        importable package
  data.py             FRED download and cache
  curves.py           bootstrapping, Nelson-Siegel, forward curves
  short_rate.py       general one-factor SDE simulator, Monte Carlo bond prices
  affine.py           Vasicek and CIR closed forms and exact transitions
  calibrate.py        historical (P) and curve (Q) calibration
  hull_white.py       Hull-White fitted to the initial curve
scripts/              run_analysis.py, fetch_data.py
tests/                pytest suite, run in CI on every push
data/  figures/  results/
```

---

## Background

For this project I wanted to work with term structures of interest rates and the methods, tools, and processes I had seen used in research publications.

My primary reference was Chapter 7 of *Dynamic Asset Pricing Theory* by Darrell Duffie, "Term-Structure Models", which I found interesting for how it expands on factors from the initial models. I supplemented it with Chapter 10 of *Stochastic Calculus for Finance II* by Steven Shreve, which has the same title.

The market has a yield curve rather than a single rate. The summary below follows Section 10.1 of Shreve, which I found more informative on affine models before moving to the Heath-Jarrow-Morton framework. The two differ in focus: affine models work bottom-up, starting from short-rate dynamics that produce the yield curve, while HJM works top-down from forward rates.

## Bootstrapping and yield curves

Consider zero-coupon bonds paying 1 at maturity, with prices $B(0,T_j)$ for dates $0=T_0 < T_1 < T_2 < \dots < T_n$. A coupon bond pays fixed amounts $C_1, C_2, \dots, C_j$ at those dates, the last including the principal, so its price at time zero is

$$\sum_{i=1}^j C_i\, B(0,T_i).$$

Given coupon-bond prices, we can solve recursively for zero-coupon prices: the bond maturing at $T_1$ gives $B(0,T_1)$, the bond maturing at $T_2$ then gives $B(0,T_2)$, and so on up to $T_n$. This is *bootstrapping* zero-coupon prices from coupon-paying bond prices.

With continuous compounding, the zero-coupon price and yield are related by

$$\text{price of zero-coupon bond} = \text{face value} \times e^{-\text{yield} \times t_{\text{maturity}}}.$$

The market therefore has a curve of yields rather than a single rate, which we can think of as an interpolation of finitely many observed maturity-yield pairs. The *short rate* is idealised as the yield at the shortest maturity, or the overnight rate.

### Implementation: bootstrapping the Treasury curve

FRED's constant-maturity Treasury (CMT) yields are **par yields** on a semiannual bond-equivalent basis. The three dates were chosen for their shapes (par yields, %):

| maturity | 2017-06-30 (normal) | 2021-06-30 (steep, near zero) | 2023-06-30 (inverted) |
|---|---|---|---|
| 1m | 0.84 | 0.05 | 5.24 |
| 3m | 1.03 | 0.05 | 5.43 |
| 6m | 1.14 | 0.06 | 5.47 |
| 1y | 1.24 | 0.07 | 5.40 |
| 2y | 1.38 | 0.25 | 4.87 |
| 3y | 1.55 | 0.46 | 4.49 |
| 5y | 1.89 | 0.87 | 4.13 |
| 7y | 2.14 | 1.21 | 3.97 |
| 10y | 2.31 | 1.45 | 3.81 |
| 20y | 2.61 | 2.00 | 4.06 |
| 30y | 2.84 | 2.06 | 3.85 |

`curves.bootstrap_zeros` works as follows:

- Quotes shorter than one coupon period carry no coupon, so they are already zero rates.
- Beyond that, par yields are interpolated onto the semiannual coupon grid and each discount factor solves the par-bond condition

$$\frac{c_n}{2}\sum_{i<n} B(0,T_i) + \left(1+\frac{c_n}{2}\right)B(0,T_n) = 1.$$

- Zero rates are reported continuously compounded, $B(0,T) = e^{-z(T)\,T}$.
- The interpolation is monotone piecewise-cubic (PCHIP). A natural cubic spline overshoots between the sparse 10y, 20y and 30y quotes and invents humps and dips that aren't in the data.

![Bootstrapped zero curves](figures/01_bootstrap.png)

### Implementation: Nelson-Siegel

A smooth parametric curve is fitted to the bootstrapped zeros:

$$z(\tau) = \beta_0 + \beta_1\frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_2\left(\frac{1-e^{-\lambda\tau}}{\lambda\tau}-e^{-\lambda\tau}\right)$$

- $\beta_0$ is the long-run level, $\beta_0+\beta_1$ the instantaneous short rate, and $\beta_2$ the hump.
- For fixed $\lambda$ the model is linear in the betas. `fit_nelson_siegel` grid-searches $\lambda$ with least squares for the betas, then refines all four jointly, so it needs no starting values.
- Only the quoted maturities are used, so interpolated points don't get extra weight.
- The instantaneous forward curve $f(0,\tau) = \beta_0 + \beta_1 e^{-\lambda\tau} + \beta_2\lambda\tau e^{-\lambda\tau}$ and its slope are available in closed form. Hull-White needs both.

| parameter | 2017-06-30 (normal) | 2021-06-30 (steep, near zero) | 2023-06-30 (inverted) |
|---|---|---|---|
| β0 level (%) | 3.22 | 2.49 | 3.66 |
| β1 slope (%) | −2.29 | −2.43 | 1.33 |
| β2 curvature (%) | 0.00 | −2.75 | 3.60 |
| λ | 0.24 | 0.53 | 2.13 |
| fit RMSE (bp) | 5.85 | 3.30 | 11.11 |

In 2023 the 20-year point sits above both the 10-year and the 30-year. This is a known Treasury anomaly, not a data error: the 20-year bond was reintroduced in 2020 and has traded cheap. No smooth three-factor curve passes through it, which is why the inverted date has the largest fit error.

![Nelson-Siegel fits](figures/02_nelson_siegel.png)

## Term structure

Set up a probability space $(\Omega,\mathscr{F},P)$ with a filtration $\mathbb{F} = \{\mathscr{F}_t : 0 \le t \le T\}$ generated by a standard Brownian motion $B$ in $\mathbb{R}^d$, $d \ge 1$.

The short rate $r$ satisfies $\int_0^T |r_t|\,dt < \infty$. Investing one unit at time $t$ and continually reinvesting at the short rate gives $e^{\int_t^s r_u\,du}$ at time $s$.

Assuming no arbitrage, there is a probability measure $Q$ under which a security paying a lump sum $Z$ at $s$ has time-$t$ price

$$E_t^{Q}\left[e^{-\int_t^s r_u\,du}\, Z\right],$$

where $E_t^Q$ is the $\mathscr{F}_t$-conditional expectation under $Q$ and $Z$ is $\mathscr{F}_s$-measurable. Setting $Z=1$, the price at $t$ of the zero-coupon bond maturing at $s$ is

$$\Lambda_{t,s} \equiv E_t^{Q}\left[e^{-\int_t^s r_u\,du}\right].$$

This is the discount function, or *the term structure of interest rates*. It is usually quoted as a yield curve, with continuously compounded yield

$$y_{t,\tau} = -\frac{\log \Lambda_{t,t+\tau}}{\tau},$$

or equivalently in terms of forward rates. In the models below the short rate is driven by a Brownian motion under $Q$, obtained from Girsanov's theorem.

### Implementation: checking $\Lambda_{t,s}$ numerically

If the simulator and the closed-form prices below are both right, Monte Carlo estimates of the expectation have to match the formulas within sampling error.

- `mc_bond_price` averages $e^{-\int_0^T r\,dt}$ over simulated paths, with the integral computed by the trapezoid rule.
- Paths use the exact transition laws: Gaussian for Vasicek, noncentral $\chi^2$ for CIR.
- The grid is weekly, with 10,000 paths.
- Parameters: $r_0 = 2\%$, $\kappa = 0.3$, $\theta = 4\%$, with $\sigma = 1\%$ for Vasicek and $\sigma = 0.05$ for CIR.

| model | T (years) | closed form | Monte Carlo | std err | abs diff / s.e. | yield diff (bp) |
|---|---|---|---|---|---|---|
| Vasicek | 1 | 0.97755 | 0.97755 | 0.00000 | 1.27 | 0.00 |
| Vasicek | 2 | 0.95139 | 0.95139 | 0.00000 | 0.52 | 0.00 |
| Vasicek | 3 | 0.92294 | 0.92294 | 0.00000 | 0.51 | −0.01 |
| Vasicek | 5 | 0.86292 | 0.86292 | 0.00001 | 0.01 | 0.00 |
| Vasicek | 7 | 0.80256 | 0.80254 | 0.00002 | 1.09 | 0.05 |
| Vasicek | 10 | 0.71627 | 0.71621 | 0.00004 | 1.54 | 0.09 |
| Vasicek | 20 | 0.48425 | 0.48420 | 0.00008 | 0.63 | 0.05 |
| Vasicek | 30 | 0.32646 | 0.32651 | 0.00009 | 0.60 | −0.06 |
| CIR | 1 | 0.97754 | 0.97753 | 0.00004 | 0.21 | 0.08 |
| CIR | 2 | 0.95135 | 0.95128 | 0.00010 | 0.78 | 0.39 |
| CIR | 3 | 0.92285 | 0.92272 | 0.00016 | 0.82 | 0.47 |
| CIR | 5 | 0.86270 | 0.86241 | 0.00028 | 1.03 | 0.66 |
| CIR | 7 | 0.80220 | 0.80200 | 0.00037 | 0.53 | 0.36 |
| CIR | 10 | 0.71579 | 0.71560 | 0.00048 | 0.40 | 0.27 |
| CIR | 20 | 0.48374 | 0.48404 | 0.00058 | 0.51 | −0.31 |
| CIR | 30 | 0.32606 | 0.32656 | 0.00051 | 0.98 | −0.51 |

Every maturity agrees within 1.6 standard errors, and within 1 bp in yield. Vasicek's standard errors are much smaller because its paths are antithetic (each shock is paired with its negative).

![Monte Carlo vs closed form](figures/04_mc_vs_closed_form.png)

## One-factor term-structure models

$$dr_t = \mu(r_t, t)\,dt + \sigma(r_t, t)\,dB_t^{Q}$$

The short rate is the only factor the yield curve depends on, so bond prices can be written $\Lambda_{t,s} = F(t,s,r_t)$ for a fixed $F: [0,T] \times [0,T] \times \mathbb{R} \rightarrow \mathbb{R}$.

Each model is a special case of the SDE

$$dr_t = \left[K_0(t) + K_1(t)\,r_t + K_2(t)\,r_t \log r_t\right]dt + \left[H_0(t) + H_1(t)\,r_t\right]^v dB_t^{Q},$$

where $K_0, K_1, K_2, H_0, H_1$ are continuous functions on $[0,T]$ and the exponent $v$ ranges from 0.5 to 1.5. The Cox-Ingersoll-Ross (CIR) model has non-zero $K_0$, $K_1$ and $H_1$ with $v = 0.5$. The Pearson-Sun model is CIR plus $H_0$, with the same exponent. With $v = 1$ we have the Dothan, Merton (Ho-Lee), Vasicek and Black-Karasinski models, and $v = 1.5$ gives the Constantinides-Ingersoll model.

### Implementation: one simulator for the whole family

`short_rate.OneFactorModel(K0, K1, K2, H0, H1, v)` implements this SDE directly, so each model is only a set of coefficients:

| Model | $K_0$ | $K_1$ | $K_2$ | $H_0$ | $H_1$ | $v$ |
|---|---|---|---|---|---|---|
| Vasicek | $\kappa\theta$ | $-\kappa$ | 0 | $\sigma$ | 0 | 1 |
| CIR | $\kappa\theta$ | $-\kappa$ | 0 | 0 | $\sigma^2$ | ½ |
| Dothan | 0 | $\mu$ | 0 | 0 | $\sigma$ | 1 |
| Black-Karasinski | 0 | $\kappa\theta_{\log}+\sigma^2/2$ | $-\kappa$ | 0 | $\sigma$ | 1 |

Black-Karasinski is $d\log r = \kappa(\theta_{\log}-\log r)\,dt + \sigma\,dB$; Itô's lemma turns it into the coefficients in the last row.

The simulator supports four schemes:
- **`euler`**: plain Euler-Maruyama, for Gaussian models, where negative rates are part of the model.
- **`euler_floor`**: Euler, then $\max(r, 0)$ after each step (the original R approach).
- **`full_truncation`**: coefficients evaluated at $\max(r,0)$ (Lord, Koekkoek and van Dijk, 2010), the standard scheme for square-root models.
- **`log_euler`**: Euler on $\log r$, for lognormal models.

These are the original toy simulations from the first version of this repo (same parameters: $r_0 = 3\%$, speed 0.1, target 5%, $\sigma = 0.02$, one year of daily steps), now with 20 paths each:

![The one-factor family](figures/03_one_factor_family.png)

The same $\sigma = 0.02$ means something different in each model:
- **Vasicek:** an absolute volatility of 2 percentage points a year, so paths can go negative.
- **CIR:** $0.02\sqrt{r} \approx 0.35$ points.
- **Black-Karasinski:** a 2% *relative* volatility.

So parameters have to be calibrated per model, not shared. The original "Hull-White" simulation used a constant target, which is identical to Vasicek, so it's replaced by a real Hull-White [below](#time-varying-coefficients-hull-white).

#### Discretisation matters for CIR

When the Feller condition $2\kappa\theta \ge \sigma^2$ fails, CIR paths hit zero often, and flooring them at zero pushes rates up. The test case is a model that violates Feller ($\kappa = 0.3$, $\theta = 4\%$, $\sigma = 0.2$, so $2\kappa\theta = 0.024 < \sigma^2 = 0.04$), used to price a 10-year bond from $r_0 = 1\%$ with 20,000 paths:

| scheme | step | share of steps at 0 | 10y yield error (bp) | ± 2 s.e. (bp) |
|---|---|---|---|---|
| Euler + floor (original) | monthly | 0.033 | **12.7** | 1.8 |
| full truncation | monthly | 0.065 | −1.2 | 1.9 |
| exact (noncentral χ²) | monthly | 0.000 | −1.4 | 2.8 |
| Euler + floor (original) | weekly | 0.014 | **4.0** | 1.8 |
| full truncation | weekly | 0.027 | −1.8 | 1.8 |
| exact (noncentral χ²) | weekly | 0.000 | −2.4 | 2.8 |

The floored scheme's bias shrinks on a finer grid but stays outside the Monte Carlo noise. Full truncation and exact sampling are within noise; everything below uses exact sampling.

#### $-K_1$: mean reversion

A negative $K_1$ acts as a mean-reversion parameter: high short rates produce negative drift and low short rates positive drift.

### Implementation: calibrating to the T-bill history (measure $P$)

**Vasicek.** Its exact discretisation is an AR(1):

$$r_{t+\Delta} = a + b\,r_t + \varepsilon_t,\qquad b=e^{-\kappa\Delta},\quad a=\theta(1-b),\quad \operatorname{Var}\varepsilon=\frac{\sigma^2(1-b^2)}{2\kappa}.$$

So OLS on weekly data ($\Delta = 1/52$) is the exact maximum-likelihood estimator, and standard errors for $(\kappa,\theta,\sigma)$ follow from the delta method (`fit_vasicek_ar1`).

**CIR.** Fitted by exact maximum likelihood using its noncentral-$\chi^2$ transition density, with standard errors from the inverse Hessian (`fit_cir_mle`). CIR can't produce zero or negative rates, so the 2009–15 and 2020–21 bill observations are floored at 1 bp.

**Data preparation.**
- The 3-month T-bill (FRED `DTB3`) is quoted on a bank-discount basis and is converted to a continuous rate first.
- Each date uses an **expanding window** from 1990 up to that date, so nothing after the date is used.

| expanding window 1990 → date | 2017-06-30 | 2021-06-30 | 2023-06-30 |
|---|---|---|---|
| Vasicek κ | 0.116 ± 0.059 | 0.107 ± 0.055 | 0.105 ± 0.054 |
| Vasicek θ (%) | 0.74 ± 1.59 | 0.36 ± 1.65 | 1.91 ± 1.22 |
| Vasicek σ (%) | 0.723 ± 0.014 | 0.702 ± 0.012 | 0.708 ± 0.012 |
| half-life ln 2 / κ (years) | 6.0 | 6.5 | 6.6 |
| CIR κ | 0.150 ± 0.068 | 0.162 ± 0.068 | 0.102 ± 0.066 |
| CIR θ (%) | 1.22 ± 0.56 | 1.13 ± 0.48 | 1.90 ± 1.20 |
| CIR σ | 0.059 ± 0.001 | 0.060 ± 0.001 | 0.060 ± 0.001 |
| CIR Feller (2κθ ≥ σ²) | yes | yes | yes |
| weekly observations | 1434 | 1642 | 1747 |

**Volatility is pinned down precisely; the drift is not.**
- The long-run mean $\theta_P$ is uncertain by ±1.2–1.7 percentage points and sits below the average rate over the sample. With $b$ this close to 1, $\theta = a/(1-b)$ divides by a tiny number, and a sample in which rates mostly trended down pulls the estimate low. This is the textbook near-unit-root problem.
- Shorter windows are worse, because a decade of rates usually contains one regime rather than repeated reversions:

| 10-year window | κ | θ (%) | half-life (years) |
|---|---|---|---|
| 2007–2017 | 1.445 | 0.18 | 0.5 |
| 2011–2021 | 0.113 | 0.63 | 6.1 |
| 2013–2023 | AR(1) b > 1 | – | no mean reversion |

![Short-rate history](figures/05_short_rate_history.png)

### Time-varying coefficients

Some model differences come purely from whether coefficients are constant or time-varying. For example, the Merton model of the term structure is called the Ho-Lee model when its coefficients vary with time.

## The affine class

Affine (constant-plus-linear) models have $K_2 = 0$ and $v = 0.5$. They include Vasicek ($H_1 = 0$), CIR ($H_0 = 0$), Merton/Ho-Lee ($K_1 = H_1 = 0$) and Pearson-Sun. Vasicek can equally be written with $v = 1$ and $H_0 = \sigma$, as in the table above. Bond prices are exponential-affine in the short rate, $\Lambda_{t,t+\tau} = A(\tau)\,e^{-B(\tau)\,r_t}$, so yields are affine in $r_t$:

$$y(\tau) = \frac{B(\tau)\,r_t - \log A(\tau)}{\tau}.$$

### Vasicek

$$dR(t) = (\alpha - \beta R(t))\,dt + \sigma\,dW(t),$$

where $\alpha$, $\beta$ and $\sigma$ are positive constants. In the code, $\kappa = \beta$ and $\theta = \alpha/\beta$, and

$$B(\tau) = \frac{1-e^{-\kappa\tau}}{\kappa},\qquad \log A(\tau) = \left(\theta - \frac{\sigma^2}{2\kappa^2}\right)\left(B(\tau)-\tau\right) - \frac{\sigma^2 B(\tau)^2}{4\kappa}.$$

### Cox-Ingersoll-Ross (CIR)

$$dR(t) = (a - b R(t))\,dt + \sigma\sqrt{R(t)}\,d\widetilde{W}(t)$$

With $\kappa = b$, $\theta = a/b$ and $\gamma = \sqrt{\kappa^2 + 2\sigma^2}$:

$$B(\tau) = \frac{2\left(e^{\gamma\tau}-1\right)}{(\gamma+\kappa)\left(e^{\gamma\tau}-1\right)+2\gamma},\qquad A(\tau) = \left[\frac{2\gamma\, e^{(\kappa+\gamma)\tau/2}}{(\gamma+\kappa)\left(e^{\gamma\tau}-1\right)+2\gamma}\right]^{2\kappa\theta/\sigma^2}.$$

### Implementation: do the affine models price the curve?

Model yield curves are built two ways on each date, both starting from that day's observed short rate $r_0$ (the 1-month zero):

- **Historical ($P$) parameters** from the calibration above, plugged straight into the closed forms.
- **Curve-implied ($Q$) parameters** (`fit_to_curve`): $\kappa$ and $\theta$ chosen so model yields match the bootstrapped zeros.

In the curve fit, $\sigma$ is held at its historical value. By Girsanov's theorem, changing from $P$ to $Q$ changes only the drift, so the diffusion coefficient is the same under both measures. With $\sigma$ left free, the optimiser picked volatilities of 20–100% on the 2023 curve to bend long yields through convexity. The difference between the $P$ and $Q$ drifts is the market price of interest-rate risk.

| Vasicek | 2017-06-30 | 2021-06-30 | 2023-06-30 |
|---|---|---|---|
| θ historical, P (%) | 0.74 | 0.36 | 1.91 |
| θ curve-implied, Q (%) | 3.19 | 3.01 | 3.57 |
| **θ_Q − θ_P (pp)** | **2.45** | **2.64** | **1.66** |
| κ historical, P | 0.12 | 0.11 | 0.10 |
| κ curve-implied, Q | 0.28 | 0.14 | 0.41 |

![Model vs market](figures/06_model_vs_market.png)

- **Historical parameters miss by 45–120 bp.** Their long-run level is dragged down by the zero-rate years, while the curve prices a long-run rate of about 3–3.6%.
- **Fitting the drift closes most of the gap, but not all of it.** A time-homogeneous one-factor model has one shape: yields move exponentially from $r_0$ toward a long-run level. It can't follow the 2021 curve's hump or the 2023 curve's front end, which rises before it falls, so it misses those by 11–24 bp.
- **Vasicek and CIR fitted to the curve are almost indistinguishable** at these volatilities.

## Time-varying coefficients: Hull-White

$$dR(t) = (a(t) - b(t) R(t))\,dt + \sigma(t)\,d\widetilde{W}(t)$$

With constant mean reversion $b$ and volatility $\sigma$ and a time-varying target, choosing

$$a(t) = \frac{\partial f(0,t)}{\partial t} + b\,f(0,t) + \frac{\sigma^2}{2b}\left(1 - e^{-2bt}\right)$$

from the initial instantaneous forward curve $f(0,t)$ makes model bond prices equal today's discount curve at every maturity. Bond prices stay exponential-affine:

$$P(t,T) = A(t,T)\,e^{-B(T-t)\,r_t},\qquad \log A(t,T) = \log\frac{P(0,T)}{P(0,t)} + B(T-t)\,f(0,t) - \frac{\sigma^2}{4b}\left(1-e^{-2bt}\right)B(T-t)^2,$$

with $B(\tau) = (1-e^{-b\tau})/b$. At $t = 0$ this returns the input curve exactly.

### Implementation: fitting today's curve exactly

`hull_white.HullWhite` works as follows (the code's `a` is the mean-reversion speed $b$ here, and `theta(t)` is the target $a(t)$):
- **Forward curve:** it comes from the Nelson-Siegel fit, whose slope $\partial f/\partial t$ is available in closed form.
- **Dynamics:** $b$ and $\sigma$ come from the historical Vasicek calibration. The dynamics come from history and the drift from today's curve.
- **Exact simulation:** $r_t = x_t + \alpha(t)$, where $x$ is a zero-mean Ornstein-Uhlenbeck process and $\alpha(t) = f(0,t) + \frac{\sigma^2}{2b^2}(1-e^{-bt})^2$.
- **Check:** Monte Carlo bond prices (weekly grid, 10,000 antithetic paths) are compared with the fitted curve.

| date | b | σ (%) | max abs yield diff, 1–30y (bp) | 2 s.e. at 30y (bp) |
|---|---|---|---|---|
| 2017-06-30 | 0.116 | 0.723 | 0.37 | 0.47 |
| 2021-06-30 | 0.107 | 0.702 | 0.39 | 0.50 |
| 2023-06-30 | 0.105 | 0.708 | 0.41 | 0.52 |

![Hull-White](figures/07_hull_white.png)

Two things show up in the check:

1. **Weekly-grid bias at 1 year.** The antithetic standard error at 1 year is tiny (about 0.003 bp). So the trapezoid rule's deterministic bias on the weekly grid (about 0.02 bp, largest for the sharply bending 2023 forward curve) shows up there as several standard errors. A daily grid removes it.
2. **The cost of fitting exactly (left panel).** The target contains $\partial f(0,t)/\partial t$, so any sharp bend in the fitted forward curve gets amplified. The 2023 curve's steep front end (Nelson-Siegel $\lambda \approx 2.1$) implies a target that spikes near $t=0$ and dips below zero around one year. In practice the target is only as good as the smoothness of the input curve.

## Results

| RMSE vs market zero yields (bp) | 2017-06-30 (normal) | 2021-06-30 (steep, near zero) | 2023-06-30 (inverted) |
|---|---|---|---|
| Vasicek, historical parameters (P) | 120.1 | 101.7 | 45.1 |
| CIR, historical parameters (P) | 100.4 | 66.3 | 49.1 |
| Vasicek, drift fitted to curve (Q) | 7.5 | 11.2 | 24.4 |
| CIR, drift fitted to curve (Q) | 7.5 | 10.9 | 24.4 |
| Hull-White (b, σ historical; drift from curve) | 5.8 | 3.3 | 11.1 |

The answer to the question, in three steps:
1. **Historical dynamics alone don't price bonds.** The risk premium moves the curve-implied long-run rate up by 1.7–2.6 percentage points.
2. **A curve-fitted one-factor model gets the level and slope right but not the shape.** Humps and inversions need either a second factor or time-varying coefficients.
3. **Hull-White fits exactly by construction.** It moves the modelling error into the curve-fitting step: its remaining error is the Nelson-Siegel fit, dominated in 2023 by the 20-year anomaly.

## Model comparison

| Model | Distinguishing feature | Pro | Con | Here |
|---|---|---|---|---|
| Vasicek | Mean-reverting, constant volatility | Closed-form prices; exact AR(1) calibration | Allows negative rates; one curve shape | Calibrated to history and curves |
| CIR | Square-root volatility | Non-negative rates (Feller) | Harder to calibrate; can't fit zero or negative bills | Exact MLE and exact simulation |
| Black-Karasinski | Lognormal short rate | Positive rates | No closed-form bond prices | Simulated |
| Hull-White | Time-dependent target | Fits the initial curve exactly | Target inherits curve-fitting noise; needs options to calibrate b, σ | Fitted to the NS forward curve |
| Nelson-Siegel | Parametric yield curve | Intuitive level, slope, curvature | Static; not arbitrage-free dynamics | Fitted to bootstrapped zeros |

## Limitations

- **Hull-White calibration:** its $b$ and $\sigma$ would normally be calibrated to cap or swaption prices, which aren't freely available. Here they come from the T-bill history.
- **The short-rate proxy:** the 3-month T-bill includes a convenience yield, and CIR requires flooring the zero-rate observations.
- **The curve data:** CMT yields are smoothed par yields rather than traded bond prices, and each curve has only 11 quoted maturities.

## Extensions

- **Two-factor models** (G2++, two-factor Hull-White) for shapes one factor can't produce, such as the 2023 inversion.
- **Heath-Jarrow-Morton:** model the whole forward curve directly, with volatility functions from a PCA of curve changes.
- **PCA** of daily yield changes (level, slope, curvature) and how it shifts across regimes.
- **Diebold-Li** dynamic Nelson-Siegel forecasts against a random walk.
- **Derivatives:** calibrate Hull-White to swaptions and price bond options.

## References

- D. Duffie, *Dynamic Asset Pricing Theory*, 3rd ed., Chapter 7.
- S. Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*, Chapter 10.
- R. Lord, R. Koekkoek, D. van Dijk (2010), "A comparison of biased simulation schemes for stochastic volatility models", *Quantitative Finance* 10(2).
- F. Diebold, C. Li (2006), "Forecasting the term structure of government bond yields", *Journal of Econometrics* 130(2).
