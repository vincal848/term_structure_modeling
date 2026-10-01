# Theory

*Continuously adding to this while reading the associated texts.*

This project came out of wanting to work with term structures of interest
rates and the various methods, tools, and processes I had seen used in other
research publications.

My primary reference for understanding and construction was Chapter 7 of
*Dynamic Asset Pricing Theory* by Darrell Duffie, titled "Term-Structure
Models." I found that chapter interesting for its expansion on factors beyond
the initially provided models. I supplemented it with Chapter 10 of *Stochastic
Calculus for Finance II* by Steven Shreve, titled the same as the Duffie
chapter.

When considering the structure of the model, the market has a yield *curve*
rather than a single rate. Below is a summary from section 10.1 of the Shreve
text, which I found more informative on the affine models before jumping into
the Heath-Jarrow-Morton framework. The two differ in focus: the affine models
operate bottom-up, starting with short rate changes that evolve toward the
yield curve, while the HJM framework works top-down, using forward rates
directly.

## Bootstrapping and yield curves

Consider a single zero-coupon bond paying 1 at maturity, with price at each
period (starting at 0) described by $B(0,T_j)$, where $T$ is a set of dates
such that $0=T_0 < T_1 < T_2 < ... < T_n$, and at each period $T_i$ we receive
a coupon payment $C$. These are fixed payments $C_1,C_2,...,C_j$ that operate
as interest payments, with $C_j$ also including the principal. The price at
time zero is then:

$$\sum_{i=1}^j C_i B(0,T_i)$$

Using the passage of time and payment information, we can recursively recover
the zero-coupon bond prices from 0 to $n$: the price of the bond maturing at
the next period $T_1$ represents the price of the $T_1$-maturity bond over the
payment at time $T_1$. Applying this iteratively up to $n$ is *bootstrapping* —
recovering zero-coupon bond prices from coupon-paying bond prices.

The zero-coupon bond price, where the yield is a continuous compounding of the
interest rate over the bond's lifetime, is:

$$\text{price of zero-coupon bond} = \text{face value} \times e^{-\text{yield} \times t_{\text{maturity}}}$$

From this there is a developing yield curve rather than a single interest
rate — an interpolation of finite maturity-yield pairs observed from the
market. The interest rate is sometimes called the short rate, idealized as the
shortest-maturity yield or the overnight rate offered by the government.

Prices of zero-coupon bonds can also be derived from the risk-neutral pricing
formula for the affine yield models, which is the treatment below.

## Term structure

Set up a probability space $(\Omega,\mathscr{F},P)$ with a filtration
$$\mathbb{F} = \{ \mathscr{F}_t : 0 \leq t \leq T \}$$
of $B$, a standard Brownian motion of dimension $d \geq 1 \in \mathscr{R}^d$.

From the short rate $r$, we require $\int_0^T |r_t|\, dt < \infty$. At any
time $t$, investing a single unit achieves a future market value based on
$e^{\int_t^s r_u\, du}$, compounding as the unit reinvests continually at rate
$s$.

Assuming absence of arbitrage, there is a probability measure $Q$ such that any
security with a lump-sum dividend payment of $Z$ at $s$ has a price of

$$E_t^{Q} \left[ e^{\int_t^s -r_u\, du} \right] \times Z$$

where $E^Q$ denotes the $\mathscr{F}_t$-conditional expectation under $Q$ ($Z$
is $\mathscr{F}_t$-measurable, so this is well defined). Setting $Z=1$, the
price at $t$ of the zero-coupon bond maturing at $s$ is

$$\Lambda_{t,s} \equiv E_t^{Q} \left[ e^{\int_t^s -r_u\, du} \right]$$

This is the discount function, or *the term structure of interest rates*. The
term structure is usually expressed as a yield curve, where the continuously
compounding yield $y_{t,\tau}$ is defined by

$$y_{t, \tau} = - \frac{\log(\Lambda_{t, t + \tau})}{\tau}$$

which can also be represented via forward interest rates. In the models below,
the short rate is driven by the standard Brownian motion under $Q$ that comes
from Girsanov's theorem.

## One-factor term-structure models

$$dr_t = \mu(r_t, t)\, dt + \sigma(r_t, t)\, dB_t^{Q}$$

The short rate is the only factor the current yield curve depends on, so the
price can be written as a function of $t$ and $s$: $\Lambda_{t,s} =
F(t,s,r_t)$ for a fixed $F: [0,T] \times [0,T] \times \mathscr{R} \rightarrow
\mathscr{R}$.

Each model is a special case of the SDE:

$$dr_t = \left[ K_0(t) + K_1(t) r_t + K_2(t) r_t \log(r_t)\right] dt + \left[H_0(t) + H_1(t) r_t\right]^v dB_t^{Q}$$

where $K_0, K_1, K_2, H_0, H_1$ are continuous functions on $[0,T]$ and $v$
ranges from 0.5 to 1.5. Each model below has a different combination of these
coefficients and exponent $v$. CIR has non-zero $K_0, K_1, H_1$ with $v=0.5$.
Pearson-Sun is the same as CIR but also includes $H_0$, still with $v=0.5$. At
$v=1$ we have Dothan, Merton (Ho-Lee), Vasicek, and Black-Karasinski. At
$v=1.5$, the Constantinides-Ingersoll model.

#### $-K_1$: mean reversion

A negative value of $K_1$ acts as a mean-reversion parameter: high or low
short rates generate low or high drift, pulling the rate back toward a central
level.

### Time-varying coefficients

Some model differences come purely from whether coefficients are allowed to
vary with time. For example, the *Merton model* of the term structure is
called the *Ho-Lee model* once coefficients vary with time.

## Describing models

Affine models (linear-plus-constant) start with single-factor models. The
affine class has $K_2 = 0$ and $v = 0.5$, which includes Vasicek ($H_1=0$),
CIR ($H_0=0$), Merton/Ho-Lee ($K_1=H_1=0$), and Pearson-Sun.

### Vasicek interest rate model

$$dR(t) = (\alpha - \beta R(t))\, dt + \sigma\, dW(t)$$

where $R(t)$ is the interest rate process and $\alpha, \beta, \sigma$ are
positive constants.

### Hull-White model

$$dR(t) = (a(t) - b(t) R(t))\, dt + \sigma(t)\, d\widetilde{W}(t)$$

The time-varying mean-reversion level $a(t)$ (usually written $\theta(t)$ when
$b(t)$ is held constant) is what lets this model match an *entire* observed
curve rather than a single long-run rate — see `R/short_rate.R` and
`hull_white_bond_price()` in `R/bond_prices.R` for the constant-$a$,
constant-$\sigma$ version implemented here.

### Cox-Ingersoll-Ross (CIR) interest rate model

$$dR(t) = (a - b R(t))\, dt + \sigma \sqrt{R(t)}\, d\widetilde{W}(t)$$

The square-root diffusion term is what keeps the process non-negative: as
$R(t) \to 0$, the volatility vanishes, so an upward drift (when the Feller
condition $2ab \geq \sigma^2$ holds) is enough to prevent the process from
crossing zero. See `simulate_cir()`'s docstring in `R/short_rate.R` for how
that non-negativity is actually enforced in simulation without the clipping
bias the original code had.

## Two-factor models

Not implemented here. The natural next step from the one-factor models above
is a second state variable — for example Longstaff-Schwartz (short rate and
its volatility) or a second mean-reverting factor added to CIR or Vasicek
(as in the Hull-White two-factor model) — to let the curve both shift and
change shape, which a one-factor model cannot do: every maturity is driven by
the same single Brownian motion, so all points on the curve are perfectly
correlated by construction. See "Future interests" in the README.

## Heath-Jarrow-Morton framework

Not implemented here. Rather than modeling the short rate and deriving the
rest of the curve from it, HJM directly specifies the dynamics of the entire
instantaneous forward-rate curve $f(t,T)$:

$$df(t,T) = \alpha(t,T)\, dt + \sigma(t,T)\, dB_t^Q$$

Under the risk-neutral measure, absence of arbitrage pins the drift to the
volatility through the HJM drift condition, $\alpha(t,T) = \sigma(t,T)
\int_t^T \sigma(t,u)\, du$ (for a one-dimensional driver), so only $\sigma$
can be specified freely. This is the top-down counterpart to the bottom-up
affine models above, and several of them (Ho-Lee, Hull-White) can be recovered
as special cases of HJM with a particular choice of $\sigma(t,T)$.
