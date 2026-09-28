"""Run the full analysis: figures go to figures/, tables to results/ (markdown).

Every number and figure in the README comes from this script, using the cached
FRED data in data/. Usage: python scripts/run_analysis.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from termstructure import curves, data  # noqa: E402
from termstructure.affine import CIR, Vasicek  # noqa: E402
from termstructure.calibrate import (  # noqa: E402
    fit_cir_mle,
    fit_to_curve,
    fit_vasicek_ar1,
    tbill_to_continuous,
    weekly,
)
from termstructure.hull_white import HullWhite  # noqa: E402
from termstructure.short_rate import black_karasinski, dothan, mc_bond_curve  # noqa: E402

FIG = ROOT / "figures"
RESULTS = ROOT / "results"
BP = 1e4
DATES = {"2017-06-30 (normal)": "2017-06-30",
         "2021-06-30 (steep, near zero)": "2021-06-30",
         "2023-06-30 (inverted)": "2023-06-30"}
TAU_FINE = np.linspace(1 / 12, 30, 400)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table writer (avoids a tabulate dependency)."""
    df = df.reset_index()
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def save_table(df, name):
    (RESULTS / f"{name}.md").write_text(to_markdown(df) + "\n", encoding="utf-8")
    print(f"\n## {name}\n{to_markdown(df)}")


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def quoted_mask(taus, curve):
    return np.isin(np.round(taus, 6), np.round(curve.index.values, 6))


# --- 1-2. Bootstrapping and Nelson-Siegel ------------------------------------


def curves_section(cmt):
    boot, ns_fits = {}, {}
    for label, d in DATES.items():
        curve = data.curve_on(d, cmt)
        taus, zeros = curves.bootstrap_zeros(curve.index.values, curve.values)
        boot[label] = (curve, taus, zeros)
        q = quoted_mask(taus, curve)
        ns_fits[label] = curves.fit_nelson_siegel(taus[q], zeros[q])

    par = pd.DataFrame({label: data.curve_on(d, cmt) * 100 for label, d in DATES.items()})
    par.index = [f"{m:g}y" if m >= 1 else f"{round(m * 12)}m" for m in par.index]
    save_table(par.round(2).rename_axis("maturity"), "01_par_curves")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, (curve, taus, zeros)) in zip(axes, boot.items()):
        ax.plot(curve.index, curve.values * 100, "o", label="Par yield (CMT)")
        ax.plot(taus, zeros * 100, "-", label="Bootstrapped zero (cont.)")
        ax.set(title=label, xlabel="Maturity (years)", ylabel="Yield (%)")
    axes[0].legend()
    save_fig(fig, "01_bootstrap.png")

    ns = pd.DataFrame({label: {"β0 level (%)": f.b0 * 100, "β1 slope (%)": f.b1 * 100,
                               "β2 curvature (%)": f.b2 * 100, "λ": f.lam, "fit RMSE (bp)": f.rmse * BP}
                       for label, f in ns_fits.items()}).round(2)
    save_table(ns.rename_axis("parameter"), "02_nelson_siegel")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, (curve, taus, zeros)) in zip(axes, boot.items()):
        fit, q = ns_fits[label], quoted_mask(taus, curve)
        ax.plot(taus[q], zeros[q] * 100, "o", label="Bootstrapped zero")
        ax.plot(TAU_FINE, fit.zero(TAU_FINE) * 100, "-", label="Nelson-Siegel zero")
        ax.plot(TAU_FINE, fit.forward(TAU_FINE) * 100, "--", label="NS instantaneous forward")
        ax.set(title=f"{label}\nRMSE {fit.rmse * BP:.1f} bp", xlabel="Maturity (years)", ylabel="Rate (%)")
    axes[0].legend()
    save_fig(fig, "02_nelson_siegel.png")
    return boot, ns_fits


# --- 3-4. One-factor family, Monte Carlo vs closed form ----------------------


def simulation_section():
    toy = {  # parameters from the original R Markdown version
        "Vasicek": (Vasicek(0.1, 0.05, 0.02).general(), "euler"),
        "CIR": (CIR(0.1, 0.05, 0.02).general(), "full_truncation"),
        "Black-Karasinski": (black_karasinski(0.1, np.log(0.05), 0.02), "log_euler"),
        "Dothan": (dothan(0.0, 0.2), "log_euler"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, (name, (model, scheme)) in zip(axes, toy.items()):
        t, paths = model.simulate(0.03, 1.0, 1 / 252, 20, seed=123, scheme=scheme, antithetic=False)
        ax.plot(t, paths * 100, lw=0.8, alpha=0.8)
        ax.set(title=name, xlabel="Years")
    axes[0].set_ylabel("Short rate (%)")
    save_fig(fig, "03_one_factor_family.png")

    r0 = 0.02
    maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    models = {"Vasicek": Vasicek(0.3, 0.04, 0.01), "CIR": CIR(0.3, 0.04, 0.05)}
    rows, mc_yields = [], {}
    for name, m in models.items():
        t, paths = m.simulate_exact(r0, 30.0, 1 / 52, 10_000, seed=11)
        prices, se = mc_bond_curve(t, paths, maturities, antithetic=(name == "Vasicek"))
        exact = m.bond_price(r0, maturities)
        mc_yields[name] = (-np.log(prices) / maturities, se / (prices * maturities))
        for T, p, s, e in zip(maturities, prices, se, exact):
            rows.append({"model": name, "T (years)": int(T), "closed form": f"{e:.5f}", "Monte Carlo": f"{p:.5f}",
                         "std err": f"{s:.5f}", "abs diff / s.e.": f"{abs(p - e) / s:.2f}",
                         "yield diff (bp)": f"{(np.log(e) - np.log(p)) / T * BP:.2f}"})
    save_table(pd.DataFrame(rows).set_index("model"), "03_mc_vs_closed_form")

    tau = np.linspace(0.25, 30, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, m) in zip(axes, models.items()):
        y, y_se = mc_yields[name]
        ax.plot(tau, m.zero_yield(r0, tau) * 100, label="Closed form")
        ax.errorbar(maturities, y * 100, yerr=2 * y_se * 100, fmt="o", capsize=3, label="Monte Carlo ± 2 s.e.")
        ax.axhline(m.long_yield * 100, ls=":", c="gray", label="Long-run yield")
        ax.set(title=f"{name}: r0 = 2%, θ = 4%", xlabel="Maturity (years)")
    axes[0].set_ylabel("Zero yield (%)")
    axes[0].legend()
    save_fig(fig, "04_mc_vs_closed_form.png")

    cir_nf, r0_nf, T = CIR(0.3, 0.04, 0.2), 0.01, 10.0
    exact_yield = cir_nf.zero_yield(r0_nf, T)
    rows = []
    for step, dt in [("monthly", 1 / 12), ("weekly", 1 / 52)]:
        runs = [(s, *cir_nf.general().simulate(r0_nf, T, dt, 20_000, seed=5, scheme=s), True)
                for s in ("euler_floor", "full_truncation")]
        runs.append(("exact (noncentral χ²)", *cir_nf.simulate_exact(r0_nf, T, dt, 20_000, seed=5), False))
        for scheme, t, paths, anti in runs:
            p, s = mc_bond_curve(t, paths, [T], antithetic=anti)
            rows.append({"scheme": scheme, "step": step, "share of steps at 0": f"{(paths[1:] == 0).mean():.3f}",
                         "10y yield error (bp)": f"{(-np.log(p[0]) / T - exact_yield) * BP:.1f}",
                         "± 2 s.e. (bp)": f"{2 * s[0] / (p[0] * T) * BP:.1f}"})
    save_table(pd.DataFrame(rows).set_index("scheme"), "04_cir_schemes")


# --- 5-7. Calibration and Hull-White -----------------------------------------


def calibration_section(boot, ns_fits):
    raw = data.load_short_rate()
    short = weekly(pd.Series(tbill_to_continuous(raw), index=raw.index))
    p_fits = {label: (fit_vasicek_ar1(short[:d]), fit_cir_mle(short[:d])) for label, d in DATES.items()}

    def pm(v, se, scale=1, digits=3):
        return f"{v * scale:.{digits}f} ± {se * scale:.{digits}f}"

    table = pd.DataFrame({label: {
        "Vasicek κ": pm(v.model.kappa, v.std_errors["kappa"]),
        "Vasicek θ (%)": pm(v.model.theta, v.std_errors["theta"], 100, 2),
        "Vasicek σ (%)": pm(v.model.sigma, v.std_errors["sigma"], 100, 3),
        "half-life (years)": f"{v.half_life:.1f}",
        "CIR κ": pm(c.model.kappa, c.std_errors["kappa"]),
        "CIR θ (%)": pm(c.model.theta, c.std_errors["theta"], 100, 2),
        "CIR σ": pm(c.model.sigma, c.std_errors["sigma"]),
        "CIR Feller (2κθ ≥ σ²)": "yes" if c.model.feller else "no",
        "weekly observations": v.n_obs,
    } for label, (v, c) in p_fits.items()})
    save_table(table.rename_axis("expanding window 1990 → date"), "05_historical_calibration")

    rows = {}
    for start, end in [("2007-06-30", "2017-06-30"), ("2011-06-30", "2021-06-30"), ("2013-06-30", "2023-06-30")]:
        window = f"{start[:4]}–{end[:4]}"
        try:
            v = fit_vasicek_ar1(short[start:end])
            rows[window] = {"κ": f"{v.model.kappa:.3f}", "θ (%)": f"{v.model.theta * 100:.2f}",
                            "half-life (years)": f"{v.half_life:.1f}"}
        except ValueError:
            rows[window] = {"κ": "AR(1) b > 1", "θ (%)": "–", "half-life (years)": "no mean reversion"}
    save_table(pd.DataFrame(rows).T.rename_axis("10-year window"), "06_rolling_windows")

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(short.index, short * 100, lw=0.8, label="3M T-bill (continuous)")
    for (label, d), color in zip(DATES.items(), ["C1", "C2", "C3"]):
        v = p_fits[label][0].model
        ax.hlines(v.theta * 100, short.index[0], pd.Timestamp(d), colors=color, linestyles="--",
                  label=f"Vasicek θ fitted 1990–{d[:4]}: {v.theta * 100:.2f}%")
        ax.axvline(pd.Timestamp(d), color=color, lw=0.8)
    ax.set(ylabel="Rate (%)", title="Short-rate history and expanding-window long-run means (measure P)")
    ax.legend(fontsize=8, loc="upper right")
    save_fig(fig, "05_short_rate_history.png")

    q_fits, errors = {}, {}
    for label, (curve, taus, zeros) in boot.items():
        q = quoted_mask(taus, curve)
        tq, zq, r0 = taus[q], zeros[q], zeros[0]
        v_p, c_p = (f.model for f in p_fits[label])
        q_fits[label] = {"Vasicek": fit_to_curve(Vasicek, tq, zq, r0, v_p.sigma),
                         "CIR": fit_to_curve(CIR, tq, zq, r0, c_p.sigma)}

        def rmse(y):
            return np.sqrt(np.mean((y - zq) ** 2)) * BP

        errors[label] = {"Vasicek, historical parameters (P)": rmse(v_p.zero_yield(r0, tq)),
                         "CIR, historical parameters (P)": rmse(c_p.zero_yield(max(r0, 1e-4), tq)),
                         "Vasicek, drift fitted to curve (Q)": q_fits[label]["Vasicek"].rmse_bp,
                         "CIR, drift fitted to curve (Q)": q_fits[label]["CIR"].rmse_bp}

    pq = pd.DataFrame({label: {"θ historical, P (%)": p_fits[label][0].model.theta * 100,
                               "θ curve-implied, Q (%)": q["Vasicek"].model.theta * 100,
                               "θ_Q − θ_P (pp)": (q["Vasicek"].model.theta - p_fits[label][0].model.theta) * 100,
                               "κ historical, P": p_fits[label][0].model.kappa,
                               "κ curve-implied, Q": q["Vasicek"].model.kappa}
                       for label, q in q_fits.items()}).round(2)
    save_table(pq.rename_axis("Vasicek"), "07_p_vs_q")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (label, (curve, taus, zeros)) in zip(axes, boot.items()):
        q, r0 = quoted_mask(taus, curve), zeros[0]
        ax.plot(taus[q], zeros[q] * 100, "ko", ms=4, label="Market zero (bootstrapped)")
        ax.plot(TAU_FINE, p_fits[label][0].model.zero_yield(r0, TAU_FINE) * 100, "--", label="Vasicek, historical P")
        ax.plot(TAU_FINE, q_fits[label]["Vasicek"].model.zero_yield(r0, TAU_FINE) * 100, "-",
                label="Vasicek, curve-fitted Q")
        ax.plot(TAU_FINE, q_fits[label]["CIR"].model.zero_yield(max(r0, 1e-4), TAU_FINE) * 100, ":",
                label="CIR, curve-fitted Q")
        ax.set(title=label, xlabel="Maturity (years)", ylabel="Zero yield (%)")
    axes[0].legend(fontsize=8)
    save_fig(fig, "06_model_vs_market.png")

    hw_models = {label: HullWhite(p_fits[label][0].model.kappa, p_fits[label][0].model.sigma, ns_fits[label])
                 for label in DATES}
    check_T = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    rows, hw_mc = [], {}
    for label, hw in hw_models.items():
        t, paths = hw.simulate_exact(30.0, 1 / 52, 10_000, seed=21)
        prices, se = mc_bond_curve(t, paths, check_T)
        target = hw.curve.discount(check_T)
        hw_mc[label] = (-np.log(prices) / check_T, se / (prices * check_T))
        diff = (np.log(target) - np.log(prices)) / check_T * BP
        rows.append({"date": label, "a": f"{hw.a:.3f}", "σ (%)": f"{hw.sigma * 100:.3f}",
                     "max abs yield diff, 1–30y (bp)": f"{np.max(np.abs(diff)):.2f}",
                     "2 s.e. at 30y (bp)": f"{2 * hw_mc[label][1][-1] * BP:.2f}"})
    save_table(pd.DataFrame(rows).set_index("date"), "08_hull_white_check")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    t_grid = np.linspace(0, 30, 300)
    for label, hw in hw_models.items():
        axes[0].plot(t_grid, hw.theta(t_grid) / hw.a * 100, label=label)
    axes[0].set(title="Hull-White target θ(t)/a implied by each curve", xlabel="t (years)", ylabel="%")
    axes[0].legend(fontsize=8)
    label = "2023-06-30 (inverted)"
    curve, taus, zeros = boot[label]
    q = quoted_mask(taus, curve)
    y, y_se = hw_mc[label]
    axes[1].plot(taus[q], zeros[q] * 100, "ko", ms=4, label="Market zero")
    axes[1].plot(TAU_FINE, ns_fits[label].zero(TAU_FINE) * 100, label="Fitted curve (NS)")
    axes[1].errorbar(check_T, y * 100, yerr=2 * y_se * 100, fmt="s", ms=4, capsize=3,
                     label="Hull-White Monte Carlo ± 2 s.e.")
    axes[1].plot(TAU_FINE, q_fits[label]["Vasicek"].model.zero_yield(zeros[0], TAU_FINE) * 100, ":",
                 label="Vasicek, curve-fitted Q")
    axes[1].set(title=f"{label}: Hull-White reprices the curve", xlabel="Maturity (years)", ylabel="Zero yield (%)")
    axes[1].legend(fontsize=8)
    save_fig(fig, "07_hull_white.png")

    results = pd.DataFrame(errors)
    results.loc["Hull-White (a, σ historical; θ(t) from curve)"] = [ns_fits[lb].rmse * BP for lb in DATES]
    save_table(results.round(1).rename_axis("RMSE vs market zero yields (bp)"), "09_results")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    cmt = data.load_cmt()
    boot, ns_fits = curves_section(cmt)
    simulation_section()
    calibration_section(boot, ns_fits)
    print(f"\nFigures in {FIG}, tables in {RESULTS}")
