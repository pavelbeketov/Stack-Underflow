"""When did the decline in Stack Overflow posts bend? Segmented regression on log(posts).

Model:  log(y_t) = a + b*t + month-of-year effects + d*max(0, t - k)
  b - monthly trend before the bend, d - change of that trend after it,
  k - bend date. ChatGPT (Dec 2022) is tested, and every other month is scanned
  as a competing bend date.

This shows WHEN the trend changed, not WHY. A causal estimate needs a control group,
for example the same series for a Q&A site that ChatGPT affects less (Math Stack Exchange).

Input next to this script: QueryResults.csv with columns year, month, questions, answers.
Only numpy, pandas and matplotlib are required.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# --- settings ---------------------------------------------------------------
SERIES = "questions"  # or "answers"
START = pd.Timestamp("2015-01-01")  # after the peak, without the growth phase
CHATGPT_MONTH = pd.Timestamp("2022-12-01")  # first full month after the 30 Nov 2022 launch
DROP_LAST_MONTH = True  # the last month of the export is almost always incomplete
HAC_LAGS = 12  # Newey-West lags for the standard error
MARGIN = 24  # months at both ends skipped by the scan (a bend needs data on both sides)
# ----------------------------------------------------------------------------

df = pd.read_csv(HERE / "QueryResults.csv")
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df = df.sort_values("date").reset_index(drop=True)
if DROP_LAST_MONTH:
    df = df.iloc[:-1]
df = df[df["date"] >= START].reset_index(drop=True)

matches = df.index[df["date"] == CHATGPT_MONTH]
if len(matches) == 0:
    raise SystemExit(f"{CHATGPT_MONTH:%Y-%m} is not inside the data range.")
k_chatgpt = float(matches[0])

y = np.log(df[SERIES].to_numpy(dtype=float))
t = np.arange(len(df), dtype=float)
seasonal = pd.get_dummies(df["date"].dt.month, drop_first=True).to_numpy(dtype=float)
base = np.column_stack([np.ones(len(t)), t, seasonal])


def fit(kink: float):
    """OLS with a bend at position `kink` (months from START)."""
    x = np.column_stack([base, np.maximum(0.0, t - kink)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return beta, x, resid, float(resid @ resid)


def newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Heteroskedasticity- and autocorrelation-robust standard errors (Bartlett kernel)."""
    xu = x * resid[:, None]
    s = xu.T @ xu
    for lag in range(1, lags + 1):
        gamma = xu[lag:].T @ xu[:-lag]
        s += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    bread = np.linalg.inv(x.T @ x)
    return np.sqrt(np.diag(bread @ s @ bread))


def per_year(monthly_log_slope: float) -> float:
    return (np.exp(12 * monthly_log_slope) - 1) * 100


# --- bend at ChatGPT --------------------------------------------------------
beta, x, resid, ssr_chatgpt = fit(k_chatgpt)
se = newey_west_se(x, resid, HAC_LAGS)
before = beta[1]
change = beta[-1]
print(f"Series: {SERIES}, {df['date'].min():%Y-%m} .. {df['date'].max():%Y-%m} ({len(df)} months)")
print(f"Bend at {CHATGPT_MONTH:%Y-%m} (ChatGPT):")
print(f"  trend before the bend: {per_year(before):+.1f}% per year")
print(f"  trend after the bend:  {per_year(before + change):+.1f}% per year")
print(f"  slope change: {change:+.4f} log-points per month, t = {change / se[-1]:.1f} "
      f"(Newey-West, {HAC_LAGS} lags)")

# --- scan of all candidate bend dates ---------------------------------------
candidates = range(MARGIN, len(df) - MARGIN)
scan = pd.DataFrame(
    {
        "date": [df["date"].iloc[k] for k in candidates],
        "ssr": [fit(float(k))[3] for k in candidates],
    }
)
scan["relative_ssr"] = scan["ssr"] / scan["ssr"].min()
rank = int((scan["ssr"] < ssr_chatgpt).sum()) + 1
print(f"\nBest-fitting bend dates (relative_ssr 1.00 = best fit):")
print(scan.nsmallest(5, "ssr")[["date", "relative_ssr"]].to_string(index=False, float_format="{:.3f}".format))
print(f"ChatGPT month ranks {rank} of {len(scan)} candidate months "
      f"(relative_ssr {ssr_chatgpt / scan['ssr'].min():.3f}).")

# --- naive counterfactual: the trend before ChatGPT continued -----------------
pre = t < k_chatgpt
beta_pre, *_ = np.linalg.lstsq(base[pre], y[pre], rcond=None)
seasonal_mean = float((seasonal @ beta[2:-1]).mean())
seasonal_mean_pre = float((seasonal @ beta_pre[2:]).mean())
fitted_trend = np.exp(beta[0] + beta[1] * t + beta[-1] * np.maximum(0.0, t - k_chatgpt) + seasonal_mean)
counterfactual = np.exp(beta_pre[0] + beta_pre[1] * t + seasonal_mean_pre)
share = df[SERIES].iloc[-1] / counterfactual[-1] * 100
print(f"\nThe last month is {share:.1f}% of the continued pre-ChatGPT trend "
      f"(a naive extrapolation, not a causal estimate).")

# --- figure -----------------------------------------------------------------
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)
ax_top.plot(df["date"], df[SERIES], color="blue", linewidth=1.2, label=f"{SERIES.capitalize()} per month")
ax_top.plot(df["date"], fitted_trend, color="black", linewidth=2,
            label=f"Fitted trend with a bend at {CHATGPT_MONTH:%b %Y}")
ax_top.plot(df["date"], counterfactual, color="gray", linestyle="--", linewidth=2,
            label="Pre-bend trend continued (naive)")
ax_top.axvline(pd.Timestamp("2022-11-30"), color="red", linestyle="--", linewidth=1.5)
ax_top.set_yscale("log")
ax_top.set_ylabel("Posts per month (log scale)")
ax_top.set_title(f"Stack Overflow {SERIES}: trend before and after ChatGPT")
ax_top.legend(loc="lower left", fontsize=9)
ax_top.grid(alpha=0.3)

ax_bottom.plot(scan["date"], scan["relative_ssr"], color="purple")
ax_bottom.axvline(pd.Timestamp("2022-11-30"), color="red", linestyle="--", linewidth=1.5)
ax_bottom.set_yscale("log")
ax_bottom.set_ylabel("Misfit by bend date\n(1 = best, log scale)")
ax_bottom.set_xlabel("Candidate bend date")
ax_bottom.grid(alpha=0.3)

fig.text(0.5, 0.005, "Red line: ChatGPT launch (30 Nov 2022). Timing evidence only, not proof of cause.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "stack_overflow_break_test.png", dpi=150, bbox_inches="tight")
plt.show()
