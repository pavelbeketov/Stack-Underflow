"""Stack Overflow activity: posts per month, Google Trends and year-over-year change.

Input files next to this script:
  QueryResults.csv   - columns year, month, questions, answers (SQL export from the site)
  multiTimeline.csv  - Google Trends export, monthly
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).parent

# --- settings ---------------------------------------------------------------
DROP_LAST_MONTH = True  # the last month of both exports is almost always incomplete
YOY_FROM_YEAR = 2011  # the first years show hundreds of percent and flatten the panel
AGENTS_LAUNCH = pd.Timestamp("2026-06-10")  # Stack Overflow for Agents beta announcement
# ----------------------------------------------------------------------------

# --- Stack Overflow ---------------------------------------------------------
df = pd.read_csv(HERE / "QueryResults.csv")
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df = df.sort_values("date").reset_index(drop=True)
if DROP_LAST_MONTH:
    df = df.iloc[:-1]

# Was (questions + answers) / 2 under the label "Total": a real total is the sum.
df["total"] = df["questions"] + df["answers"]
# center=True: a trailing window would shift every bend about six months to the right.
df["total_smooth"] = df["total"].rolling(12, center=True).mean()


def like_for_like_yoy(series: pd.Series) -> pd.DataFrame:
    """Year-over-year change that compares each month with the same month a year earlier.

    Partial years (the first and the last) are not distorted: only months present
    in both years enter the sums.
    """
    previous = series.shift(1, freq=pd.DateOffset(years=1))
    pairs = pd.concat([series, previous], axis=1, keys=["current", "previous"]).dropna()
    out = pairs.groupby(pairs.index.year).agg(
        current=("current", "sum"),
        previous=("previous", "sum"),
        months=("current", "size"),
    )
    out["yoy"] = (out["current"] / out["previous"] - 1) * 100
    # Plot at mid-year so the point lines up with the monthly panel above.
    out["date"] = pd.to_datetime(out.index.astype(str) + "-07-01")
    return out[out.index >= YOY_FROM_YEAR]


annual = like_for_like_yoy(df.set_index("date")["questions"])

# --- Google Trends ----------------------------------------------------------
trends = pd.read_csv(
    HERE / "multiTimeline.csv",
    skiprows=2,
    header=0,
    names=["date", "interest"],
    encoding="utf-8-sig",
)
trends["date"] = pd.to_datetime(trends["date"], format="%Y-%m")
trends["interest"] = pd.to_numeric(trends["interest"], errors="coerce")  # "<1" -> NaN
trends = trends.sort_values("date").reset_index(drop=True)
if DROP_LAST_MONTH:
    trends = trends.iloc[:-1]
peak_date = trends.loc[trends["interest"].idxmax(), "date"]

# --- figure -----------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, (ax_activity, ax_yoy) = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

sns.lineplot(data=df, x="date", y="questions", label="Questions", color="blue", ax=ax_activity)
sns.lineplot(data=df, x="date", y="answers", label="Answers", color="green", ax=ax_activity)
sns.lineplot(
    data=df,
    x="date",
    y="total_smooth",
    label="Questions + answers (12-month centered mean)",
    color="darkred",
    linewidth=2.5,
    ax=ax_activity,
)

ax_trends = ax_activity.twinx()
sns.lineplot(
    data=trends,
    x="date",
    y="interest",
    label="Stack Overflow (Google Trends)",
    color="teal",
    linestyle="-.",
    linewidth=2,
    legend=False,
    ax=ax_trends,
)
ax_trends.set_ylim(0, 100)
ax_trends.set_ylabel(f"Search interest (100 = {peak_date:%b %Y})", color="teal")
ax_trends.tick_params(axis="y", colors="teal")
ax_trends.grid(False)

# Year-over-year panel: filled markers = full years, hollow = partial years.
full_years = annual[annual["months"] == 12]
partial_years = annual[annual["months"] < 12]
ax_yoy.plot(annual["date"], annual["yoy"], color="purple", linewidth=1.5)
ax_yoy.scatter(full_years["date"], full_years["yoy"], color="purple", zorder=3, label="Questions YoY")
if not partial_years.empty:
    ax_yoy.scatter(
        partial_years["date"],
        partial_years["yoy"],
        facecolors="white",
        edgecolors="purple",
        zorder=3,
        label="Partial year (same months only)",
    )
    for _, row in partial_years.iterrows():
        ax_yoy.annotate(
            f"{int(row['months'])}/12 months",
            (row["date"], row["yoy"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )
ax_yoy.axhline(0, color="gray", linestyle="--", linewidth=1)
ax_yoy.set_title("Year-over-year change in questions (same months compared)")
ax_yoy.set_xlabel("Year")
ax_yoy.set_ylabel("Change from previous year")
ax_yoy.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax_yoy.grid(alpha=0.3)
ax_yoy.legend(loc="lower left", fontsize=9)

# Event lines.
events = [
    (pd.Timestamp("2008-09-15"), "green", 1.5, "Stack Overflow Public Launch"),
    (pd.Timestamp("2020-03-11"), "orange", 2, "COVID-19 Emergency"),
    (pd.Timestamp("2022-11-30"), "red", 2, "ChatGPT Launch"),
    (AGENTS_LAUNCH, "gray", 1.5, "Stack Overflow for Agents (beta)"),
]
for date, color, width, label in events:
    if date <= df["date"].max():
        ax_activity.axvline(date, color=color, linestyle="--", linewidth=width, label=label)

ax_activity.text(
    pd.Timestamp("2022-11-30"),
    ax_activity.get_ylim()[1] * 0.9,
    "ChatGPT Launch",
    rotation=90,
    color="red",
    va="top",
)
ax_activity.set_title(
    f"Stack Overflow Activity ({df['date'].dt.year.min()}-{df['date'].dt.year.max()})"
)
ax_activity.set_xlabel("")
ax_activity.set_ylabel("Posts per month")
ax_activity.grid(alpha=0.3)
activity_handles, activity_labels = ax_activity.get_legend_handles_labels()
trends_handles, trends_labels = ax_trends.get_legend_handles_labels()
ax_activity.legend(
    activity_handles + trends_handles,
    activity_labels + trends_labels,
    loc="lower left",
    bbox_to_anchor=(0, 1.08),
    ncol=3,
    fontsize=9,
)

# Footnotes.
notes = [
    "Google Trends: monthly values of the search term (not the topic). "
    "A one-week spike in early Apr 2026 (cause unknown) is averaged into its month."
]
if DROP_LAST_MONTH:
    notes.append("The last, incomplete month is dropped from both datasets.")
if not partial_years.empty:
    partial_text = ", ".join(f"{year} ({int(m)}/12 months)" for year, m in partial_years["months"].items())
    notes.append(f"YoY compares the same months of consecutive years. Partial years: {partial_text}.")
fig.text(0.5, 0.012, "\n".join(notes), ha="center", fontsize=9)

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "stack_overflow_activity.png", dpi=150, bbox_inches="tight")
plt.show()
