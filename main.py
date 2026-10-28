from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

df = pd.read_csv(Path(__file__).with_name("QueryResults.csv"))
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df["total"] = (df["questions"] + df["answers"]) / 2

df = df.sort_values("date")
df["rolling_12"] = df["total"].rolling(12).mean()

annual = df.groupby("year")[["questions", "answers"]].sum()
annual["questions_yoy"] = annual["questions"].pct_change() * 100
annual["date"] = pd.to_datetime(annual.index.astype(str), format="%Y")

trends = pd.read_csv(
    Path(__file__).with_name("multiTimeline.csv"),
    skiprows=2,
    header=0,
    names=["date", "interest"],
    encoding="utf-8-sig",
)
trends["date"] = pd.to_datetime(trends["date"], format="%Y-%m")
trends["interest"] = pd.to_numeric(trends["interest"])
trends = trends.sort_values("date")

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
    y="rolling_12",
    label="Total",
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
ax_trends.set_ylabel("Search interest (0–100)", color="teal")
ax_trends.tick_params(axis="y", colors="teal")
ax_trends.grid(False)

sns.lineplot(
    data=annual,
    x="date",
    y="questions_yoy",
    marker="o",
    color="purple",
    label="Questions YoY",
    ax=ax_yoy,
)
ax_yoy.axhline(0, color="gray", linestyle="--", linewidth=1)
ax_yoy.set_title("Year-over-year change in questions")
ax_yoy.set_xlabel("Year")
ax_yoy.set_ylabel("Change from previous year")
ax_yoy.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax_yoy.grid(alpha=0.3)

# Вертикальные линии событий
ax_activity.axvline(
    pd.Timestamp("2008-09-15"),
    color="green",
    linestyle="--",
    label="Stack Overflow Public Launch"
)

ax_activity.axvline(
    pd.Timestamp("2022-11-30"),
    color="red",
    linestyle="--",
    linewidth=2,
    label="ChatGPT Launch"
)
ax_activity.text(
    pd.Timestamp("2022-11-30"),
    ax_activity.get_ylim()[1] * 0.9,
    "ChatGPT Launch",
    rotation=90,
    color="red",
    va="top",
)
ax_activity.axvline(
    pd.Timestamp("2020-03-11"),
    color="orange",
    linestyle="--",
    linewidth=2,
    label="COVID-19 Emergency"
)
ax_activity.set_title(
    f"Stack Overflow Activity ({df['date'].dt.year.min()}–{df['date'].dt.year.max()})"
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

month_counts = df.groupby("year")["month"].nunique()
partial_years = month_counts[month_counts < 12]
if not partial_years.empty:
    partial_year_note = ", ".join(
        f"{year} ({months}/12 months)" for year, months in partial_years.items()
    )
    fig.text(
        0.5,
        0.015,
        f"YoY compares recorded annual totals. Partial years: {partial_year_note}.",
        ha="center",
        fontsize=9,
    )

fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()
