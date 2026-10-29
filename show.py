import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df1 = pd.read_csv('./active_authors/QueryResults (1).csv')
df2 = pd.read_csv('./active_authors/QueryResults (2).csv')
df3 = pd.read_csv('./active_authors/QueryResults (3).csv')
df4 = pd.read_csv('./active_authors/QueryResults (4).csv')
df5 = pd.read_csv('./active_authors/QueryResults (5).csv')
df6 = pd.read_csv('./active_authors/QueryResults (6).csv')
df7 = pd.read_csv('./active_authors/QueryResults (7).csv')
df8 = pd.read_csv('./active_authors/QueryResults (8).csv')
df9 = pd.read_csv('./active_authors/QueryResults (9).csv')
df10 = pd.read_csv('./active_authors/QueryResults (10).csv')
df11 = pd.read_csv('./active_authors/QueryResults (11).csv')
df12 = pd.read_csv('./active_authors/QueryResults (12).csv')
df13 = pd.read_csv('./active_authors/QueryResults (13).csv')
df14 = pd.read_csv('./active_authors/QueryResults (14).csv')
df15 = pd.read_csv('./active_authors/QueryResults (15).csv')
df16 = pd.read_csv('./active_authors/QueryResults (16).csv')
df17 = pd.read_csv('./active_authors/QueryResults (17).csv')

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17 ], ignore_index=True)
df["month"] = pd.to_datetime(df["month"])
plt.figure(figsize=(12, 6))


# Построение
sns.lineplot(
    data=df,
    x="month",
    y="active_authors",
    # marker="o",
    color="royalblue",
    linewidth=2
)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Оформление
plt.title("Stack Overflow Active Authors by Year")
plt.xlabel("Year")
plt.ylabel("Active Authors")
plt.tight_layout()

# plt.show()

df["month"] = pd.to_datetime(df["month"])
df["year"] = df["month"].dt.year.astype(str)
df["month"] = df["month"].dt.month.astype(str)

df = df[["year", "month", "active_authors"]]
df = df.rename(columns={"active_authors": "authors"})

df.to_csv("merged.csv", index=False, quoting=1)




print(df)