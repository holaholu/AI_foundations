

#
grouped = df.groupby("column_name")

for name, group in grouped:
    print(name)
    print(group)
    
grouped.mean()
grouped.sum()

df.groupby("category_column")["numeric_column"].mean() # group by category column and calculate mean of numeric column
df.groupby("category_column").agg({"numeric_column":["mean", "max", "min"]}) # group by category column and calculate mean, max, min of numeric column

pivot = df.pivot_table(
    values="numeric_column",
    index="category_column",
    aggfunc="mean"
)

def range_func(x):
    return x.max() - x.min()

df.groupby("category_column")["numeric_column"].agg(range_func)

df.groupby("category_column")["numeric_column"].mean()
df.groupby("category_column")["numeric_column"].max()
df.groupby("category_column")["numeric_column"].min()

df.groupby("category_column").agg(
    {"numeric_column": ["mean", "max", "min"]}
)









