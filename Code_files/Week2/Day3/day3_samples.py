import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"])
# print(s)

# data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
# df = pd.DataFrame(data)
# print(df)

# Viewing Data
print(df.head()) # print first 5 rows
print(df.tail(3)) # print last 3 rows

print(df.info()) # print info about the dataset
print(df.describe()) # print summary statistics

print(df[["Name", "Age"]]) # print specific columns

print(df[df["Age"] > 25]) # print rows where age is greater than 25

print(df.iloc[0]) # print first row
print(df.iloc[:, 0]) # print first column

print(df.loc[0]) # print first row
print(df.loc[:, "Name"]) # print first column


# df = pd.read_csv("data.csv")
# df.to_csv("data.csv", index=False) # save to csv without index
# df.to_excel("data.xlsx", index=False) # save to excel without index