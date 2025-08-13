
df =df.dropna() # drop rows with missing values
df =df.dropna(axis=1) # drop columns with missing values
df = df.fillna(0) # fill missing values with 0
df = df.fillna(method="ffill") # fill missing values with previous value
df = df.fillna(method="bfill") # fill missing values with next value
df = df.replace("", "") # replace missing values with empty string
df["column_name"]= df["column_name"].interpolate() # interpolate missing values
df = df.rename(columns={"column_name":"new_column_name"}) # rename column
df["column_name"] = df["column_name"].astype("int") # convert column to integer
df = df.drop_duplicates() # drop duplicate rows
df["column_name"] = pd.to_datetime(df["column_name"]) # convert column to datetime



combined = pd.concat([df1, df2], axis=0)
combined = pd.concat([df1, df2], axis=1)

merged = pd.merge(df1, df2, on="common_column")
merged = pd.merge(df1, df2, how="left", on="common_column")
merged = pd.merge(df1, df2, how="inner", on="common_column")


joined = df1.join(df2, how="inner")