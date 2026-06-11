import pandas as pd 

data = {
    'name':["Benjamin","Seid"],
    'age':[24,7]
}

df = pd.DataFrame(data)

df2 = pd.read_csv('reviews.csv')

print(df2)

# print(df2)
# print(pd.options.display.max_rows)

# print(df2.head())
# print(df2.tail())

# print(df2.info())


## Brisemo redove sa praznim komentarom
# df2.dropna(inplace=True)

# print(df2.info())

# # Ciscenje
# new_df = df2[["comments"]]
# print(new_df) 

# reduced = new_df.sample(n=9000, random_state=45)
# print(reduced)