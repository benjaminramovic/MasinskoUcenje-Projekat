import pandas as pd

# Učitavanje celog skupa
df = pd.read_csv("reviews.csv")

# Nasumičan izbor 50 recenzija
sample = df.sample(n=100, random_state=42)

# Sačuvaj ih u novi fajl
sample.to_csv("reviews_sample.csv", index=False)

attributes = ["cleanliness", "location", "luxury", "family_friendly"]

for attr in attributes:
    sample[f"{attr}_benjamin"] = ""
    sample[f"{attr}_mirnesa"] = ""

sample.to_csv("reviews_sample_full.csv", index=False)