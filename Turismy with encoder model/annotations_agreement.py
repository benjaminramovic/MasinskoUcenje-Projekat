import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


df = pd.read_csv("reviews_sample_full.csv")
pd.set_option("display.max_columns", None)
print(df.info())

attributes = [
    "cleanliness",
    "location",
    "luxury",
    "family_friendly"
]


print("\nFleiss kapa za pojedinacne labele:")

for attr in attributes: 
    table =[]
    for _, row in df.iterrows():

        votes = [
            int(row[attr]),
            int(row[f"{attr}_benjamin"]),
            int(row[f"{attr}_mirnesa"])
        ]
        table.append([
            votes.count(0),
            votes.count(1)
        ])
    kappa = fleiss_kappa(table)
    print(f"--> {attr}: {kappa:.3f}")

total = 0
agree = 0

for attr in attributes:

    for _, row in df.iterrows():

        votes = [
            int(row[attr]),
            int(row[f"{attr}_benjamin"]),
            int(row[f"{attr}_mirnesa"])
        ]


        total += 1
        if votes[0] == votes[1] == votes[2]:
            agree += 1




print("\nUkupno slaganje izmedju anotatora: Benjamin, Mirnesa i ChatGPT iznosi:", round(agree / total, 3))
print("\n")


