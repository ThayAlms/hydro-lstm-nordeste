import pandas as pd

url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

df = pd.read_csv(
    url,
    sep=r"\s+",
    engine="python"
)

print(df.head())

df.to_csv(
    "data/oni.csv",
    index=False
)

print("Arquivo salvo em data/noaa/oni.csv")