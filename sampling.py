import pandas as pd 
import re

def preProccess(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    return text

df = pd.read_csv("cleaned92k.csv")

print(df.columns)

pos = df[df["label"] == 1]
neg = df[df["label"] == 0]

n = min(len(pos), len(neg))

pos_s = pos.sample(n=n, random_state=42)
neg_s = neg.sample(n=n, random_state=42)

balanced = pd.concat([pos_s, neg_s]).sample(frac=1, random_state=42)

balanced.to_csv("sampled92k.csv", index=False)