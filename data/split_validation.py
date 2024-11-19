import pandas as pd

df = pd.read_csv("./data/train.csv")

sampled_df = df.sample(frac= 0.1, random_state = 42)
sampled_df.to_csv("./data/validation.csv", index=False)

remaining_df = df.drop(sampled_df.index)
remaining_df.to_csv("./data/re_train.csv", index=False)

print(f"10% 샘플링 완료 : {sampled_df.shape[0]} 개")
