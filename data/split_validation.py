import pandas as pd

df = pd.read_csv("./v0.1/v0.1.6.csv")

sampled_df = df.sample(frac= 0.1, random_state = 42)
sampled_df.to_csv("./val_v0.1.6.csv", index=False)

remaining_df = df.drop(sampled_df.index)
remaining_df.to_csv("./train_v0.1.6.csv", index=False)

print(f"10% 샘플링 완료 : {sampled_df.shape[0]} 개")
