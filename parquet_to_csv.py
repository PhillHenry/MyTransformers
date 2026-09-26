import sys
import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_parquet(input_file)
df = df.drop("symbol", axis=1)
df = df.rename(columns={"ts": "timestamp"})
df.to_csv(output_file, index=False)