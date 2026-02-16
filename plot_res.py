
import pandas as pd

import matplotlib.pyplot as plt

f_path = '/work/Sultan/runs/train_nopickups21_1_26/results_1.csv'
df = pd.read_csv(f_path, index_col=None)
col_names = list(df.columns)[5:11]
col_len = len(col_names)
row = 2
col = int(col_len / row)
fig, ax = plt.subplots(row,col, figsize=(10, 5)) 
for name, ax in zip(col_names, ax.flatten()):
    ax.set_title(name)
    ax.plot(df[name])
plt.tight_layout()
plt.show()

