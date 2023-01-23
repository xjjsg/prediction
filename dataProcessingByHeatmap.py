import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataProcessing/firstData.csv')
df.drop(columns=['Unnamed: 0', 'date', 'stochrsi_6', 'wr', 'cci', 'cci_6', 'mdi', 'kdj-k','pdi'], inplace=True)

sns.set()
corr = df.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, annot=True, linewidths=2, fmt=".2f", ax=ax, square=True)
plt.tight_layout()
plt.savefig("secondHot.png", dpi=500)
plt.show()
