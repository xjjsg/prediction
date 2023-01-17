import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data.csv")
df = df[
    [
        "volume",
        "stochrsi",
        "stochrsi_6",
        "wt2",
        "trix",
        "vr",
        "vr_6",
        "wr",
        "cci",
        "cci_6",
        "atr_5",
        "dma",
        "pdi",
        "mdi",
        "dx",
        "adxr",
        "kdj-k",
        "kdj-j",
        "bias_6",
        "ROC",
        "lower",
        "MACD",
        "MACDhist",
        "return"
    ]
]
df.to_csv("firstData.csv")
sns.set()
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
corr = df.corr()
f, ax = plt.subplots(figsize=(50, 50))
sns.heatmap(corr, annot=False, linewidths=2.5, fmt=".2f", ax=ax)
plt.tight_layout()
plt.savefig("firstHot.png", dpi=1000)
plt.show()
