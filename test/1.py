import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv("data.csv")
X = data[
    [
        "close",
        "high",
        "low",
        "volume",
        "close_-1_s",
        "rsi",
        "close_2_s",
        "rsi_6",
        "middle",
        "stochrsi",
        "stochrsi_6",
        "wt1",
        "wt2",
        "close_7_smma",
        "trix",
        "middle_10_trix",
        "tema",
        "middle_10_tema",
        "vr",
        "vr_6",
        "wr",
        "wr_6",
        "cci",
        "cci_6",
        "atr",
        "atr_5",
        "supertrend",
        "dma",
        "pdi",
        "mdi",
        "dx",
        "adx",
        "adxr",
        "kdj-k",
        "kdj-d",
        "kdj-j",
        "bias_6",
        "bias_12",
        "bias_24",
        "MA1",
        "MA2",
        "MA3",
        "ROC",
        "upper",
        "BOLL",
        "lower",
        "MACD",
        "MACDsignal",
        "MACDhist",
        "return",
    ]
]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]
print(vif_data)
