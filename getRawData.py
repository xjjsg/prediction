import time
import akshare as ak
import math
import pandas as pd
import numpy as np
import stockstats
from sklearn.preprocessing import StandardScaler
import talib
import matplotlib.pyplot as plt
import seaborn as sns


def getReturnRate(close, close1):
    if close == 0:
        return 0.0
    else:
        return math.log(close1 / close)


def getStockData(code):
    dailyFrame = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date="20200101",
        end_date="20221231",
        adjust="qfq",
    )
    returnRate = []
    for i in range(dailyFrame.shape[0] - 1):
        returnRate.append(getReturnRate(dailyFrame.loc[i][2], dailyFrame.loc[i + 1][2]))
    dailyFrame.drop([len(dailyFrame) - 1], inplace=True)
    dailyFrame.loc[:, "return"] = returnRate
    return dailyFrame


def dataProcessing(midData):
    midData.rename(
        columns={
            "日期": "date",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        },
        inplace=True,
    )
    midData.set_index("date", inplace=True)

    midData.drop(["开盘", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"], axis=1, inplace=True)

    stockStat = stockstats.StockDataFrame.retype(midData)

    midData["close_-1_s"] = stockStat[["close_-1_s"]]
    midData["rsi"] = stockStat[["rsi"]]
    midData["close_2_s"] = stockStat[["close_2_s"]]
    midData["rsi_6"] = stockStat[["rsi_6"]]
    midData["middle"] = stockStat[["middle"]]
    midData["stochrsi"] = stockStat[["stochrsi"]]
    midData["stochrsi_6"] = stockStat[["stochrsi_6"]]
    midData["wt1"] = stockStat[["wt1"]]
    midData["wt2"] = stockStat[["wt2"]]
    midData["close_7_smma"] = stockStat[["close_7_smma"]]
    midData["trix"] = stockStat[["trix"]]
    midData["middle_10_trix"] = stockStat[["middle_10_trix"]]
    midData["tema"] = stockStat[["tema"]]
    midData["middle_10_tema"] = stockStat[["middle_10_tema"]]
    midData["vr"] = stockStat[["vr"]]
    midData["vr_6"] = stockStat[["vr_6"]]
    midData["wr"] = stockStat[["wr"]]
    midData["wr_6"] = stockStat[["wr_6"]]
    midData["cci"] = stockStat[["cci"]]
    midData["cci_6"] = stockStat[["cci_6"]]
    midData["atr"] = stockStat[["atr"]]
    midData["atr_5"] = stockStat[["atr_5"]]
    midData["supertrend"] = stockStat[["supertrend"]]
    midData["dma"] = stockStat[["dma"]]
    midData["pdi"] = stockStat[["pdi"]]
    midData["mdi"] = stockStat[["mdi"]]
    midData["dx"] = stockStat[["dx"]]
    midData["adx"] = stockStat[["adx"]]
    midData["adxr"] = stockStat[["adxr"]]
    midData["kdj-k"], midData["kdj-d"] = talib.STOCH(
        midData.high, midData.low, midData.close
    )
    midData["kdj-j"] = midData["kdj-k"] * 3 - midData["kdj-d"] * 2
    midData["bias_6"] = (
        (midData["close"] - midData["close"].rolling(6, min_periods=1).mean())
        / midData["close"].rolling(6, min_periods=1).mean()
        * 100
    )
    midData["bias_12"] = (
        (midData["close"] - midData["close"].rolling(12, min_periods=1).mean())
        / midData["close"].rolling(12, min_periods=1).mean()
        * 100
    )
    midData["bias_24"] = (
        (midData["close"] - midData["close"].rolling(24, min_periods=1).mean())
        / midData["close"].rolling(24, min_periods=1).mean()
        * 100
    )
    midData["bias_6"] = round(midData["bias_6"], 2)
    midData["bias_12"] = round(midData["bias_12"], 2)
    midData["bias_24"] = round(midData["bias_24"], 2)
    midData["MA1"] = talib.MA(np.array(midData.close), timeperiod=5)
    midData["MA2"] = talib.MA(np.array(midData.close), timeperiod=10)
    midData["MA3"] = talib.MA(np.array(midData.close), timeperiod=20)
    midData["ROC"] = talib.ROC(midData["close"], timeperiod=10)
    midData["upper"], midData["BOLL"], midData["lower"] = talib.BBANDS(
        midData.close,
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0,
    )
    midData["MACD"], midData["MACDsignal"], midData["MACDhist"] = talib.MACD(
        np.array(midData.close), fastperiod=6, slowperiod=12, signalperiod=9
    )
    # 缺失值处理
    midData = midData.dropna(axis=0, how="any")
    # 去无穷值
    df_inf = np.isinf(midData)
    midData[df_inf] = 0
    # 数据标准化
    finalData = midData.iloc[:, 0 : midData.shape[1]]
    scaler = StandardScaler()
    scaler.fit(finalData)
    finalData = scaler.transform(finalData)
    finalData = pd.DataFrame(
        finalData, columns=list(midData.iloc[:, 0 : midData.shape[1]].columns)
    )
    finalData.drop(["return"], axis=1, inplace=True)
    finalData.insert(finalData.shape[1], "return", midData["return"].values)

    for i in midData.columns:
        midData.drop([i], axis=1, inplace=True)
    for i in finalData.columns:
        midData.insert(midData.shape[1], i, finalData[i].values)
    finalData = midData
    finalData.sort_index(inplace=True)
    return finalData


def getFirstData(df):
    fd = df[
        [
            "date",
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
            "return",
        ]
    ]
    return fd


frameList = []
i = 0
codes = []
with open("codes.txt") as f:
    for line in f.readlines():
        line = line.strip("\n")
        codes.append(line)

for code in codes:
    i = i + 1
    if i == 200:
        i = 0
        time.sleep(30)
    frameList.append(getStockData(code))
midData = pd.concat(frameList)
finalData = dataProcessing(midData)
testData = dataProcessing(getStockData("000685"))
finalData.to_csv("data.csv")

sns.set()
corr = finalData.corr()
f, ax = plt.subplots(figsize=(50, 50))
sns.heatmap(corr, annot=False, linewidths=2.5, fmt=".2f", ax=ax, square=True)
plt.tight_layout()
plt.savefig("hot.png", dpi=500)
plt.show()

finalData=pd.read_csv("data.csv")
testData = getFirstData(testData)
testData.to_csv("test.csv")
finalData=pd.read_csv('data.csv')
finalData = getFirstData(finalData)

finalData.to_csv("firstData.csv")
sns.set()
corr = finalData.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, annot=False, linewidths=2, fmt=".2f", ax=ax, square=True)
plt.tight_layout()
plt.savefig("firstHot.png", dpi=500)
plt.show()
