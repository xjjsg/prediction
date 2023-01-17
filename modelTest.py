import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = keras.models.load_model('my_model')  ##导入模型

data = pd.read_csv('x.csv')  ##导入因子

log = pd.read_csv("log.csv")  ##导入结果

#数据就用standard_finaldata，要选哪些因子自己根据相关系数矩阵调，几个因子间相关性>0.6的保留一个，控制在20个左右
data = np.array(data.iloc[:, 0:18])
log = log.iloc[:, 0:1]
data = data[:, :, np.newaxis]
log_predict = model.predict(data)

plt.plot(log, label="actual value", linewidth=2)
plt.plot(log_predict, label="predictive value", linewidth=1)
plt.ylabel("log-return")
plt.legend(loc=4)
plt.show()