import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, AveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def dec2bin(num):
    l = []
    if num < 0:
        return '-' + dec2bin(abs(num))
    while True:
        num, remainder = divmod(num, 2)
        l.append(str(remainder))
        if num == 0:
            return ''.join(l[::-1])

dl=102869
c1=194
c2=322
c3=660
ds=1992
do=0.026649218750000002
List = [int(dl), int(c1), int(c2), int(c3), int(ds), do]
print(List)
select = str(dec2bin(int(dl)).zfill(17))
dataList = []
for i in range(len(select)):
    if (int(select[i]) == 1):
        dataList.append(i)
df = pd.read_csv('finalData.csv')  ##导入数据
df.set_index('date', inplace=True)
print(dataList)
dataCount = []
for i in dataList:
    dataCount.append(df.iloc[:, i])
x = np.transpose(np.asarray(dataCount))
y = np.array(df.iloc[:, 18:19])
x = x[:, :, np.newaxis]
y = y[:, :, np.newaxis]
#代码分测试集训练集
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    shuffle=False)
# ### 模型的建立
model = Sequential()
#模型第一层，卷积层
model.add(
    Conv1D(int(c1),
           4,
           padding='same',
           activation='tanh',
           input_shape=(x.shape[1], 1)))  #Conv1D后面的数字是可以调的
model.add(AveragePooling1D(2))
model.add(
    Conv1D(int(c2),
           4,
           padding='same',
           activation='tanh',
           input_shape=(x.shape[1], 1)))  #三层卷积，每层conv1D后面的数字不同（一般递增）
model.add(
    Conv1D(int(c3),
           4,
           padding='same',
           activation='tanh',
           input_shape=(x.shape[1], 1)))
model.add(AveragePooling1D(2))
#模型第三层，将2维数据摊平成1维
model.add(Flatten())
#简单全连接网络/防止过拟合
model.add(Dense(int(ds)))  #Dense是可以调的
model.add(Dropout(do))
model.add(Activation('tanh'))
#输出层
model.add(Dense(1))
model.add(Activation('tanh'))
model.compile(loss='mse', optimizer=Adam(lr=3e-4),
              metrics=['accuracy'])  #这个Adam方法和后面的SGD二选一
# model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=20)
# loss = history.history["loss"]
# score = 0
# for i in loss:
#     score += i
# score /= 30
# score += model.evaluate(x_test, y_test, verbose='auto')[0]
# print(1 / score)
y_predict = model.predict(x_test)
n = 0
y_test_data = y_test[:, :, n]  #降为用，要不画不出图
plt.plot(y_test_data[0:100], linewidth=2, label="actual value")
plt.plot(y_predict[0:100], 'r', linewidth=1, label="predictive value")
plt.legend(loc=4)
plt.show()
model.save("my_model")