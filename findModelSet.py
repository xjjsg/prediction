from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import RouletteWheelSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, AveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

indv_template = BinaryIndividual(ranges=[(500, 1000), (1000, 2000),
                                         (2000, 5000), (5000, 10000),
                                         (5000, 100000), (0.000001, 0.3)],
                                 eps=0.001)
population = Population(indv_template=indv_template, size=100).init()

selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.5, alpha=0.6)

engine = GAEngine(population=population,
                  selection=selection,
                  crossover=crossover,
                  mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])


@engine.fitness_register
def fitness(indv):
    c1, c2, c3, c4, ds, do = indv.solution
    List = [int(c1), int(c2), int(c3), int(c4), int(ds), do]
    print(List)
    df = pd.read_csv('data.csv')  ##导入数据
    df.set_index('date', inplace=True)

    #数据就用standard_finaldata，要选哪些因子自己根据相关系数矩阵调，几个因子间相关性>0.6的保留一个，控制在20个左右

    x = np.array(df.iloc[:, 0:20])
    y = np.array(df.iloc[:, 20:21])
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
    model.add(MaxPooling1D(2))
    model.add(
        Conv1D(int(c2),
               4,
               padding='same',
               activation='tanh',
               input_shape=(x.shape[1], 1)))  #三层卷积，每层conv1D后面的数字不同（一般递增）
    model.add(MaxPooling1D(2))
    model.add(
        Conv1D(int(c3),
               4,
               padding='same',
               activation='tanh',
               input_shape=(x.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(
        Conv1D(int(c4),
               4,
               padding='same',
               activation='tanh',
               input_shape=(x.shape[1], 1)))
    model.add(MaxPooling1D(2))
    #模型第三层，将2维数据摊平成1维
    model.add(Flatten())
    #简单全连接网络/防止过拟合
    model.add(Dense(int(ds)))  #Dense是可以调的
    model.add(Dropout(do))
    model.add(Activation('tanh'))
    #输出层
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer=Adam(lr=6e-11),
                  metrics=['accuracy'])  #这个Adam方法和后面的SGD二选一
    history = model.fit(x_train, y_train, epochs=15, batch_size=20)

    predictive_y_train = model.predict(x_train)
    predictive_y_test = model.predict(x_test)

    n = 0
    y_test_data = y_test[:, :, n]
    y_train_data = y_train[:, :, n]

    plt.plot(predictive_y_train[0:100], label="predictive value")
    plt.plot(y_train_data[0:100], 'r', label="actual value")
    plt.title("train")

    plt.figure()
    plt.plot(predictive_y_test[0:100], label="predictive value")
    plt.plot(y_test_data[0:100], 'r', label="actual value")
    plt.title("test")
    loss = history.history["loss"]
    loss = sum(loss) / len(loss)
    print("loss is :{}".format(loss))
    return float(1 / loss)

engine.run(ng=20)