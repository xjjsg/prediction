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

codes = []
datas=[]
with open("codes.txt") as f:
    for line in f.readlines():
        line = line.strip("\n")
        codes.append(line)
for i in codes:
    df=pd.read_csv('data/'+i+'.csv')
    df.drop('Unnamed: 0', axis=1,inplace=True)
    datas.append(df)


# model.add(TimeDistributed(Conv2D(...))
# model.add(TimeDistributed(MaxPooling2D(...)))
# model.add(TimeDistributed(Flatten()))
# # define LSTM model
# model.add(LSTM(...))
# model.add(Dense(...))