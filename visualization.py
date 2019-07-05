# 实现的是双层注意力机制，没有用到batchnormalization

import numpy as np
import pandas as pd
from collections import defaultdict
import re

import sys
import os


from keras.layers import Dense, Input, multiply
from keras.layers import GRU, Bidirectional, TimeDistributed, Dropout, BatchNormalization, LSTM
from keras.models import Model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import Adam

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from Attention import AttLayer
# 加载数据
from data_processing import load_data
train, train_label, test, test_label, name = load_data()

MAX_SENTS = 10  # 句子数量，即多少个时间步的
WORD_LENTTH = 1
MAX_SENT_LENGTH = 196  # 即多少个特征值

# 利用TimesereisGenerator生成序列数据
time_steps = MAX_SENTS
batch_size = 1024
# 先把训练集划分出一部分作为验证集
train = train[:(172032+time_steps), :]   # 4096 * 42 = 172032
train = train.reshape(-1,  MAX_SENT_LENGTH)
train_label = train_label[:(172032+time_steps), ]
test = test[:(81920+time_steps), :]  # 4096 * 20 = 81920
test = test.reshape(-1, MAX_SENT_LENGTH)
test_label = test_label[:(81920+time_steps), ]
# 数据集生成器   需要检查一下是否正确，主要是TimeseriesGenerator的使用情况
train_label_ = np.insert(train_label, 0, 0, axis=0)
test_label_ = np.insert(test_label, 0, 0, axis=0)
train_generator = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)


sentence_input = Input(shape=(MAX_SENT_LENGTH, ))
attention_probs = Dense(MAX_SENT_LENGTH, activation='softmax', name='attention_vec')(sentence_input)
attention_mul = multiply([sentence_input, attention_probs])
# ATTENTION PART FINISHES HERE

sentEncoder = Model(sentence_input, attention_mul)
print('Encoder句子summary: ')
sentEncoder.summary()


review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH))  # 文档级别输入
review_encoder = TimeDistributed(sentEncoder)(review_input)  # 对每一个文档中的每一个句子进行句子级别的特征表示操作

l_lstm_sent_0 = Bidirectional(GRU(32, return_sequences=True, activation='tanh', recurrent_dropout=0.1))(review_encoder)  # 对映射后的文档进行操作
lstm_drop_0 = Dropout(0.5)(l_lstm_sent_0)

l_lstm_sent_1 = Bidirectional(GRU(12, return_sequences=True, activation='tanh', recurrent_dropout=0.1, name='gru_2'))(lstm_drop_0)  # 对映射后的文档进行操作
lstm_drop_1 = Dropout(0.5)(l_lstm_sent_1)
# l_lstm_sent_2 = Bidirectional(GRU(16, return_sequences=True, activation='tanh'))(l_lstm_sent_1)  # 对映射后的文档进行操作

l_att_sent = AttLayer(MAX_SENTS)(lstm_drop_1)  # 文档级别的注意力机制  64, 16, 8, 1 能到98.5%，没有dropout


dense_2 = Dense(6, activation='relu')(l_att_sent)

preds = Dense(1, activation='sigmoid')(dense_2)

model = Model(review_input, preds)
print('Encoder文档summary: ')
model.summary()
optimize = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8)

model.compile(loss='binary_crossentropy',
              optimizer=optimize,
              metrics=['acc'])

model.load_weights('./hierarchical_attention/hierarchical_test_10_for_figure.hdf5')
# model.load_weights('./hierarchical_attention/hierarchical_10.hdf5')





test_probabilities = model.predict_generator(test_generator, verbose=1)
np.save('test_predicted.npy', test_probabilities)


test_pred = test_probabilities > 0.5
test_label_original = test_label_[(time_steps-1):-2, ]
np.save('test_original.npy', test_label_original)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(test_label_original, test_pred))




# normal 26, 888, anomaly 15, 1
index = 45
x, y = test_generator[index]
number = 686
sen = x[number, :, :]
lal = y[number]
input_data = sen.reshape(1, MAX_SENTS, -1)
get_x = Model(input=model.input, output=model.layers[5].output)
inter_output = get_x.predict(input_data)  # gru output

predicted = model.predict(input_data)
print('y = ', lal)
print('predicted = ', predicted)

w = model.layers[6].get_weights()[0]
b = model.layers[6].get_weights()[1]
u = model.layers[6].get_weights()[2]

uit = np.exp(np.dot(inter_output.reshape(MAX_SENTS, 24), w) + b)
ait = np.dot(uit, u)
ait = np.squeeze(ait, -1)
ait = np.exp(ait)
ait /= np.sum(ait, axis=0, keepdims=True)
ait = np.expand_dims(ait, 0)


get_y = Model(input=sentEncoder.input, output=sentEncoder.layers[1].output)
wit = []
word = input_data.reshape(-1, 196)
for index in range(MAX_SENTS):
    word_input = word[index, :]
    result = get_y.predict(word_input.reshape(1, 196))
    wit.append(result)


import matplotlib.pyplot as plt
from matplotlib import cm

def draw_heatmap(word_att, sentence_att, xlabels, ylabels):
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    # cmap=cm.get_cmap('bone_r', 10)
    cmap = cm.get_cmap('CMRmap_r', 10)
    plt.figure(figsize=(6, 18))
    grid = plt.GridSpec(8, 8, wspace=0.5, hspace=0.5)

    ax = plt.subplot(grid[0:7, 0:8])
    # plt.plot(x,y,'ok',markersize=3,alpha=0.2)

    # ax.set_yticks(range(len(ylabels)))
    # ax.set_yticklabels(ylabels, font)
    ax.set_ylabel('Feature Number', font)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, font)
    plt.title('Slice-based Attention Map and Feature-based Attention Map.  (Predicted: Attack.  Actual: Attack.)', font)
    vmax=word_att[0][0]
    vmin=word_att[0][0]
    for i in word_att:
            for j in i:
                if j>vmax:
                    vmax=j
                if j<vmin:
                    vmin=j
    map=ax.imshow(word_att.T, interpolation='nearest', cmap=cmap, aspect='auto', vmin=-0.02, vmax=vmax)  # 这样子会让最小值作为白色，最大值作为黑色。

    plt.grid()
    # cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)

    low_ax = plt.subplot(grid[7, 0:8])
    low_ax.set_xticks(range(len(xlabels)))
    low_ax.set_xticklabels(xlabels, font)
    low_ax.set_ylabel('Probability', font)
    # low_ax.set_ylim(0.098, 0.101)
    plt.bar(range(MAX_SENTS), sentence_att)
    plt.show()


word_att = np.array(wit).reshape(MAX_SENTS, 196)
sentence_att = np.array(ait).reshape(MAX_SENTS, )

xlables = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6', 'step_7', 'step_8', 'step_9', 'step_10']
ylabels = name
draw_heatmap(word_att, sentence_att, xlables, ylabels)
print('done')