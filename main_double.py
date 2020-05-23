import numpy as np
import pandas as pd
from collections import defaultdict
import re

import sys
import os


from keras.layers import Dense, Input
from keras.layers import GRU, Bidirectional, TimeDistributed, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import TimeseriesGenerator

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from Attention import AttLayer
# 加载数据
from data_processing import load_data
train, train_label, test, test_label, name = load_data()

MAX_SENTS = 8  # 句子数量，即多少个时间步的
WORD_LENTTH = 1
MAX_SENT_LENGTH = 196  # 即多少个特征值

# 利用TimesereisGenerator生成序列数据
time_steps = MAX_SENTS
batch_size = 1024
# 先把训练集划分出一部分作为验证集
train = train[:(172032+time_steps), :]   # 4096 * 42 = 172032
train = train.reshape(-1, WORD_LENTTH, MAX_SENT_LENGTH)
train_label = train_label[:(172032+time_steps), ]
test = test[:(81920+time_steps), :]  # 4096 * 20 = 81920
test = test.reshape(-1, WORD_LENTTH, MAX_SENT_LENGTH)
test_label = test_label[:(81920+time_steps), ]
# 数据集生成器   需要检查一下是否正确，主要是TimeseriesGenerator的使用情况
train_label_ = np.insert(train_label, 0, 0, axis=0)
test_label_ = np.insert(test_label, 0, 0, axis=0)
train_generator = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)


sentence_input = Input(shape=(1, MAX_SENT_LENGTH))
l_lstm_0 = Bidirectional(GRU(128, return_sequences=True))(sentence_input)
l_lstm_0_drop = Dropout(0.5)(l_lstm_0)

l_lstm = Bidirectional(GRU(32, return_sequences=True))(l_lstm_0_drop)

l_att = AttLayer(32)(l_lstm)  # 100是attention_dim
sentEncoder = Model(sentence_input, l_att)
print('Encoder句子summary: ')
sentEncoder.summary()

review_input = Input(shape=(MAX_SENTS, 1, MAX_SENT_LENGTH))
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(32, return_sequences=True))(review_encoder)
l_lstm_sent_drop = Dropout(0.5)(l_lstm_sent)

l_lstm_sent_2 = Bidirectional(GRU(16, return_sequences=True))(l_lstm_sent_drop)
l_att_sent = AttLayer(16)(l_lstm_sent)

dense_0_batch = BatchNormalization()(l_att_sent)
dense_0 = Dense(16, activation='relu')(l_att_sent)

preds_batch = BatchNormalization()(dense_0)
preds = Dense(1, activation='sigmoid')(dense_0)
model = Model(review_input, preds)
print('Encoder文档summary: ')
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# 进行训练
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
print("model fitting - Hierachical attention network")
save_dir = os.path.join(os.getcwd(), 'hierarchical attention')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath="best_model_1.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=True,
                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduc_lr = ReduceLROnPlateau(monitor='val_acc', patience=20, mode='max', factor=0.2, min_delta=0.0001)
model.fit_generator(train_generator, epochs=200, verbose=2, steps_per_epoch=168,
                                   callbacks=[checkpoint, tbCallBack, reduc_lr],
                                   validation_data=test_generator, shuffle=0, validation_steps=80)
model.load_weights('./models/best_model.hdf5')
train_probabilities = model.predict_generator(train_generator, verbose=1)

train_pred = train_probabilities > 0.5
train_label_original = train_label_[(time_steps-1):-2, ]

test_probabilities = model.predict_generator(test_generator, verbose=1)
test_pred = test_probabilities > 0.5
test_label_original = test_label_[(time_steps-1):-2, ]

from sklearn.metrics import confusion_matrix, classification_report

print('Train_set Confusion Matrix')
print(confusion_matrix(train_label_original, train_pred))

print('Test_set Confusion Matrix')
print(confusion_matrix(test_label_original, test_pred))

print('classification report')
print(classification_report(test_label_original, test_pred))