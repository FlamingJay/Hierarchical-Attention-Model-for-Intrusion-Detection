import numpy as np
import os
from keras.layers import Dense, Input
from keras.layers import GRU, Bidirectional, TimeDistributed, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import TimeseriesGenerator
from Attention import AttLayer

# 加载数据
from data_processing import load_data
train, train_label, test, test_label, name = load_data()

MAX_SENTS = 11  # 句子数量，即多少个时间步的
WORD_LENTTH = 1
MAX_SENT_LENGTH = 196  # 即多少个特征值

# 利用TimesereisGenerator生成序列数据
time_steps = MAX_SENTS
batch_size = 1024
# 先把训练集划分出一部分作为验证集
train = train[:(172032+time_steps), :]   # 4096 * 42 = 172032
train = train.reshape(-1, MAX_SENT_LENGTH)
train_label = train_label[:(172032+time_steps), ]
test = test[:(81920+time_steps), :]  # 4096 * 20 = 81920
test = test.reshape(-1, MAX_SENT_LENGTH)
test_label = test_label[:(81920+time_steps), ]
# 数据集生成器   需要检查一下是否正确，主要是TimeseriesGenerator的使用情况
train_label_ = np.insert(train_label, 0, 0, axis=0)
test_label_ = np.insert(test_label, 0, 0, axis=0)
train_generator = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH)) # 4个时间步的输入
lstm_1_sent = Bidirectional(GRU(32, return_sequences=True, activation='tanh', recurrent_dropout=0.1))(review_input)  # 对映射后的文档进行操作
lstm_1_sent_drop = Dropout(0.5)(lstm_1_sent)

lstm_2_sent = Bidirectional(GRU(12, return_sequences=True, activation='tanh', recurrent_dropout=0.1))(lstm_1_sent_drop)  # 对映射后的文档进行操作
lstm_2_sent_drop = Dropout(0.5)(lstm_2_sent)

l_att_sent = AttLayer(MAX_SENTS)(lstm_2_sent_drop)  # 文档级别的注意力机制

dense_0 = Dense(6, activation='relu')(l_att_sent)
preds = Dense(1, activation='sigmoid')(dense_0)

model = Model(review_input, preds)

print('Encoder文档summary: ')
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# 进行训练
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
print("model fitting - one layer attention network")
save_dir = os.path.join(os.getcwd(), 'TimeComparison')
filepath="singleFortime{}_.hdf5".format(time_steps)
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=True,
                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduc_lr = ReduceLROnPlateau(monitor='val_acc', patience=10, mode='max', factor=0.2, min_delta=0.0001)
model.fit_generator(train_generator, epochs=1, verbose=2, steps_per_epoch=168,
                                   callbacks=[checkpoint, tbCallBack, reduc_lr],
                                   validation_data=test_generator, shuffle=0, validation_steps=80)
model.load_weights('./TimeComparison/singleFortime{}_.hdf5'.format(time_steps))

# 计算时间
import time
start = time.time()
test_probabilities = model.predict_generator(test_generator, verbose=1)
end = time.time()
print("Step:{}, Time:{:.4f}".format(time_steps, (end-start)))