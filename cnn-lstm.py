from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Lambda
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Bidirectional, TimeDistributed, Reshape
from keras.layers import Conv2D, MaxPool2D, TimeDistributed, LSTM, GlobalAveragePooling2D, Convolution2D, MaxPooling2D, BatchNormalization, GRU
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from tensorflow.keras import regularizers


# input with shape of height=64 and width=64
inputs = Input(shape=(64,64,1))
 
# block1
conv_1 = Conv2D(32, (3,3), activation = 'relu', padding='same')(inputs)
batch_norm_1 = BatchNormalization()(conv_1)
# block 2
conv_2 = Conv2D(32, (3,3), activation = 'relu', padding='same')(batch_norm_1)
pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)
batch_norm_2 = BatchNormalization()(pool_2)
# block 3
conv_3 = Conv2D(64, (3,3), activation = 'relu', padding='same')(batch_norm_2)
batch_norm_3 = BatchNormalization()(conv_3)
# block 4
conv_4 = Conv2D(64, (3,3), activation = 'relu', padding='same')(batch_norm_3)
pool_4 = MaxPool2D(pool_size=(2, 2))(conv_4)
batch_norm_4 = BatchNormalization()(pool_4)
# block 5
conv_5 = Conv2D(128, (3,3), activation = 'relu', padding='same')(batch_norm_4)
batch_norm_5 = BatchNormalization()(conv_5)
# block 6
conv_6 = Conv2D(128, (3,3), activation = 'relu', padding='same')(batch_norm_5)
pool_6 = MaxPool2D(pool_size=(1, 2))(conv_6)
batch_norm_6 = BatchNormalization()(pool_6)
# block 7
conv_7 = Conv2D(256, (3,3), activation = 'relu', padding='same')(batch_norm_6)
batch_norm_7 = BatchNormalization()(conv_7)
# block 8
conv_8 = Conv2D(256, (3,3), activation = 'relu', padding='same')(batch_norm_7)
pool_8 = MaxPool2D(pool_size=(1, 2))(conv_8)
batch_norm_8 = BatchNormalization()(pool_8)
# block 9
conv_9 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_8)
pool_9 = MaxPool2D(pool_size=(1, 2))(conv_9)
batch_norm_9 = BatchNormalization()(pool_9)
# block 10
conv_10 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_9)
pool_10 = MaxPool2D(pool_size=(1, 2))(conv_10)
batch_norm_10 = BatchNormalization()(pool_10)
 
reshaped = Reshape((1,16,512))(batch_norm_10)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(reshaped)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)

# kernel_regularizer=regularizers.l2(0.01)
outputs = Dense(10, activation = 'softmax' )(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
act_model.summary()