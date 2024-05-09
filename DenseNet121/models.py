from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import load_model

# 진행사항 표시
from keras.callbacks import Callback
from tqdm import tqdm
from time import time

total_history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': []
}

class TqdmProgressCallback(Callback):

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.start_time = time()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()
        print('\nEpoch %d/%d' % (epoch + 1, self.epochs))
        self.tqdm_bar = tqdm(total=self.params['steps'], position=0, leave=True)
        self.logs = {}

    def on_batch_end(self, batch, logs={}):
        self.tqdm_bar.update()
        self.logs = logs

    def on_epoch_end(self, epoch, logs={}):
        self.tqdm_bar.close()
        epoch_time = time() - self.epoch_start_time
        self.times.append(epoch_time)
        avg_time_per_epoch = sum(self.times) / len(self.times)
        remaining_time = avg_time_per_epoch * (self.epochs - epoch - 1)
        total_time = time() - self.start_time
        print(' - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f - epoch_time: %.4f s - total_time: %.4f s - remaining_time: %.4f s' %
              (logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc'], epoch_time, total_time, remaining_time))
        total_history['loss'].append(round(logs['loss'], 4))
        total_history['acc'].append(round(logs['acc'], 4))
        total_history['val_loss'].append(round(logs['val_loss'], 4))
        total_history['val_acc'].append(round(logs['val_acc'], 4))
        print(total_history['loss'])
        print(total_history['acc'])
        print(total_history['val_loss'])
        print(total_history['val_acc'])


from keras.applications import DenseNet121
model_base = DenseNet121(weights="imagenet",
                      include_top=False,
                      input_shape=(64,64,3),
                      )

model_base.trainable = False   # Convolution Layer 동결!
model_base.summary()

# 모델 완성하기
model = Sequential()

model.add(model_base)  # 우리 모델의 앞부분에 pretrained network을 삽입!

# classifier를 구현
model.add(Flatten())

# Hidden Layer
model.add(Dense(units=512,
                activation='relu'))
model.add(Dropout(rate=0.5))

# Output Layer
model.add(Dense(units=10,
                activation='softmax'))

# model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

# Callback 설정
from keras.callbacks import ModelCheckpoint, EarlyStopping
cp_cb = ModelCheckpoint(filepath='/content/save/best_model.ckpt',
                        save_weights_only=True,
                        save_best_only=True,
                        monitor='val_acc',
                        verbose=1)

es_cb = EarlyStopping(monitor='loss',
                      patience=3)
