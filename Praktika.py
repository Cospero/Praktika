import numpy as np
import pandas as pd
import seaborn as sns
import os

from tensorflow import keras

from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator, load_img
from keras_preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D
import matplotlib.pyplot as plt
import time

import platform
import psutil


import warnings
warnings.filterwarnings('ignore')

osInf = platform.uname()
print(f"Тип системы:", osInf.system)
print(f"Версия системы:", osInf.version)
print(f"Архитектура системы:", osInf.machine)
print(f"Процессор:", osInf.processor)

svmem = psutil.virtual_memory()
convertMd=1024**2
print(f"Всего памяти:", svmem.total//convertMd, "MB")
print(f"Доступной памяти:", svmem.available//convertMd, "MB")

cpu = psutil.cpu_freq()
print(f"Текущая частота работы процессора:", cpu.current, "Mhz")
print("Количество логических ядер процессора:", os.cpu_count())


images_path = 'img_align_celeba/img_align_celeba'
attributes_path = 'list_attr_celeba.csv'

attributes = pd.read_csv(attributes_path)
print("base", attributes)

attributes = attributes.set_index('image_id')
print("Измененный индекс", attributes)

print("Список атрибутов", attributes.columns.tolist())



attributes = attributes.astype(int)


attributes = (attributes + 1) // 2
print("Нормализованные атрибуты", attributes)







def load_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0
    return image


X = []
y = []

for img_name in attributes.index[:10000]:
    img_path = os.path.join(images_path, img_name)
    X.append(load_image(img_path))
    y.append(attributes.loc[img_name].values)

X = np.array(X)
y = np.array(y)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train)//32)
print(len(X_val) // 32)


model = keras.Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation="relu", input_shape=(128,128,3)))
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(40, activation='relu'))

print("Архитектура модели")
model.summary()

print("model compiling in process")
model.compile(optimizer="rmsprop", loss='mean_squared_error',metrics=["mae"])
print("model compiling finished")




StartTime= time.time()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=8, shuffle=True)
endTime = time.time() - StartTime

fig,axes = plt.subplots(1,2,figsize=(15,7))
acc = history.history['mae']
val_acc=history.history['val_mae']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs = range(len(acc))
sns.lineplot(x=epochs,y=acc,linestyle='--',ax=axes[0])
sns.lineplot(x=epochs,y=val_acc,linestyle='-.',ax=axes[0])
sns.lineplot(x=epochs,y=loss,linestyle='--',ax=axes[1])
sns.lineplot(x=epochs,y=val_loss,linestyle='-.',ax=axes[1])
axes[0].legend(['Training mae', 'Validation mae'])
axes[1].legend(['Training loss', 'Validation loss'])
fig.suptitle('Training metrics')

plt.show()


print("Время обучения модели", endTime)


test_loss, test_mae = model.evaluate(X_val, y_val)
print("Среднеквадратичная ошибка:", test_mae)
print("Потери: ", test_loss)


sample_image_path = 'TestPhoto.jpg'
sample_image = load_image(sample_image_path)
sample_image = np.expand_dims(sample_image, axis=0)

predicted_attributes = model.predict(sample_image)[0]
predicted_attributes = (predicted_attributes > 0.5).astype(int)

atrArr=[]
print("Предсказанные атрибуты:")
for attr, attValue in zip(attributes.columns, predicted_attributes):
    print(attr," ", attValue)
    strAt=attr+" "+ str(attValue)
    atrArr.append(strAt)
print(atrArr)