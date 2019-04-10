import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
import numpy as np

num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256 # batch size: for SGD, how many data are we used for each train
epochs = 5

# READ IN KAGGLE DATA
with open("data/kaggle_fer2013/fer2013.csv") as file:
     data = file.readlines()

lines = np.array(data)
x_train, y_train, x_test, y_test = [], [], [], []

# helper function for normalization
def normalize(value, min_val, max_val):
	if min_val == max_val:
		normalized = 0
	else:
		normalized = (value - min_val)/(max_val - min_val)
	return normalized
# 1. A) SPLIT DATA INTO TEST AND TRAIN
cnt =0
for i in range(1,lines.size):
    emotion, img, usage = lines[i].split(",")#emotion(0-6),img(pixel data seperate by " ",usage:)
    val = img.split(" ")#pixel as number in a list
    #print(len(val))#result is 2304 = 48*48
    #val is string
    pixels = np.array(val, 'float32')#list becauses a numpy array
    min_num = np.min(pixels)
    max_num = np.max(pixels)
    for i in range(pixels.size):
    	pixels[i] = normalize(pixels[i],min_num,max_num)#normalization process
    #print(pixels.dtype) #the data type is already float32
    #pixels_norm = np.true_divide(pixels,255)
    pixels = pixels.astype('float32')
    reshaped_pixels = pixels.reshape(48,48,1) #reshape into (48,48,1)
    emotion = keras.utils.to_categorical(emotion, num_classes)

    if 'Training' in usage:
        y_train.append(emotion)
        x_train.append(reshaped_pixels)
    elif 'PublicTest' in usage:
        y_test.append(emotion)
        x_test.append(reshaped_pixels)

y_train = np.array(y_train)
x_train = np.array(x_train)
y_test = np.array(y_test)
x_test = np.array(x_test)
#print(x_train.shape)
#print(len(x_train))
# print(x_train[0])
# print(len(y_train))
# 1. B) CAST AND NORMALIZE DATA
# as above
# 1. C) RESHAPE DATA
# as above
# 2. CREATE CNN MODEL
#input = Input(shape=(48, 48, 1, ))
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu',input_shape=(48, 48, 1)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2))) ----usually adding another 2 layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(7, activation = 'softmax'))

# 3. A) DATA BATCH PROCESS
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)#what is the type of data for x_train???
# 3. B) TRAIN AND SAVE MODEL
model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['accuracy'])
# 3. C) TRAIN MODEL
STEP_SIZE_TRAIN = len(x_train)/batch_size
model.fit_generator(generator = train_generator, steps_per_epoch = STEP_SIZE_TRAIN, epochs = 5)
model.save('model_1.h5')
# #evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
# score = model.evaluate(x_test,y_test)
# print("Test_loss: ", score[0])
# print("Test_accuracy: ", score[1])
