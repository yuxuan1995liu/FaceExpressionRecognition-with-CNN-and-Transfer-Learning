from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import keras
from visualization_gui import plot_emotion_prediction


num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 7
epochs = 3
emotion_tag = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad': 4, 'surprise':5, 'neutral':6}
#load train data
list_dir = [i for i in os.listdir("data/my_data/train") if i!='.DS_Store']
x_train = []
y_train = []
for dirs in list_dir:
	#print(dirs)
	img = image.load_img("data/my_data/train/" + dirs, grayscale=True, target_size = (48,48))
	x = image.img_to_array(img)
	#x = np.expand_dims(x,axis=0)
	x/=255
	emotion = emotion_tag[dirs[0:-4]]
	emotion = keras.utils.to_categorical(emotion, num_classes)
	x_train.append(x)
	y_train.append(emotion)
y_train = np.array(y_train)
x_train = np.array(x_train)

#load test data
list_dir_test = [i for i in os.listdir("data/my_data/test") if i!='.DS_Store']
x_test = []
y_test = []
emotion_label = []
for dirs in list_dir:
	#print(dirs)
	img = image.load_img("data/my_data/test/" + dirs, grayscale=True, target_size = (48,48))
	x = image.img_to_array(img)
	#x = np.expand_dims(x,axis=0)
	x/=255
	emotion_label.append(dirs[0:-4])
	emotion_test = emotion_tag[dirs[0:-4]]
	emotion_test = keras.utils.to_categorical(emotion_test, num_classes)
	x_test.append(x)
	y_test.append(emotion)
y_test = np.array(y_test)
x_test = np.array(x_test)

#train model on train data

model = load_model('model_2_dropout.h5')
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(x_train,y_train,batch_size=128)
model.fit_generator(generator = train_generator, epochs = 5)
#model.fit(x_train,y_train, epochs = 5, batch_size=128)
model.save('model_3.h5')
# test on test data
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

# plot result
#model = load_model('model_3.h5')
for i in range(x_test.size):
	img_data=np.expand_dims(x_test[i], axis = 0)
	print(emotion_label[i])
	custom = model.predict(img_data)
	plot_emotion_prediction(custom[0])
