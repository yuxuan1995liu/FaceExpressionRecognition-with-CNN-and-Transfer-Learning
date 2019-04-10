import keras
import random
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential

import numpy as np

from data.iPhoneX.faces import face_samples

num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 12
epochs = 6
emotion_tag = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad': 4, 'surprise':5, 'neutral':6}
test_train_ratio = 0.3

#emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#print(len(face_samples['A 18:48:11']['angry']['y']))

#face_samples is a dictionary, with 50 sample_ids, emotion_tags 7 , face_samples[sample_ids][emotion_tags][x,y,z], 3 layers of keys
#x, y, z with 1220 points
# 9. READ IN iPHONE X DATA AND SHAPE
#voxel data
voxel = [[[[0 for z in range(24)] for y in range(24)] for x in range(24)] for i in range(350)]
emotion_cnt = 0
for sample_id in face_samples.keys():
	for emotion in face_samples[sample_id].keys():
		range_x = max(face_samples[sample_id][emotion]['x'])-min(face_samples[sample_id][emotion]['x'])
		range_y = max(face_samples[sample_id][emotion]['y'])-min(face_samples[sample_id][emotion]['y'])
		range_z = max(face_samples[sample_id][emotion]['z'])-min(face_samples[sample_id][emotion]['z'])
		step_x = range_x/24
		step_y = range_y/24
		step_z = range_z/24
		for i in range(len(face_samples[sample_id][emotion]['x'])):
			x_dim=face_samples[sample_id][emotion]['x'][i]
			y_dim=face_samples[sample_id][emotion]['y'][i]
			z_dim=face_samples[sample_id][emotion]['z'][i]
			x_raise = int((x_dim-min(face_samples[sample_id][emotion]['x']))/step_x)
			y_raise = int((y_dim-min(face_samples[sample_id][emotion]['y']))/step_y)
			z_raise = int((z_dim-min(face_samples[sample_id][emotion]['z']))/step_z)
			if x_raise ==24:
				x_raise = 23
			if y_raise ==24:
				y_raise = 23
			if z_raise == 24:
				z_raise =23
			# dim_range_x_low = min(face_samples[sample_id][emotion]['x'])+step_x*x
			# dim_range_x_high = min(face_samples[sample_id][emotion]['x'])+step_x*(x+1)
			# dim_range_y_low = min(face_samples[sample_id][emotion]['y'])+step_y*y
			# dim_range_y_high = min(face_samples[sample_id][emotion]['y'])+step_y*(y+1)
			# dim_range_z_low = min(face_samples[sample_id][emotion]['z'])+step_z*z
			# dim_range_z_high = min(face_samples[sample_id][emotion]['z'])+step_z*(z+1)
						# if x_dim>=dim_range_x_low and x_dim<=dim_range_x_high:
						# 	if y_dim>=dim_range_y_low and y_dim<=dim_range_y_high:
						# 		if z_dim>=dim_range_z_low and z_dim<=dim_range_z_high:
			voxel[emotion_cnt][x_raise][y_raise][z_raise] +=1
		emotion_cnt+=1
#y_test emotion data
emotion_train = []
for sample_id in face_samples.keys():
	for emotion in face_samples[sample_id].keys():
		emotion = emotion_tag[emotion]
		emotion_array = keras.utils.to_categorical(emotion, num_classes)
		emotion_train.append(emotion_array)
#random sample
index = range(350)
index_test = random.sample(index,int((test_train_ratio*350)/(test_train_ratio+1)))
index_train=[]
for i in range(350):
	if i not in index_test:
		index_train.append(i)
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(len(voxel)):
	if i in index_train:
		x_train.append(voxel[i])
		y_train.append(emotion_train[i])
	else:
		x_test.append(voxel[i])
		y_test.append(emotion_train[i])
num_test = int((test_train_ratio*350)/(test_train_ratio+1))
num_train = 350 - num_test
x_train = np.array(x_train).reshape(num_train,24,24,24,1)
y_train = np.array(y_train)
x_test = np.array(x_test).reshape(num_test,24,24,24,1)
y_test = np.array(y_test)

# 10. CREATE MODEL OF CHOICE

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(24,24,24,1)))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
model.add(Dropout(0.1))

model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(24,24,24,1)))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
# 11. TRAIN AND TEST MODEL
model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, batch_size = batch_size, epochs = epochs)
model.save('model_4.h5')

#evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
# score = model.evaluate(x_test,y_test)
plot_model(model, to_file = 'model_4.png', show_shapes = True, show_layer_names = True)
