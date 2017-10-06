from keras import backend as K
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np

#from internet example
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model

#other math libraries
import math

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#Variables for the model
batch_size = 512
num_classes = 10

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#test some shit
data = np.load('images.npy')

#list of 2d arrays
# 60% Training set
# 15% Validation set
# 25% Test set

print('X_train original shape:', data.shape)

# flatten 28*28 images to a 784 vector for each image
num_pixels = 784 #data.shape[0] * data.shape[0]
data = data.reshape(data.shape[0], num_pixels).astype('float32')
data = data / 255
print('Reshaped X_train shape:', data.shape)


#portion off the training data *** this method works. make sure that the shape of these arrays match that of data
X_train_set = np.array([]).reshape(0, 784)
X_valid_set = np.array([]).reshape(0, 784)
X_test_set = np.array([]).reshape(0, 784)

print (X_train_set.shape)
print (X_valid_set.shape)
print (X_test_set.shape)


#get the labels
labels = np.load('labels.npy')
Y_train = labels
print('Y_train original shape:', Y_train.shape)

#portion off the training data
#initialize Y values to empty 1d arrays, since they only need to hold the corresponding labels
Y_train_set = np.empty(0)
Y_valid_set = np.empty(0)
Y_test_set = np.empty(0)

print (Y_train_set.shape)
print (Y_valid_set.shape)
print (Y_test_set.shape)


#more sofisticated data splitting
#use a dictionary (like hashmap) to get a count of each label, split up the data by percent
letterCounts = {}
for i in labels:
	if i in letterCounts.keys(): #if this keyis already contained, increment it's value by one
		letterCounts[i] += 1
	else: #otherwise, insert a new pair and initialize it's value to 1
		letterCounts[i] = 1

#for i in letterCounts:
 #   print (i, letterCounts[i])
#Counts for each letter
# 0 651
# 1 728
# 2 636
# 3 669
# 4 654
# 5 568
# 6 664
# 7 686
# 8 600
# 9 644
# 60% Training set
# 15% Validation set
# 25% Test set

#calculate the number of each label to go in each set
training_set_sizes = {}
validation_set_sizes = {}
test_set_sizes = {}
for i in letterCounts:
	training_set_sizes[i] = math.floor(letterCounts[i]*.6)
	validation_set_sizes[i] = math.floor(letterCounts[i]*.15)
	test_set_sizes[i] = math.floor(letterCounts[i]*.25)
	#calculate the new total -> increment out test set amount until the total from each count = the totals from letter counts
	#This prevents losing data by using math.floor
	newTotal = training_set_sizes[i] + validation_set_sizes[i] + test_set_sizes[i]
	#get the difference and add it to our test_set size
	difference = letterCounts[i] - newTotal
	test_set_sizes[i] += difference
	#print("Assure no data loss for ", i, ", count: ", training_set_sizes[i] + validation_set_sizes[i] + test_set_sizes[i])

# see the # of datas for the training set
for i in training_set_sizes:
	#training_size += training_set_sizes[i]
	print (i, "has training: ", training_set_sizes[i], " || validation:", validation_set_sizes[i], " || test:", test_set_sizes[i], " || total: ",  training_set_sizes[i]+validation_set_sizes[i]+test_set_sizes[i])



#now we need to fill our data sets with the right amounts from above
#use these to count the data in each, insert the data into out X_train, valid, test vars
training_set_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
validation_set_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
test_set_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
counter = 0 #use to keep track of the index of the respective element in the data array

print("Partitioning the data.. This could take a minute")
for i in labels:
	#i = 0, 1, 2,6, 3 etc
	if (training_set_count[i] < training_set_sizes[i]): #insert in each bin until full	
		X_train_set = np.append(X_train_set, [data[counter]],  axis=0)
		Y_train_set = np.append(Y_train_set, i)
		training_set_count[i] += 1
	elif (validation_set_count[i] < validation_set_sizes[i]):
		X_valid_set = np.append(X_valid_set, [data[counter]],  axis=0)
		Y_valid_set = np.append(Y_valid_set, i)
		validation_set_count[i] += 1
	else:
		X_test_set = np.append(X_test_set, [data[counter]],  axis=0)
		Y_test_set = np.append(Y_test_set, i)
		test_set_count[i] += 1
	counter += 1

#check the sizes of our new test data arrays
X_train_set = np.array(X_train_set);
X_valid_set = np.array(X_valid_set);
X_test_set = np.array(X_test_set);
print("Went through ", counter, " labels")
print("X_train_set: ", len(X_train_set), " in shape ", X_train_set.shape, " Y_train_set: ", Y_train_set.shape)
print("X_valid_set: ", len(X_valid_set),  " in shape ", X_valid_set.shape, " Y_valid_set: ", Y_valid_set.shape)
print("X_test_set: ", len(X_test_set), " in shape ", X_test_set.shape, " Y_test_set: ", Y_test_set.shape)


#Print out the number of digits that should be placed in each bin for data partitioning
#for i in training_set_count:
#	print("Training set: number,", i, "has ", training_set_count[i], "pieces of data")
#for i in validation_set_count:
#	print("Validation set: number,", i, "has ", validation_set_count[i], "pieces of data")
#for i in test_set_count:
#	print("Test set: number,", i, "has ", test_set_count[i], "pieces of data")


#store an extra copy of the OG labels to check at the end for poor predictions
Y_test_labels = Y_test_set

# one hot encode outputs
Y_train_set = np_utils.to_categorical(Y_train_set)
Y_valid_set = np_utils.to_categorical(Y_valid_set)
Y_test_set = np_utils.to_categorical(Y_test_set)


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	#input layer
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='random_uniform', activation='relu'))

	#additional layers
	model.add(Dense(num_pixels, kernel_initializer='he_normal'))

	#softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. 
	model.add(Dense(num_classes, kernel_initializer='he_normal', activation='softmax'))
	
	# Compile model, adam optimizer as well (different calculous) rmsprop
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
model.save('proj3.h5')

#Check to see if model can be loaded
#del model
#model = load_model('proj3.h5')

# Fit the model, use the training set and periodically check against the validation set. Repeat the training on the training set # of epoch times.
# Find out which epoch opitimzed the validation set.
history = model.fit(X_train_set, Y_train_set, validation_data=(X_valid_set, Y_valid_set), epochs=20, batch_size=batch_size, verbose=2)

#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=batch_size, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test_set, Y_test_set, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print (history.history)



y_pred = model.predict_classes(X_test_set)

p = model.predict_proba(X_test_set)
target_names = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(np.argmax(Y_test_set, axis=1), y_pred, target_names = target_names))
print(confusion_matrix(np.argmax(Y_test_set, axis = 1), y_pred))

print("Find the wrongly predicted labels")
missed_predictions = []
cnt = 0;
for i in y_pred:
	#print("cnt", cnt, "prediction", i, "actual", Y_test_labels[cnt])
	if(i != Y_test_labels[cnt]):
		print("Missed prediction at index", cnt, "of X_test_set. Model prediction", i, ".. actual", Y_test_labels[cnt])
		missed_predictions.append(cnt)
	cnt += 1;
	
#reshape the data to 28 x 28
X_test_set = X_test_set.reshape(X_test_set.shape[0], 28, 28).astype('float32')
for i in range(3):
	plt.subplot(221+i)
	plt.imshow(X_test_set[missed_predictions[i]], cmap=plt.get_cmap('gray'))

plt.show()










