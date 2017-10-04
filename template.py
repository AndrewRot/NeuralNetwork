from keras import backend as K
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np


#from internet example
from keras.datasets import mnist
from keras.utils import np_utils

#other math libraries
import math


batch_size = 128

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)



#test some shit
data = np.load('images.npy')
#print(data)

#list of 2d arrays
# 60% Training set
# 15% Validation set
# 25% Test set

print('X_train original shape:', data.shape)

X_train_setasasd = data[0:3900]
print("data shape: ", X_train_setasasd.shape)

# flatten 28*28 images to a 784 vector for each image
num_pixels = 784 #data.shape[0] * data.shape[0]
data = data.reshape(data.shape[0], num_pixels).astype('float32')
data = data / 255

print('Reshaped X_train shape:', data.shape)

#portion off the training data *** this method works. but not an even distrubtion of data passed through the model
#X_train_set = data[0:3900]
#X_valid_set = data[3900:4875]
#X_test_set = data[4876:6499]

#X_train_set = np.array([[]])
#X_valid_set = np.array([[]])
#X_test_set = np.array([[]])

s = (3895,784)
X_train_set = np.zeros(s)
s2 = (971, 784)
X_valid_set = np.zeros(s2)
s3 = (1634, 784)
X_test_set = np.zeros(s3)

#print ("data[0]: ", data[0])
#print ("data[1]: ", data[1])

print (X_train_set.shape)
print (X_valid_set.shape)
print (X_test_set.shape)


#get the labels
labels = np.load('labels.npy')
Y_train = labels
print('Y_train original shape:', Y_train.shape)

#portion off the training data
#Y_train_set = labels[0:0]
#Y_valid_set = labels[0:0]
#Y_test_set = labels[0:0]

#initialize Y values to the same length, these values are precalulcated from the data
s4 = (3895)
Y_train_set = np.zeros(s4)
s5 = (971)
Y_valid_set = np.zeros(s5)
s6 = (1634)
Y_test_set = np.zeros(s6)

print (len(Y_train_set))
print (len(Y_valid_set))
print (len(Y_test_set))


#num_pixels = X_train_set.shape[0] * X_train_set.shape[0]
# flatten 28*28 images to a 784 vector for each image
#X_train_set = X_train_set.reshape(X_train_set.shape[0], num_pixels).astype('float32')
#X_valid_set = X_valid_set.reshape(X_valid_set.shape[0], num_pixels).astype('float32')
#X_test_set = X_test_set.reshape(X_test_set.shape[0], num_pixels).astype('float32')

#X_train_set = X_train_set / 255
#X_valid_set = X_valid_set / 255
#X_test_set = X_test_set / 255

#****************************************************************************************************************
#FIX THE SHIT IN HERE

#more sofisticated data splitting
#use a dictionary (like hashmap) to get a count of each label, split up the data by percent
letterCounts = {}
for i in labels:
	if i in letterCounts.keys(): #if this keyis already contained, increment it's value by one
		letterCounts[i] += 1
	else: #otherwise, insert a new pair and initialize it's value to 1
		letterCounts[i] = 1

#for i in letterCounts:
#    print (i, letterCounts[i])

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
	print("Assure no data loss for ", i, ", count: ", training_set_sizes[i] + validation_set_sizes[i] + test_set_sizes[i])

# see the # of datas for the training set
training_size = 0
for i in training_set_sizes:
	training_size += training_set_sizes[i]
print ("training_set_size: ", training_size)
valid_size = 0
for i in validation_set_sizes:
	valid_size += validation_set_sizes[i]
print ("valid_set_size: ", valid_size)
test_size = 0
for i in test_set_sizes:
	test_size += test_set_sizes[i]
print ("training_set_size: ", test_size)


#now we need to fill our data sets with the right amounts from above
#use these to count the data in each, insert the data into out X_train, valid, test vars
training_set_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
validation_set_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
test_set_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
counter = 0 #use to keep track of the index of the respective element in the data array
#print (data[counter].shape)
for i in labels:
	#print (i, "data[counter].shape ", data[counter].shape, " X_train_set.size: ", X_train_set.shape)

	#i = 0, 1, 2,6, 3 etc
	if (training_set_count[i] < training_set_sizes[i]): #insert in each bin until full
		#X_train_set.append(data[counter])
		#X_train_set = np.append(X_train_set, data[counter],  axis=0)
		X_train_set[training_set_count[i]] = data[counter]
		Y_train_set[training_set_count[i]] = labels[counter]
		#print("X_train_set[", training_set_count[i], "] = ", labels[counter])
		training_set_count[i] += 1
	elif (validation_set_count[i] < validation_set_sizes[i]):
		#X_valid_set.append(data[counter])
		#X_valid_set = np.append(X_valid_set, data[counter],  axis=0)
		X_valid_set[validation_set_count[i]] = data[counter]
		Y_valid_set[validation_set_count[i]] = labels[counter]
		#print("X_valid_set[", validation_set_count[i], "] = ", labels[counter])
		validation_set_count[i] += 1
	else:
		#X_test_set.append(data[counter])
		#X_test_set = np.append(X_test_set, data[counter],  axis=0)
		X_test_set[test_set_count[i]] = data[counter]
		Y_test_set[test_set_count[i]] = labels[counter]
		#print("X_test_set[", test_set_count[i], "] = ", labels[counter])
		test_set_count[i] += 1
	counter += 1

#check the sizes of our new test data arrays
print("Went through ", counter, " labels")
print("X_train_set: ", len(X_train_set), " in shape ", X_train_set.shape, " Y_train_set: ", Y_train_set)
print("X_valid_set: ", len(X_valid_set),  " in shape ", X_valid_set.shape, " Y_valid_set: ", Y_valid_set)
print("X_test_set: ", len(X_test_set), " in shape ", X_test_set.shape, " Y_test_set: ", Y_test_set)

#X_train_set = X_train_set.reshape(training_size, num_pixels).astype('float32')
##X_valid_set = X_valid_set.reshape(valid_size, num_pixels).astype('float32')
#X_test_set = X_train_set.reshape(test_size, num_pixels).astype('float32')

#print("Reshaped again")
#print("X_train_set: ", len(X_train_set), " in shape ", X_train_set.shape)
#print("X_valid_set: ", len(X_valid_set),  " in shape ", X_valid_set.shape)
#print("X_test_set: ", len(X_test_set), " in shape ", X_test_set.shape)


#****************************************************************************************************************




#indices = [(0, 3899), (3900, 4874), (4875, 6500)]
#print ([data[s:e+1] for s,e in indices])
#print('Training dataset: ', X_train_set.shape)


#print(data[10])



#print('X_train original shape:', X_train.shape)
#print('y_train original shape:', y_train.shape)
#print('X_test original shape:', X_test.shape)
#print('y_test original shape:', y_test.shape)

# flatten 28*28 images to a 784 vector for each image
#num_pixels = X_train.shape[1] * X_train.shape[2]



#X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')



# normalize inputs from 0-255 to 0-1

#X_train = X_train / 255
#X_test = X_test / 255


# one hot encode outputs
Y_train_set = np_utils.to_categorical(Y_train_set)
Y_test_set = np_utils.to_categorical(Y_test_set)

Y_valid_set = np_utils.to_categorical(Y_valid_set)
#Y_test_set = np_utils.to_categorical(Y_test_set)

num_classes = Y_test_set.shape[1]#10

#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	#softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. 
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model, adam optimizer as well (different calculous)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# build the model
model = baseline_model()
# Fit the model, use the training set and periodically check against the validation set. Repeat the training on the training set # of epoch times.
# Find out which epoch opitimzed the validation set.
history = model.fit(X_train_set, Y_train_set, validation_data=(X_valid_set, Y_valid_set), epochs=10, batch_size=batch_size, verbose=2)

#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=batch_size, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test_set, Y_test_set, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print (history.history)
#need to predict
#model.predict()



# Model Template

#model = Sequential() # declare model
#model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
#model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#
#model.add(Dense(10, kernel_initializer='he_normal')) # last layer
#model.add(Activation('softmax'))


# Compile Model
#model.compile(optimizer='sgd',
#              loss='categorical_crossentropy', 
#              metrics=['accuracy'])



# Train Model
#history = model.fit(X_train, Y_train, 
#history = model.fit(
#                    validation_data = (x_val, y_val), 
#                    epochs=10, 
#                    batch_size=256)

#score = model.evaluate(x_test, y_test, batch_size = batch_size)



# Report Results

#print(history.history)
#model.predict()


