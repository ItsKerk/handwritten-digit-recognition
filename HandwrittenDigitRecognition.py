#pip install tensorflow

import tensorflow as tf #machine learning

#MNIST Dataset:Contains 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels.
mnist = tf.keras.datasets.mnist #get_data

#The dataset is split into:
#x_train: The training images (60,000 examples).
#y_train: The training TRUE labels OR TRUE VALUE
#x_test: The testing images (10,000 examples).
#y_test: The testing TRUE labels OR TRUE VALUE
(x_train, y_train),(x_test, y_test) =  mnist.load_data()

#normalize it from 0-255 to 0-1 (a pixel have 255 lightness) helps the model learn more effectively
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#neuronetwork model
model = tf.keras.models.Sequential()

#InputImage: A 28x28 pixel image
#Flatten: Converts the 28x28 image into a 1D vector of 784 values
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) 

#Dense Layer: A fully connected layer with X Neurons, transforming the 784 values into X output values.
#Here we have 3 Dense Layers with diff Neurons
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(32,activation='relu')) 

#Output Layer: The final fully connected layer with 10 neurons
#The Softmax activation converts the 10 values into probabilities
#Representing the likelihood of the input belonging to each of the 10 classes (digits 0 to 9).
model.add(tf.keras.layers.Dense(10,activation='softmax')) 

#Configures the model for training
#Sparse categorical crossentropy calculate loss function for multi-class classification.
#Adam optimizer to adjust weights based on loss 
#Monitor accuracy as a performance metric during training and evaluation
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train the machine learning model
#Epochs = Loops
model.fit(x_train, y_train, epochs=15)

#Save model
model.save('HandwrittenDigitRecognition.keras')

#Load model
model = tf.keras.models.load_model('HandwrittenDigitRecognition.keras')

#Test model
loss , accuracy = model.evaluate(x_test,y_test)
print(loss)
print(accuracy)