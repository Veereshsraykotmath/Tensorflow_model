import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix,classification_report
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()



#to check the dimensions of the data
print("Training image :",x_train.shape)

plt.imshow(x_train[0],cmap="gray")
plt.title("Image")
plt.show()
print(x_train[0]) 

# Flatten the training images to vectors of size 784
x_train = x_train.reshape(x_train.shape[0], -1)
print(x_train[0])
print(y_train[:10])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])  
