import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
import seaborn as sns



#to check the dimensions of the data
print("Training image :",x_train.shape)

plt.imshow(x_train[0],cmap="gray")
plt.title("Image")
plt.show()
print(x_train[0])

# Flatten the training images to vectors of size 784
# x_train = x_train.reshape(x_train.shape[0], -1) # This line was causing the error
print(x_train[0])
print(y_train[:10])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)), # Recommended way to specify input shape
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history=model.fit(x_train,y_train,epochs=10,batch_size=30,validation_split=0.2)
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"],label="training accuracy")
plt.plot(history.history["val_accuracy"],label="val acc")
plt.legend()
plt.show()

plt.figure(figsize=(15,8))
plt.plot(history.history["loss"],label="training loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.legend()
plt.show()

predictions=model.predict(x_test)
predictions=np.argmax(predictions,axis=1)
print(predictions[0])

y_pred=predictions
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(15,9))
sns.heatmap(cm,cmap="Blues",fmt="d")
plt.show()
print(classification_report(y_test,y_pred))

#For saving and loading the model
model.save("mnist.keras")
print("saved")
load_model=tf.keras.models.load_model("mnist.keras")
print("model loaded")
