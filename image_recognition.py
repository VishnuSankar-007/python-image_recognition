import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values (scale them to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define class names for CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Step 2: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 output classes for CIFAR-10
])

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Step 4: Train the Model
print("Training the model...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

# Step 6: Make Predictions
predictions = model.predict(x_test)

# Function to visualize predictions
def visualize_prediction(index):
    plt.figure(figsize=(4, 4))
    plt.imshow(x_test[index])
    predicted_class = np.argmax(predictions[index])
    true_class = y_test[index][0]
    plt.title(f"Predicted: {class_names[predicted_class]}\nTrue: {class_names[true_class]}")
    plt.axis('off')
    plt.show()

# Step 7: Visualize a Few Predictions
for i in range(5):  # Display first 5 test images and their predictions
    visualize_prediction(i)
