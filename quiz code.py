#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load a time series dataset
# For demonstration, let's consider a synthetic time series dataset
# Replace this with your actual dataset loading process
# Example:
# data = pd.read_csv('your_dataset.csv')
# Make sure the dataset has a column for the timestamp and the target variable
# Assuming 'timestamp' is the name of the timestamp column and 'target' is the name of the target variable

# Generate synthetic time series data for demonstration
data = pd.DataFrame({'timestamp': pd.date_range(start='2020-01-01', periods=100),
                     'target': np.sin(np.arange(100) / 10)})

# Step 2: Prepare the data for modeling
# Assuming the target variable is to be forecasted based on historical values
# You may need to preprocess your data, handle missing values, etc.

# Convert timestamp column to datetime type
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Sort data by timestamp
data.sort_values(by='timestamp', inplace=True)

# Step 3: Prepare features and target variable
# Let's use the previous 10 timestamps to predict the next timestamp
lookback = 10

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i:i+lookback]['target'].values)
        y.append(data.iloc[i+lookback]['target'])
    return np.array(X), np.array(y)

X, y = create_sequences(data, lookback)

# Reshape input data to be 3D for LSTM (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Step 5: Evaluate the model's performance
# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)


# In[4]:


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 2: Preprocess the data
# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data into 4D tensor (samples, rows, columns, channels)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Step 3: Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Step 4: Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callback for reducing learning rate on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, callbacks=[reduce_lr])

# Step 5: Evaluate the model's performance
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Step 6: Visualize training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# In[ ]:




