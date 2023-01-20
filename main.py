    # Creating a Machine Learning Model Predicting KOOSJR Score at 365D
                        # // Allyson Pfeil \\ #

# Import the necessary libraries
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the patient chart data as a pandas dataframe
patient_data = pd.read_csv("C:\dev\opencv.test\TKA data for ML csv.csv")

# Convert all columns of the dataframe to float values and drop empty values
patient_data = patient_data.dropna()
patient_data = patient_data.astype(float)

# Split data into features and target
X = patient_data[[
    'PatientID','Sex','BMI','AGE @ Surgery','Surgeon','KOOS PRE-OP']]
y = patient_data['KOOS POST-OP 365']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert dataframe to numpy array
X_test = X_test.values
y_test = y_test.values

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(X_train)

# Transform the training and test data using the fitted scaler
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to Tensor objects
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

# Initialize the model
# Input layer
input_layer = Input(shape=(6,))

# Hidden layers
hidden_layer_1 = Dense(512, activation='elu')(input_layer)
hidden_layer_2 = Dense(256, activation='elu')(hidden_layer_1)
hidden_layer_3 = Dense(128, activation='elu')(hidden_layer_2)

# Output layer
output_layer = Dense(1)(hidden_layer_3)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='Adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_loss)

# Load the validation data
# validation_data = pd.read_csv("C:\dev\opencv.test\TKA Pred Validation Data.csv")

# Convert all columns of the dataframe to float values
# validation_data = validation_data.astype(float)

# Scale the validation data using the same scaler that was used for the training data
# validation_data = scaler.transform(validation_data)

# Convert the validation data to a Tensor
# validation_data = tf.convert_to_tensor(validation_data)

# Make predictions on the validation data
# predictions = model.predict(validation_data)

# Print the predictions
# print(predictions)
