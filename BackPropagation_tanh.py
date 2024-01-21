import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Initialize the weights to small random values
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(0)
    weights_hidden = np.random.randn(hidden_size, input_size)
    weights_output = np.random.randn(output_size, hidden_size)
    return weights_hidden, weights_output

# Step 2: Repeat until overall error Eav becomes acceptably low
def backpropagation(training_data, learning_rate, num_iterations):
    input_size = len(training_data[0][0])
    hidden_size = 100  # Increased number of neurons in the hidden layer
    output_size = len(training_data[0][1])

    weights_hidden, weights_output = initialize_weights(input_size, hidden_size, output_size)

    # Use Mean Squared Error Loss
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    for iteration in range(num_iterations):
        total_error = 0

        for x, y in training_data:
            # Forward step
            hidden_layer_input = np.dot(weights_hidden, x)
            hidden_layer_output = tanh(hidden_layer_input)

            output_layer_input = np.dot(weights_output, hidden_layer_output)
            output_layer_output = output_layer_input

            # Backward step
            output_error = y - output_layer_output
            total_error += mean_squared_error(y, output_layer_output)

            output_delta = output_error
            hidden_error = np.dot(weights_output.T, output_delta)
            hidden_delta = hidden_error * tanh_derivative(hidden_layer_output)

            # Update weights
            weights_output += learning_rate * np.outer(output_delta, hidden_layer_output)
            weights_hidden += learning_rate * np.outer(hidden_delta, x)

        # Calculate average error for this iteration
        Eav = total_error / len(training_data)

        # Check if overall error is acceptably low
        if Eav < 0.01:
            break

    return weights_hidden, weights_output

# Activation function (tanh)
def tanh(x):
    return np.tanh(x)

# Derivative of tanh function
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Softmax function for multiclass classification
def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def custom_mapping(label):
    if label == 'BOMBAY':
        return 0
    elif label == 'CALI':
        return 1
    elif label == 'SIRA':
        return 2

def ta(hl, eboch, eta):
    # Load Dry Beans dataset
    df = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')
    df['Class'] = df['Class'].apply(custom_mapping)

    # Separate features and labels
    X = df[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']].values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=133)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    training_data = list(zip(X_train, np.eye(len(set(y)))[y_train]))
    testing_data = list(zip(X_test, y_test))

    input_size = len(training_data[0][0])
    hidden_size = hl  # Increased number of neurons in the hidden layer
    output_size = len(set(y))

    weights_hidden, weights_output = initialize_weights(input_size, hidden_size, output_size)

    learning_rate = eta  # Experiment with different learning rates
    max_iteration = eboch  # Increase the maximum number of iterations

    weights_hidden, weights_output = backpropagation(training_data, learning_rate, max_iteration)

    # Training Accuracy
    correct_predictions_train = 0
    total_predictions_train = len(training_data)

    for x, target in training_data:
        hidden_net = np.dot(weights_hidden, x)
        hidden_output = tanh(hidden_net)

        output_net = np.dot(weights_output, hidden_output)
        output = softmax(output_net)

        predicted_class = np.argmax(output)

        if predicted_class == np.argmax(target):
            correct_predictions_train += 1

    training_accuracy = (correct_predictions_train / total_predictions_train) * 100

    # Testing Accuracy
    correct_predictions_test = 0
    total_predictions_test = len(testing_data)

    for inputs, target in testing_data:
        hidden_net = np.dot(weights_hidden, inputs)
        hidden_output = tanh(hidden_net)

        output_net = np.dot(weights_output, hidden_output)
        output = softmax(output_net)

        predicted_class = np.argmax(output)

        if predicted_class == target:
            correct_predictions_test += 1

    testing_accuracy = (correct_predictions_test / total_predictions_test) * 100

    print(f"Training Accuracy of BP with Tan_h: {training_accuracy:.2f}%")
    print(f"Testing Accuracy of BP with Tan_h: {testing_accuracy:.2f}%")



