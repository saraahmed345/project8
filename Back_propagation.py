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
    
    # Use Cross-Entropy Loss
    def cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)

    for iteration in range(num_iterations):
        total_error = 0
        correct_predictions = 0
        
        for x, y in training_data:
            # Forward step
            hidden_layer_input = np.dot(weights_hidden, x) #net
            hidden_layer_output = sigmoid(hidden_layer_input)#ACTIVATION function on hidden layer
            
            output_layer_input = np.dot(weights_output, hidden_layer_output)
            output_layer_output = softmax(output_layer_input)# activation function on output layer
            
            # Backward step
            output_error = y - output_layer_output #the difference between the true output (y) and the predicted output (output_layer_output).
            total_error += cross_entropy_loss(y, output_layer_output)
            
            output_delta = output_error
            hidden_error = np.dot(weights_output.T, output_delta)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
            
            # Update weights
            weights_output += learning_rate * np.outer(output_delta, hidden_layer_output)
            weights_hidden += learning_rate * np.outer(hidden_delta, x)
        
        # Calculate average error for this iteration
        Eav = total_error / len(training_data)
        
        # Calculate training accuracy
        correct_predictions = 0
        total_predictions = len(training_data)
        
        for x, y in training_data:
            hidden_net = np.dot(weights_hidden, x)
            hidden_output = sigmoid(hidden_net)
            
            output_net = np.dot(weights_output, hidden_output)
            output = softmax(output_net)
            
            predicted_class = np.argmax(output)
            
            if predicted_class == np.argmax(y):
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) * 100
       # print(f"Iteration {iteration + 1}/{num_iterations} - Average Error: {Eav:.6f}, Training Accuracy: {accuracy:.2f}%")
        
        # Check if overall error is acceptably low
        if Eav < 0.01:
            break
    
    return weights_hidden, weights_output, accuracy

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Activation function (tanh)
def softmax(x):
    exps = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exps / np.sum(exps)

def custom_mapping(label):
    if label == 'BOMBAY':
        return 0
    elif label == 'CALI':
        return 1
    elif label == 'SIRA':
        return 2

def sig(hl, eta, eboch):
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
    weights_hidden, weights_output, training_accuracy = backpropagation(training_data, eta, eboch)
    correct_predictions = 0
    total_predictions = len(testing_data)

    for inputs, target in testing_data:
        # Forward Step
        hidden_net = np.dot(weights_hidden, inputs)
        hidden_output = sigmoid(hidden_net)

        output_net = np.dot(weights_output, hidden_output)  # Fix the forward step
        output = softmax(output_net)
        predicted_class = np.argmax(output)
        # Check if the prediction is correct
        if predicted_class == target:
            correct_predictions += 1
    testing_accuracy = (correct_predictions / total_predictions) * 100
    print(f"Testing Accuracy of BP with sigmoid: {testing_accuracy:.2f}%")
    print(f"Training Accuracy of BP with sigmoid: {training_accuracy:.2f}%")