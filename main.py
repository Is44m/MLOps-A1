import numpy as np
import pandas as pd

# Load the housing data
data = pd.read_csv('housing.csv')

# Separate features and target variable
X = data.drop(columns='price')
y = data['price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Convert categorical features to numeric using one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Convert to numpy array
X_values = X_encoded.values

# Normalize numerical features manually
def normalize_features(X, numerical_cols):
    X_numeric = X[:, [X_encoded.columns.get_loc(col) for col in numerical_cols]]
    X_mean = X_numeric.mean(axis=0)
    X_std = X_numeric.std(axis=0)
    X_numeric = (X_numeric - X_mean) / X_std
    X[:, [X_encoded.columns.get_loc(col) for col in numerical_cols]] = X_numeric
    return X, X_mean, X_std

X_values, X_mean, X_std = normalize_features(X_values, numerical_cols)

# Normalize target (price)
y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std

# Manually split the data into training and test sets
def train_test_split(X, y, test_size=0.2):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    test_size = int(num_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X_values, y_normalized)

# Add a bias term (intercept) to X
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Initialize model parameters (weights)
theta = np.zeros(X_train.shape[1])

# Define hyperparameters
learning_rate = 0.01  # Lower learning rate
iterations = 4000       # Increase iterations
regularization_param = 0.001  # L2 regularization

# Define the cost function (with regularization)
def compute_cost(X, y, theta, reg_param):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    
    # Add regularization (ignore bias term)
    reg_cost = (reg_param / (2 * m)) * np.sum(theta[1:] ** 2)
    
    return cost + reg_cost

# Define the gradient descent function (with regularization)
def gradient_descent(X, y, theta, learning_rate, iterations, reg_param):
    m = len(y)
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        
        # Regularization term (skip bias term)
        gradient[1:] += (reg_param / m) * theta[1:]
        
        theta -= learning_rate * gradient
        
        if i % 100 == 0:
            cost = compute_cost(X, y, theta, reg_param)
            print(f"Iteration {i}, Loss: {cost}")
    
    return theta

# Train the model
theta = gradient_descent(X_train, y_train, theta, learning_rate, iterations, regularization_param)

# Print the trained weights
print("Trained weights (theta):", theta)

# Predict on test set
def predict(X, theta):
    return X @ theta

y_pred_normalized = predict(X_test, theta)

# Convert predictions and true values back to original scale
def convert_to_original_scale(y_normalized, y_mean, y_std):
    return y_normalized * y_std + y_mean

y_pred = convert_to_original_scale(y_pred_normalized, y_mean, y_std)
y_test_original = convert_to_original_scale(y_test, y_mean, y_std)

# Calculate the accuracy of the model
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

rmse = calculate_rmse(y_test_original, y_pred)
print(f"Root Mean Squared Error (RMSE) on test set: {rmse}")

# Example prediction
predicted_price = predict(X_test[:1], theta)
predicted_price_actual = convert_to_original_scale(predicted_price, y_mean, y_std)

print(f"Predicted price for first test house: {predicted_price_actual[0]}, Actual price: {y_test_original[:1].values[0]}")
