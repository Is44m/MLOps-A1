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
# Normalize numerical features manually
def normalize_features(X, numerical_cols):
    # Get indices of the numerical columns
    numeric_indices = [X_encoded.columns.get_loc(col) for col in numerical_cols]
    
    # Extract numerical data
    X_numeric = X[:, numeric_indices]
    
    # Ensure X_numeric is a NumPy array (which it should be in this case)
    X_numeric = np.array(X_numeric, dtype=float)
    
    # Calculate mean and standard deviation
    X_mean = X_numeric.mean(axis=0)
    X_std = X_numeric.std(axis=0)
    
    # Normalize the numerical columns
    X_numeric = (X_numeric - X_mean) / X_std
    
    # Assign the normalized values back to the original array
    X[:, numeric_indices] = X_numeric
    
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
# Define the gradient descent function (with regularization)
def gradient_descent(X, y, theta, learning_rate, iterations, reg_param):
    # Ensure that all inputs are of float64 type to avoid casting issues
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    theta = np.array(theta, dtype=np.float64)
    
    m = len(y)
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        
        # Regularization term (skip bias term)
        gradient[1:] += (reg_param / m) * theta[1:]
        
        # Update the weights
        theta -= learning_rate * gradient
        
        if i % 100 == 0:
            cost = compute_cost(X, y, theta, reg_param)
            print(f"Iteration {i}, Loss: {cost}")
    
    return theta

# Convert predictions and true values back to original scale
def convert_to_original_scale(y_normalized, y_mean, y_std):
    return y_normalized * y_std + y_mean

# Calculate the accuracy of the model
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# Grid search hyperparameter optimization
# Hyperparameter grid search
def grid_search_hyperparameters(X_train, X_test, y_train, y_test, learning_rates, iterations_list, regularizations):
    best_rmse = float('inf')
    best_hyperparams = None
    best_theta = None
    
    for lr in learning_rates:
        for iters in iterations_list:
            for reg in regularizations:
                print(f"Testing with learning rate: {lr}, iterations: {iters}, regularization: {reg}")
                
                # Initialize model parameters (weights)
                theta = np.zeros(X_train.shape[1])
                
                # Train the model with the current hyperparameters
                theta = gradient_descent(X_train, y_train, theta, lr, iters, reg)
                
                # Make predictions on the test set
                y_pred_normalized = predict(X_test, theta)
                
                # Convert predictions and true values back to original scale
                y_pred = convert_to_original_scale(y_pred_normalized, y_mean, y_std)
                y_test_original = convert_to_original_scale(y_test, y_mean, y_std)
                
                # Calculate RMSE
                rmse = calculate_rmse(y_test_original, y_pred)
                print(f"RMSE: {rmse}")
                
                # Track the best hyperparameters
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_hyperparams = (lr, iters, reg)
                    best_theta = theta
    
    return best_rmse, best_hyperparams, best_theta



# Predict on test set
def predict(X, theta):
    return X @ theta

# Define potential values for hyperparameters
learning_rates = [0.001, 0.01, 0.1]
iterations_list = [2000, 4000, 6000]
reg_params = [0.0001, 0.001, 0.01]


# Perform grid search
best_rmse, best_hyperparams, best_theta = grid_search_hyperparameters(
    X_train, X_test, y_train, y_test, learning_rates, iterations_list, reg_params
)


# Print the best results
print(f"Best RMSE: {best_rmse}")
print(f"Best hyperparameters: {best_hyperparams}")

# Predict on test set using best theta
y_pred_normalized = predict(X_test, best_theta)

# Convert predictions and true values back to original scale
def convert_to_original_scale(y_normalized, y_mean, y_std):
    return y_normalized * y_std + y_mean

y_pred = convert_to_original_scale(y_pred_normalized, y_mean, y_std)
y_test_original = convert_to_original_scale(y_test, y_mean, y_std)

# Calculate RMSE on test set
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

rmse = calculate_rmse(y_test_original, y_pred)
print(f"Final Root Mean Squared Error (RMSE) on test set: {rmse}")

# Example prediction
def predict(X, theta):
    return X @ theta

predicted_price = predict(X_test[:1], best_theta)
predicted_price_actual = convert_to_original_scale(predicted_price, y_mean, y_std)

print(f"Predicted price for first test house: {predicted_price_actual[0]}, Actual price: {y_test_original[:1].values[0]}")
