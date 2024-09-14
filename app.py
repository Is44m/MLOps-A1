from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model parameters and scaling parameters
best_theta = np.load('best_theta.npy')
y_mean, y_std = np.load('y_mean_std.npy')
X_mean, X_std = np.load('X_mean_std.npy')

# Define the columns
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Column names used during training
X_encoded_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_no', 'mainroad_yes', 'guestroom_no', 'guestroom_yes',
    'basement_no', 'basement_yes', 'hotwaterheating_no', 'hotwaterheating_yes',
    'airconditioning_no', 'airconditioning_yes', 'prefarea_no', 'prefarea_yes',
    'furnishingstatus_furnished','furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

# Normalize numerical features manually
def normalize_features(X, X_mean, X_std):
    numeric_indices = [i for i, col in enumerate(X_encoded_columns) if col in numerical_cols]
    X_numeric = X[:, numeric_indices]
    X_numeric = (X_numeric - X_mean) / X_std
    X[:, numeric_indices] = X_numeric
    return X

# Convert normalized data to original scale
def convert_to_original_scale(y_normalized, y_mean, y_std):
    return y_normalized * y_std + y_mean

# Predict using the model
def predict(X, theta):
    return X @ theta

# Preprocess input data
def preprocess_input(input_df):
    # One-hot encode the categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Ensure the input data has the same columns as the model expects
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded_columns, fill_value=0)
    
    return input_df_encoded

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Parse JSON input
        input_data = request.json
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        input_df_encoded = preprocess_input(input_df)
        
        # Convert to numpy array and normalize
        X_input = input_df_encoded.values
        X_input = normalize_features(X_input, X_mean, X_std)
        
        # Add bias term
        X_input = np.hstack([np.ones((X_input.shape[0], 1)), X_input])
        
        # Make prediction
        y_pred_normalized = predict(X_input, best_theta)
        y_pred = convert_to_original_scale(y_pred_normalized, y_mean, y_std)
        
        # Return the prediction as a JSON response
        return jsonify({'predicted_price': y_pred[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
