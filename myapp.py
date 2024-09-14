from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model parameters and scaling parameters
best_theta = np.load('best_theta.npy')
y_mean, y_std = np.load('y_mean_std.npy')
X_mean, X_std = np.load('X_mean_std.npy')

# Define the columns
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
furnishingstatus_options = ['furnished', 'semi-furnished', 'unfurnished']

X_encoded_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_no', 'mainroad_yes', 'guestroom_no', 'guestroom_yes',
    'basement_no', 'basement_yes', 'hotwaterheating_no', 'hotwaterheating_yes',
    'airconditioning_no', 'airconditioning_yes', 'prefarea_no', 'prefarea_yes',
    'furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

# Serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Normalize numerical features
def normalize_features(X, X_mean, X_std):
    numeric_indices = [i for i, col in enumerate(X_encoded_columns) if col in numerical_cols]
    X_numeric = X[:, numeric_indices]
    X_numeric = (X_numeric - X_mean) / X_std
    X[:, numeric_indices] = X_numeric
    return X

def convert_to_original_scale(y_normalized, y_mean, y_std):
    return y_normalized * y_std + y_mean

def predict(X, theta):
    return X @ theta

def validate_input(data):
    required_fields = numerical_cols + categorical_cols + ['furnishingstatus']
    for field in required_fields:
        if field not in data:
            return False, f'Missing required field: {field}'
    for num_field in numerical_cols:
        try:
            value = int(data[num_field])
            if value < 0:
                return False, f'Value for {num_field} must be non-negative'
        except ValueError:
            return False, f'Value for {num_field} must be a number'
    for cat_field in categorical_cols:
        if data[cat_field] not in ['yes', 'no']:
            return False, f'Value for {cat_field} must be "yes" or "no"'
    if data['furnishingstatus'] not in furnishingstatus_options:
        return False, 'Value for furnishingstatus must be one of "furnished", "semi-furnished", or "unfurnished"'
    return True, ''

def preprocess_input(input_df):
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    furnishingstatus_map = {
        'furnished': ['furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'],
        'semi-furnished': ['furnishingstatus_semi-furnished', 'furnishingstatus_furnished', 'furnishingstatus_unfurnished'],
        'unfurnished': ['furnishingstatus_unfurnished', 'furnishingstatus_furnished', 'furnishingstatus_semi-furnished']
    }
    if 'furnishingstatus' in input_df_encoded.columns:
        furnishingstatus = input_df_encoded['furnishingstatus'].values[0]
        if furnishingstatus in furnishingstatus_map:
            for col in furnishingstatus_map[furnishingstatus]:
                input_df_encoded[col] = 1
            input_df_encoded[['furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']] = input_df_encoded[['furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']].fillna(0)
        input_df_encoded.drop(columns='furnishingstatus', inplace=True)
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded_columns, fill_value=0)
    return input_df_encoded

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Parse input data from the request
        input_data = request.json

        # Validate the input
        is_valid, error_message = validate_input(input_data)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Create a DataFrame from input data
        input_df = pd.DataFrame([input_data])

        # Preprocess the input
        input_df_encoded = preprocess_input(input_df)

        # Ensure the numeric columns are converted to float or int before normalization
        for col in numerical_cols:
            input_df_encoded[col] = pd.to_numeric(input_df_encoded[col], errors='coerce')

        # Convert to numpy array
        X_input = input_df_encoded.values

        # Normalize the input features
        X_input = normalize_features(X_input, X_mean, X_std)

        # Add bias term (for the intercept)
        X_input = np.hstack([np.ones((X_input.shape[0], 1)), X_input])

        # Predict using the model
        y_pred_normalized = predict(X_input, best_theta)

        # Convert prediction back to original scale
        y_pred = convert_to_original_scale(y_pred_normalized, y_mean, y_std)

        # Return the predicted price
        return jsonify({'predicted_price': y_pred[0]})

    except Exception as e:
        # Log the error for debugging
        app.logger.error(f'Exception occurred: {str(e)}')
        return jsonify({'error': 'An error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)
