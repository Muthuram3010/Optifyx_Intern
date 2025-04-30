from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('regression_model.pkl')

# Define relevant features (only 3 now)
feature_names = ['area', 'bedrooms', 'bathrooms']

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data and handle conversion issues
        area = float(request.form['area'].strip())
        bedrooms = int(request.form['bedrooms'].strip())
        bathrooms = int(request.form['bathrooms'].strip())

        # Create input array with the 3 features
        data = np.array([[area, bedrooms, bathrooms]])

        # Make prediction
        prediction = model.predict(data)[0]
        formatted_prediction = f"{prediction:,.2f} Taka"  # Format with commas

        # Render the result
        return render_template('index.html', prediction=formatted_prediction, feature_names=feature_names)

    except Exception as e:
        # Handle exceptions gracefully and show error message
        return jsonify({"error": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
