from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle  # For loading your pre-trained model
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and encoders
model = pickle.load(open('model.pkl', 'rb'))  # Replace with the correct path to your model file
le_gender = pickle.load(open('gender_encoder.pkl', 'rb'))  # Replace with the correct path to your encoder
le_treatment = pickle.load(open('treatment_encoder.pkl', 'rb'))  # Replace with the correct path to your encoder
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))  # Replace with the correct path to your preprocessor

# Load complaints data
df = pd.read_csv('afzalpur.csv')  # Replace with the correct path to your dataset
complaints_list = df['complaint_cleaned'].tolist()

# Route for serving the homepage
@app.route('/')
def home():
    return render_template('index.html')

# API to fetch complaints dynamically based on primary complaint
@app.route('/api/filter_complaints', methods=['POST'])
def filter_complaints():
    try:
        # Get the primary complaint from the user
        primary_complaint = request.json.get('primary_complaint', '').lower()

        # Filter relevant complaints
        filtered_complaints = [complaint for complaint in complaints_list if primary_complaint in complaint.lower()]

        # Return the filtered complaints
        return jsonify({'filtered_complaints': filtered_complaints})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API to predict treatment based on user input
@app.route('/api/predict', methods=['POST'])
def predict_treatment():
    try:
        # Get input from the request
        data = request.json
        age = int(data['age'])
        gender = data['gender']
        complaint = data['complaint']
        duration = int(data['duration'])

        # Validate gender input
        if gender not in ['Male', 'Female']:
            return jsonify({'error': "Invalid gender. Use 'Male' or 'Female'."}), 400

        # Transform inputs
        gender_encoded = le_gender.transform([gender])[0]
        new_sample = pd.DataFrame({
            'age_cleaned': [age],
            'gender_encoded': [gender_encoded],
            'complaint_cleaned': [complaint],
            'duration_cleaned': [duration]
        })

        # Preprocess and predict
        user_input_transformed = preprocessor.transform(new_sample)
        predicted_treatment_encoded = model.predict(user_input_transformed)
        predicted_treatment = le_treatment.inverse_transform(predicted_treatment_encoded.astype(int))[0]

        # Return prediction
        return jsonify({'treatment': predicted_treatment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
#if __name__ == '__main__':
#    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable
    app.run(host='0.0.0.0', port=port)
