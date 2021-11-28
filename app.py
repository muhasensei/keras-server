from flask import Flask, jsonify, request, abort, make_response
from keras.models import load_model
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('keras.h5')

@app.route('/', methods=['GET'])
def get_tasks():
    return 'works'

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    data = request.json
    sample = {
        "salary(thousand kzt)": data['salary'],
        "n_of_individual_projects": data['projects'],
        "motivation_r": data['motivation'] / 100,
        "relationship_with_others_r": data['relationship'] / 100,
        "communication_skills_r": data['communication'] / 100,
        "task_management_r": data['task_management'] / 100,
        "total_rating": data['total_rating'] / 100,
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict)
    return {
        'prediction': int(predictions[0][0] * 100),
        'formData': data,
    }

if __name__ == '__main__':
    app.run(debug=True)
