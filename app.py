from flask import Flask, jsonify, request, abort, make_response
from keras.models import load_model
import tensorflow as tf
from flask_cors import CORS
import pickle
import pandas as pd
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def get_tasks():
    return 'works'



@app.route('/predict/forest', methods=['POST'])
def predict_forest():
    if not request.json:
        abort(400)
    data = request.json
    sample = {
        "n_of_individual_projects": data['projects'],
        "motivation_r": data['motivation'] / 100,
        "relationship_with_others_r": data['relationship'] / 100,
        "communication_skills_r": data['communication'] / 100,
        "task_management_r": data['task_management'] / 100,
        "salary(thousand kzt)": 400.0,
        "total_rating": data['total_rating'] / 100,
    }
    df = pd.DataFrame(data=sample, index=[0])
    with open('models/model_rf', 'rb') as f:
        loaded_model = pickle.load(f)
    prediction = loaded_model.predict(df)
    return {
        'prediction': pd.Series(prediction).to_json(orient='values'),
        'dataframe': df.to_json(orient='values'),
    }



@app.route('/predict/gradient', methods=['POST'])
def predict_gradient():
    if not request.json:
        abort(400)
    data = request.json
    sample = {
        "n_of_individual_projects": data['projects'],
        "motivation_r": data['motivation'] / 100,
        "relationship_with_others_r": data['relationship'] / 100,
        "communication_skills_r": data['communication'] / 100,
        "task_management_r": data['task_management'] / 100,
        "salary(thousand kzt)": 400.0,
        "total_rating": data['total_rating'] / 100,
    }
    df = pd.DataFrame(data = sample, index=[0])
    with open('models/model_gb', 'rb') as f:
        loaded_model = pickle.load(f)
    prediction = loaded_model.predict(df)
    return {
        'prediction': pd.Series(prediction).to_json(orient='values'),
    }




@app.route('/predict/keras', methods=['POST'])
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
    model = load_model('models/keras.h5')
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict)
    return {
        'prediction': int(predictions[0][0] * 100),
        'formData': data,
    }

if __name__ == '__main__':
    app.run(debug=True)