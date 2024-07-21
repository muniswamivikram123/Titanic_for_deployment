from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Load the training data
try:
    train_t = pd.read_csv('train.csv')
    train_t = train_t[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    train_t = train_t.dropna()
except FileNotFoundError:
    raise FileNotFoundError("The train.csv file is missing. Please ensure it is in the correct directory.")

# Preprocess function
def preprocess_input(data):
    pclass = to_categorical(data['Pclass'] - 1, num_classes=3)
    sex = data['Sex'].apply(lambda x: 1 if x == "male" else 0).to_numpy()
    age = (data['Age'].to_numpy() - train_t['Age'].mean()) / train_t['Age'].std()
    sibsp = (data['SibSp'].to_numpy() - train_t['SibSp'].mean()) / train_t['SibSp'].std()
    parch = (data['Parch'].to_numpy() - train_t['Parch'].mean()) / train_t['Parch'].std()
    fare = (data['Fare'].to_numpy() - train_t['Fare'].mean()) / train_t['Fare'].std()
    return np.hstack((pclass, sex.reshape(-1, 1), age.reshape(-1, 1), sibsp.reshape(-1, 1), parch.reshape(-1, 1), fare.reshape(-1, 1)))

# Load the trained model
model = joblib.load('titanic_model.joblib')

@app.route('/')
def index():
    return render_template('index.html', survival_percentage=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': float(request.form['Fare'])
        }
        input_df = pd.DataFrame([data])
        processed_input = preprocess_input(input_df)
        prediction = model.predict(processed_input)
        survival_probability = model.predict_proba(processed_input)[0][1] * 100
        return render_template('index.html', prediction=prediction[0], survival_percentage=survival_probability)
    return render_template('index.html', survival_percentage=None)

if __name__ == '__main__':
    app.run(debug=True)
