from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)
CORS(app)
d = pd.read_csv('D:\\summerintern\\api\\.venv\\Crop.csv')
x = d.drop(['crop_num'], axis=1)
y = d['crop_num']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train,y_train)

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_received = request.json
        
        input = np.array(list(data_received.values())).reshape(1, -1)
        print(input)
        p = model.predict(input)[0]
        res = crop_dict[p]
        return jsonify({'message': str(res)})
    except Exception as e:
        return jsonify({'message': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)