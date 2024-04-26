from flask import Flask, render_template, request, jsonify
from src.model import predict
from pickle import load

app = Flask(__name__)

filename = "src/ann_model_128_64_32000_10000.pkl"
with open(filename, 'rb') as file:
    network = load(file)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predictDigit():
    data = request.get_json()
    record = []
    for row in data["data"]:
        for i in row:
            record.append(i)
    prediction = str(predict(network, record))    
    result = {"message": prediction}
    return jsonify(result)

if __name__ == '__main__':
    app.run()
