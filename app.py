from flask import Flask, render_template, request, jsonify
from src.model import predict
from pickle import load
from src.utils import condense

app = Flask(__name__)

filename = "models/ann_model_cleaned.pkl"
with open(filename, 'rb') as file:
    network = load(file)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predictDigit():
    canvas = request.get_json()["data"]
    for i in range(len(canvas)):
        canvas[i] /= 100
    canvas = condense(canvas, 28)
    prediction = str(predict(network, canvas))
    show(canvas)
    print(prediction, "\n")
    result = {"message": prediction}
    return jsonify(result)

def show(data):
    data = ['⬤' if item != 0 else ' ' for item in data]
    data = [['│'] + data[i:i+28] + ['│'] for i in range(0, len(data), 28)]

    print("\n")
    print("┌" + ("─" * 57) + "┐")
    for row in data:
        print(" ".join(row))
    print("└" + ("─" * 57) + "┘")

if __name__ == '__main__':
    app.run()
