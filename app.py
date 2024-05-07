from flask import Flask, render_template, request, jsonify
from src.model import predict
from pickle import load
from src.utils import condense
from PIL import Image

app = Flask(__name__)

filename = "models/ann_model_cleaned.pkl"
with open(filename, 'rb') as file:
    network = load(file)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/identifyCanvas', methods=['POST'])
def identifyCanvas():
    canvas = request.get_json()["data"]
    canvas = condense(canvas, 28)
    
    prediction = str(predict(network, canvas))
    show(canvas)
    print(prediction, "\n")
    
    result = {"message": prediction}
    return jsonify(result)

@app.route('/identifyImage', methods=['POST'])
def identifyImage():
    image = request.files['image']
    image_arr = convertImage(image, 0.2)
    
    prediction = str(predict(network, image_arr))
    show(image_arr)
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

def convertImage(image, limit):
    img = Image.open(image)
    img_gray = img.convert('L')
    pixel_values = list(img_gray.getdata())
    
    pixel_values = [i/255 for i in pixel_values]
    for i in range(len(pixel_values)):
        if pixel_values[i] <= limit:
            pixel_values[i] = 0

    return pixel_values

if __name__ == '__main__':
    app.run()
