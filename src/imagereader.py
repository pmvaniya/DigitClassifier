from PIL import Image
import os
import random

random.seed(0)

def getPixelValues(image_path):
    image = Image.open(image_path)

    # Convert image to grayscale
    image = image.convert('L')

    width, height = image.size

    pixel_values = []
    for y in range(height):
        row_values = []
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            row_values.append(pixel_value / 100)
        pixel_values += row_values

    return pixel_values

def getDataSet(folderPath):
    dataset = []

    for directory_name in range(10):
        directory_path = os.path.join(folderPath, str(directory_name))

        for image in os.listdir(directory_path):
            filename = os.path.join(directory_path, image)
            pixelvalues = getPixelValues(filename)
            pixelvalues += [directory_name]
            dataset.append(pixelvalues)

    random.shuffle(dataset)
    
    return dataset[:6000], dataset[6000:7000]

if __name__ == "__main__":

    trainingData, testingData = getDataSet("../data/trainingSet")
    for i in range(len(trainingData[0])):
        if i % 28 == 0:
            print()
        print("%3.2f"%trainingData[0][i], end=" ")
    print()

    for i in range(len(testingData[0])):
        if i % 28 == 0:
            print()
        print("%3.2f"%testingData[0][i], end=" ")
    print()
