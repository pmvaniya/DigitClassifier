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
            pixel_value = image.getpixel((x, y)) / 100
            if pixel_value < 0.2:
                row_values.append(0)
            else:
                row_values.append(pixel_value)
        pixel_values += row_values

    return pixel_values

def getDataSet(folderPath):
    dataset = []

    for directory_name in range(10):
        directory_path = os.path.join(folderPath, str(directory_name))

        for image in os.listdir(directory_path):
            filename = os.path.join(directory_path, image)

            # pixelvalues = [directory_name]
            # pixelvalues += getPixelValues(filename)

            pixelvalues = getPixelValues(filename)
            pixelvalues += [directory_name]
            dataset.append(pixelvalues)

    random.shuffle(dataset)
    
    # return dataset
    return dataset[:32000], dataset[32000:]

if __name__ == "__main__":

    dataset = getDataSet("../data")

    # with open("dataset.csv", 'a') as file:
    #     for record in dataset:
    #         file.write(",".join([str(i) for i in record]) + '\n')
    # print("Record appended successfully.")

    trainingData, testingData = getDataSet("../data")
    
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
