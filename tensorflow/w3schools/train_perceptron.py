import random
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self, no, learning_rate = 0.00001):
        self.no = no + 1
        self.learnc = learning_rate
        self.bias = 1
        self.weights = []
        for i in range(self.no):
            self.weights.append(random.random()*2 - 1)

    def activate(self, inputs):
        sum = 0.0
        i = 0
        while i < len(inputs):
            sum += inputs[i] * self.weights[i]
            i += 1
        if sum > 0:
            return 1
        else:
            return 0

    def train(self, realinputs, desired):
        inputs = realinputs.copy()
        inputs.append(self.bias)
        guess = self.activate(inputs)
        error = desired - guess
        if error != 0:
            i = 0
            while i < len(inputs):
                self.weights[i] += self.learnc * error * inputs[i]
                i += 1


if __name__ == "__main__":
    numPoints = 500
    xMin = 0
    yMin = 0
    xMax = 800
    yMax = 600
    xPoints = []
    yPoints = []
    for i in range(numPoints):
        xPoints.append(random.random() * xMax)
        yPoints.append(random.random() * yMax)
    fig, ax = plt.subplots()
    plt.scatter(xPoints, yPoints)
    ax.axline((0,50),slope=1.2)
    plt.show()
    desired = []
    for i in range(numPoints):
        if yPoints[i] > (xPoints[i] * 1.2 + 50):
            desired.append(1)
        else:
            desired.append(0)

    ptron = Perceptron(2)
    for j in range(10000):
        for i in range(numPoints):
            ptron.train([xPoints[i], yPoints[i]], desired[i])

    fig, ax = plt.subplots()
    ax.axline((0, 50), slope=1.2)
    xPoints1 = []
    yPoints1 = []
    for i in range(numPoints):
        xPoints1.append(random.random() * xMax)
        yPoints1.append(random.random() * yMax)
    for i in range(numPoints):
        x = xPoints1[i]
        y = yPoints1[i]
        guess = ptron.activate([x, y, ptron.bias])
        if guess == 0:
            plt.scatter(x, y, color='r')
        else:
            plt.scatter(x, y, color='g')
    plt.show()
