
class MLTrainer:
    def __init__(self, xArray, yArray, learnc=0.00001):
        self.xArray = xArray.copy()
        self.yArray = yArray.copy()
        self.points = len(xArray)
        self.weight = 0
        self.bias = 1
        self.cost = 0
        self.learnc = learnc

    def cost_error(self):
        total = 0
        for i in range(self.points):
            total +=(self.yArray[i] - (self.weight * self.xArray[i] + self.bias)) **2
        return total/self.points

    def train(self, iter):
        for i in range(iter):
            self.update_weights()
        self.cost = self.cost_error()

    def update_weights(self):
        w_deriv = 0
        b_deriv = 0
        for i in range(self.points):
            wx = self.yArray[i] - (self.weight*self.xArray[i] + self.bias)
            w_deriv += -2 * wx * self.xArray[i]
            b_deriv += -2 * wx
        self.weight -= (w_deriv/self.points) * self.learnc
        self.bias -= (b_deriv/self.points) * self.learnc