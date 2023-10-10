import numpy as np
class logistic_regression:
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate


    def __hypothesis(self, X):
        return 1/(1+np.exp(-np.matmul(X, self.__theta)))	

    def __gradient(self,X, y):
        h = self.__hypothesis(X)
        grad = np.dot(X.transpose(), (h - y))
        return grad

    def __cost(self,X, y):
        h = self.__hypothesis(X)
        J = np.dot((h - y).transpose(), (h - y))
        J /= 2
        return J[0]

    def __create_mini_batches(self,X, y):
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // self.batch_size
        i = 0

        for i in range(n_minibatches + 1):
            mini_batch = data[i * self.batch_size:(i + 1)*self.batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % self.batch_size != 0:
            mini_batch = data[i * self.batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    # function to perform mini-batch gradient descent


    def train(self,X, y,epochs = 10 ,batch_size = 32):
        self.batch_size = batch_size
        self.__theta = np.zeros((X.shape[1], 1))
        self.__error_list = []
        for itr in range(epochs):
            mini_batches = self.__create_mini_batches(X, y)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                self.__theta = self.__theta - self.learning_rate * self.__gradient(X_mini, y_mini)
                self.__error_list.append(self.__cost(X_mini, y_mini))

    def predict(self,X,threshold = 0.5):
        y_prediction = np.array(self.__hypothesis(X))
#         y_prediction = y_prediction.reshape(y_prediction.shape[0])
# #         print(y_prediction)
        y_pred = []
        for s in y_prediction:
            if s >= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
    
    def get_theta(self):
        return self.__theta

    # def get_error(self):
    #     return self.__error_list