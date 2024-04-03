
import numpy as np

class NN_advanced:
    def __init__(self, hidden_layers:list, input_dim, output_dim):

        self.Ws = []

        for i, hidden_layer in enumerate(hidden_layers):
            if i == 0:
                self.Ws.append(np.random.randn(input_dim + 1, hidden_layers[i]))
            else:
                self.Ws.append(np.random.randn(hidden_layers[i-1] + 1, hidden_layers[i]))
    
        self.Ws.append(np.random.randn(hidden_layers[-1] + 1, output_dim))

        self.N_weights = len(self.Ws)


    @staticmethod
    def add_ones(x):
        [dim, N] = x.shape
        return np.vstack((x, np.ones((1, N))))
    
    @staticmethod
    def accuracy(y, y_pred):
        return np.mean(np.argmax(y, 0) == np.argmax(y_pred, 0))
    
    
    def ReLU(self, x):
        return np.maximum(x, 0)
    
    def softmax(self, x):
        return  np.exp(x)/(np.exp(x).sum(0))

    def forward(self, x):
        hi = x.copy()
        save = []

        for i in range(self.N_weights):
            
            zi = self.Ws[i].T @ self.add_ones(hi)
            
            if i!=self.N_weights-1:
                hi = self.ReLU(zi)
                save.append(hi)
            else:
                output = self.softmax(zi)

        return output, save


    
    def cross_entropy(self, y, y_pred):
        return -np.sum(y * np.log(y_pred))


    def train(self, x, y, n_iter, lr = 0.001, print_every=1):

        for i in range(n_iter):
            y_pred, h = self.forward(x)

            loss = self.cross_entropy(y, y_pred)
            if (i%print_every==0) and (i!=0): print(f"{i}: loss: {loss}")


            delta_l_star = delta_i_plus_one = y_pred - y

            Qs = []

            #print(h.shape, delta_2.shape)

            delta_i =  delta_l_star

            for i in range(1, self.N_weights):
                Qi = self.add_ones(h[-i]) @ delta_i.T

                Qs.append(Qi)

                delta_i = (h[-i] > 0) * (self.Ws[-i][:-1,:] @ delta_i_plus_one)
        
                delta_i_plus_one = delta_i
        
            Qs.append(self.add_ones(x) @ delta_i.T)

            #print([i.shape for i in self.Ws])
            #print([i.shape for i in Qs])



            for j, Q in enumerate(Qs):
                self.Ws[-(j+1)] = self.Ws[-(j+1)] - lr*Qs[j]

        
            # #print(self.W_1.shape, Q_1.shape)
            # self.W_1 = self.W_1 - lr * Q_1
            # self.W_2 = self.W_2 - lr * Q_2

        print(f'final loss: {loss}')
        print(f"final accuracy: {self.accuracy(y, y_pred)}")




if __name__ == '__main__':
    from make_data import make_data
    import matplotlib.pyplot as plt

    X_train, Y, X_test, dim = make_data(example_nr=2)

    X_train = X_train
    X_test = X_test
    Y = Y

    [dim, N] = X_train.shape

    input_dim = dim
    output_dim = len(np.unique(Y))

    print(f"amount of samples: {N}")
    print(f"predicting {np.unique(Y)}")
    print(f"X_train: {[dim, N]}")


    mean = np.mean(X_train, axis=1)
    std = np.std(X_train, axis=1)
    X_train = ((X_train.T - mean)/std).T

    model = NN_advanced(hidden_layers=[100,100], output_dim=output_dim, input_dim=input_dim)

    output, save = model.forward(X_train)

    #print(output, save)

    # plt.plot(X_train[:,0]*std[0] + mean[0], X_train[:,1]*std[1] + mean[1], '.')

    model.train(X_train, Y, 10, lr=0.0001, print_every=1)


    y_pred, _ = model.forward(((X_test.T - mean)/std).T)

    empty = np.zeros((100,100))

    empty[X_test[0], X_test[1]] = np.argmax(y_pred, axis=0)

    plt.imshow(empty)
    plt.plot(X_train[0][Y[0]] * std[0] + mean[0], X_train[1][Y[0]] * std[1] + mean[1], 'r.')
    plt.plot(X_train[0][Y[1]] * std[0] + mean[0], X_train[1][Y[1]] * std[1] + mean[1], 'g.')
    plt.show()
