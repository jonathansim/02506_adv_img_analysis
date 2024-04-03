import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from a_neural_network import NN_advanced

digits = sklearn.datasets.load_digits()
images = digits['images']
targets = digits['target']

X = images.reshape((images.shape[0], -1)).T
h = X.max()/2
X = (X - h)/h
I = np.eye(10, dtype=bool)
T = I[targets].T

def disp(i):
    '''Helping function whens showing images'''
    return (i.reshape(8, 8) + 1) / 2

nr_images = images.shape[0]
nr_show = 8
rg = np.random.default_rng()
random_subset = sorted(rg.choice(range(nr_images), nr_show, replace=False))

fig, ax = plt.subplots(1, nr_show)
for s, a in zip(random_subset, ax):
    a.imshow(disp(X[:, s]))
    a.set_title(f'Im. {s}\nTarget {np.argmax(T[:, s])}')
plt.show()

# Dividing data into training and test set. The same apprach can be used for
# dividing into training and validation.


train_percentage = 0.8
permuted = rg.permutation(range(nr_images))
c = int(nr_images*train_percentage)

train = sorted(permuted[:c])
test = sorted(permuted[c:])

X_train= X[:, train]
T_train= T[:, train]

X_test = X[:, test]
T_test = T[:, test]

model = NN_advanced([50, 50], input_dim=64, output_dim=10)

model.train(X_train, T_train, n_iter=1000, lr=0.00001, print_every=50)