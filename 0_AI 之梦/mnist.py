from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")
import scipy.io as scio
data = scio.loadmat('mnist-original.mat')
x, y = data['data'], data['label']
x = x.T
y = y.reshape(-1)

def mnist_plot(digitnumber):
    digitshape = x[y==digitnumber].shape[0]
    some_digit = x[y==digitnumber][np.random.randint(0,digitshape-1,1)[0]]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    return some_digit_image

def plot_digits(images_per_row=10, **options):
    instances = np.r_[x[:12000:600], x[13000:30600:600], x[30600:60000:590]]
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")



def mnist_preidct(digitarray):
    clf = load('sgd_clf.joblib')
    return clf.predict(digitarray.reshape(1,784))


