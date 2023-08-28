
# @title Import dependancies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# @title Create dataset with 100 datas and 2 variable
x,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

# displays x and y dimensions
print("dimension of x:", x.shape)
print("dimension of y:", y.shape)

plt.scatter(x[:,0], x[:,1], c=y, cmap="summer")

# @title createt initialization function

def initialization(X):
  """
  ceci nous donnera un vecteur W (2,1 ) car l'idée c'est d'avoir un vecteur
  w qui contient autant de parametre qu'il y'a de variable
  """
  W = np.random.randn(X.shape[1], 1)

  """ pour le parametre b(biais) nous lui passons un nombre réel
  car la fonction d'initialisation est z = w1x1 + w2x2 + b
  """
  b = np.random.randn(1)

  return (W, b)

# test
W, b = initialization(x)
print(W.shape)
print(b.shape)

# @title implement our model function
def model(X, W, b):
  """
    the first things we are doing is build Z function (Z= XW + b)
    then we are compute activation function A = 1 / 1 + e(-Z)
  """
  Z = X.dot(W) + b
  A = 1 / (1 + np.exp(-Z))

  return A

A = model(x, W, b)
A.shape

# @title implement the Log Loss function(fonction coût)

def log_loss(A, y):
  """
    in theorie L = -1/m sum(log(ai)*yi + (1-yi)*log(1-ai))
    m = number of data in our dataset then m = len(y)
  """
  epsilon = 1e-15
  return  1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y)*np.log(1 - A +epsilon))

  # this function return a real number which measure of error our model
# test
log_loss(A, y)

# @title Create Gradient function

def gradients(A, x, y):
  """
    we have two gradients the jacobien that we note dW and db( derivative of
    log_loss function with respect to b)
    dW = 1/m * trans(X).(A-Y)
    db = 1 /m * sum(A-Y)
  """
  dW = 1 / len(y) * np.dot(x.T, A-y)
  db = 1 / len(y) * np.sum(A - y)
  return (dW, db)
# test
dw,db= gradients(A, x, y)
print(dw.shape)
db

# @title build the update function

"""
this function take as input the gradients, W, b and learning rate
"""
def update(dW, db, W, b, learning_rate):
  # nous allons implementer l'agorithme de la descencte de gradient
  """
  wi = wi - a(dl/dwi)  a= learning rate and (dl/dwi) = dW
  bi = bi - a(dl/dbi)  (dl/dbi) = db
  """
  W = W - learning_rate * dW
  b = b - learning_rate * db

  return (W, b)

# test
W, b = update(dw, db, W, b, 2)
W

# @title create a prediction function

def predict(X, W, b):
  # computer the output of the model (activation)
  A = model(X, W, b)
  # print(A)
  return A >= 0.5

from sklearn.metrics import accuracy_score

# @title build our Artificial neural

"""
  cette fonction va prendre en entré nos données x et y , un pas d'apprentissage
  pour notre fonction de mise a jour et nombre d'iteration pour notre algo
  d'apprentissage
"""
def artificial_neuron(X, y, learning_rate=0.5, n_iter=100):
  # initialization of parameter w and b
  W,b = initialization(X)

  Loss = []
  # create learning loop
  for i in range(n_iter):
    # launch result of our model
    A = model(X, W, b)

    # capture error of our model
    Loss.append(log_loss(A, y))

    #create dW, db gradient
    dW,db = gradients(A, X, y)

    # update W and b parameters
    W, b = update(dW,db, W, b, learning_rate)

  # compute the preddiction of all data x in dataset
  # en d'autre terme on calcul ce que la machine predit pour ces san valeurs
  y_pred = predict(X,W, b)

  # display the performance of our model in computing the accuracy metrics
  print(accuracy_score(y, y_pred))

  plt.plot(Loss)
  plt.show()

  # then return W and b parameter which model learned
  return (W, b)

W, b = artificial_neuron(x, y)

# @title Predict the class of the new_data not in our dataset
new_data = np.array([2, 1])

#  dessinons la frontiere de décision
"""
on sait A = 50% signifie qu'il existe un couple (x1, x2) pour lesquels z = 0
z(x1,x2) = 0
w1x1 + w2x2 + b = 0
pour construire cette droite on a:
X1  | X2
-2|
 08 |
pour trouver les valeurs de x2 on se sert de l'équation précedente
x2 = (-w1x1 - b) / w2
"""
x0 = np.linspace(-2, 4, 100)

x1 = (-W[0] * x0 - b) / W[1]

plt.scatter(x[:,0], x[:,1], c=y, cmap="summer")
plt.scatter(new_data[0], new_data[1], c="r")
plt.plot(x0, x1, c="orange", lw=3)
plt.show()

predict(new_data, W, b)
"""
on constate que grace a cette frontière de décision
la machine a predit que une perfomance du modèle sur les 100 données
ce qui veut dire que si la performance est 92% alors 8 données ne s'aurait
être du bon côté de la frontière de décision
"""

