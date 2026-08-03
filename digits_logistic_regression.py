import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn import linear_model 
from sklearn import datasets
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score
from scipy.stats import mode


digits = datasets.load_digits()

print('=' * 50)
print("digits.data.shape= ", digits.data.shape)
print()
print("digits.images.shape= ", digits.images.shape)
print('=' * 50)
# print(digits.data)
# print(digits.images)
# print(digits.target)
# print()

# plt.imshow(digits.images[100])
# plt.show()

X = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

reg_logestic = linear_model.LogisticRegression()
reg_logestic.fit(x_train, y_train)

out_prod = reg_logestic.predict(x_test)

# print(out_prod)
# print(y_test)

cor = 0
for i in range(y_test.size):
    if out_prod[i] == y_test[i]:
        cor += 1

# print (cor, incor)
# print(y_test.size)

correct_percentage = (cor * 100) / y_test.size
print("correct prediction= ", correct_percentage, "%")

msr = sm.mean_squared_error(y_test, out_prod)
print('mean squared error= ', msr)


image = []
for i in range (10):
    image.append(cv2.imread(f'test dataset\{i}.jpg'))
    image[i] = cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB)  # turn BGR to RGB
    image[i] = image[i][:, :, 0]
    image[i] = image[i].flatten()

# plt.imshow(image[1].reshape(8, 8))
# plt.show()
out = reg_logestic.predict(image)
print(out)

real_out = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(accuracy_score(real_out, out))
