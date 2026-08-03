from sklearn import linear_model 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


digits = datasets.load_digits()

# print(digits.data)
# print(digits.images)
# print(digits.data.shape)
# print(digits.images.shape)
# print(digits.target)
# print()

# plt.imshow(digits.images[100])
# plt.show()


new_data = []
new_target = []
for i in range (digits.data.shape[0]):
    if digits.target[i] == 0 or digits.target[i] == 1:
        new_data.append(digits.data[i])
        new_target.append(digits.target[i])


x_train, x_test, y_train, y_test = train_test_split(new_data, new_target, test_size=0.2)

reg_logestic = linear_model.LogisticRegression()
reg_logestic.fit(x_train, y_train)

out_prod = reg_logestic.predict(x_test)

print('Out put prediction is= ')
print(out_prod)
print('Real out put is=')
print(y_test)

err = y_test - out_prod
cor = 0
for i in err:
    if err[i] == 0:
        cor += 1

correct_percentage = (cor * 100) / len(y_test)
print("correct prediction= ", correct_percentage, "%")

