import numpy as np
import pandas as pd
from sklearn import svm
#Visual Data
import matplotlib.pyplot as plt
import seaborn  as sns; sns.set(font_scale =1.2)

recipes = pd.read_csv("Men vs Femal.csv")
print(recipes.head())

#plot the Data
sns.lmplot(x="Height", y="Wight", data=recipes, hue="Gender", palette="Set1", fit_reg= False,scatter_kws={"s": 70});

#forat for processig Data
type_label =np.where(recipes["Gender"]=="Male", 0,1)
recipes_features = recipes.columns.values[1:].tolist
ingredients = recipes[["Height", "Wight"]].values
print (ingredients)

model = svm.SVC(kernel="linear", decision_function_shape="ovr")
model.fit(ingredients, type_label)

b = model.support_vectors_[0]
c = model.support_vectors_[-1]
w = model.coef_[0]
a = -w[0]/ w[1]
xx = np.linspace(150, 180)
yy = a*xx-(model.intercept_[0])/w[1]
yy_down = a * xx + (b[1] - a * b[0])
yy_up = a * xx + (c[1] - a * c[0])

plt.plot(xx, yy, linewidth = 2, color = "black")
plt.plot(xx, yy_down, "k--")
plt.plot(xx, yy_up, "k--")

def Prediction_Function(height, wight):
    if model.predict([[height, wight]])==0:
        print("This is Male's Data")
    else:
        print("This is a Female's Data")

Prediction_Function(170, 180)

plt.plot(170, 160, markersize = "9")

#Display the plot
plt.show()