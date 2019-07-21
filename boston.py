import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('Boston.csv')
print(dataset.head(5))

# Drop
x = dataset.drop(['Unnamed: 0', 'medv'], axis=1)
y = dataset['medv']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)

plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
