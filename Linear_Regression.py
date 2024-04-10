import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
%matplotlib inline
df = pd.read_csv('Ecommerce Customers')
df.head()
df.describe()
df.info()
sns.jointplot(x='Time on Website',y='Yearly Amount Spent', data=df)
sns.jointplot(x='Time on App',y='Yearly Amount Spent', data=df)
sns.jointplot(x='Time on App', y='Length of Membership',data=df,kind= 'hex', color='grey')
sns.pairplot(df)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y= df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
pred=lm.predict(X_test)
plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
sns.distplot((y_test-pred),bins=50)
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

