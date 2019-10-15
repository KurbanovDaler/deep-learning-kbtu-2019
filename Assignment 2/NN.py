

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
import keras 
from keras.models import Sequential
from keras.layers import Dense



col_names=['PassengerId',
           'Survived',
           'Pclass',
           'Name',
           'Sex',
           'Age', 
           'SibSp',
           'Parch',
           'Ticket',
           'Fare',
           'Cabin',
           'Embarked']



#ds=pd.read_csv('train.csv', ',', names=col_names, index_col=False)
ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')
ds.head()



ds.shape

print("sup")


ds.info()



ds['Sex'].replace(['female','male'], [0,1], inplace = True)
test_ds['Sex'].replace(['female','male'], [0,1], inplace = True)



drop_names = ['PassengerId',
              'Name',
              'Ticket',
              'Fare',
              'Embarked',
              'Cabin']
drop_names_test = ['Name',
                   'Ticket',
                   'Fare',
                   'Embarked',
                   'Cabin']



ds.fillna(ds.mean(), inplace = True)
test_ds.fillna(test_ds.mean(), inplace = True)



ds = ds.drop(columns = drop_names)
test_ds = test_ds.drop(columns = drop_names_test)
test_ds.head()



ds.describe()



test_ds.describe()



ds.Age.hist()



ds.corr()



import seaborn as sns
corr=ds.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))



ds[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)



ds[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)



ds[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)



ds[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



ds[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)



g = sns.FacetGrid(ds, col='Survived')
g.map(plt.hist, 'Age', bins=20)



g = sns.FacetGrid(ds, col='Survived')
g.map(plt.hist, 'Pclass', bins=20)



grid = sns.FacetGrid(ds, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();




# X_train = ds.drop("Survived", axis=1)
# Y_train = ds["Survived"]
# X_test = test_ds.drop(columns = 'PassengerId')
# #X_test  = ds.drop("PassengerId", axis=1).copy()
# X_train.shape, Y_train.shape, X_test.shape



# ds.head()



# X_test.head()



# X_test.tail()



# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# acc_log



# pd.DataFrame({
#         "PassengerId": test_ds["PassengerId"],
#         "Survived": Y_pred
#     }).to_csv('submission.csv', index=False)


X_train = ds.drop("Survived", axis=1)
Y_train = ds["Survived"]
X_test = test_ds.drop(columns = 'PassengerId')

model = Sequential()

model.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
model.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, Y_train, batch_size = 64, epochs = 300)

y_pred = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': test_ds['PassengerId'], 'Survived': y_final})
output.to_csv('prediction-ann.csv', index=False)