

```python
import re
import numpy as np
import pandas as pd
import sklearn
```


```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
```


```python
#open up the data (which is in .txt format)

with open("pima-indians-diabetes.data.txt") as script:
    data = script.read()
```


```python
#split text into strings corresponding to each instance

data = re.split("\n|,",data)
```


```python
#construct a random data frame to be filled in later

df = pd.DataFrame(np.random.randn(769, 9),columns =['# times pregnant','Glucose after 2 hours','Blood Pressure','Tricept skin fold thickness','2hr serum insulin','BMI','Diabetes pedigree function','Age','class'])

```


```python
#convert from list of strings to data frame

for j in range(0,768):
    for i in range(0,9):
        df.iloc[j,i] = data[i+9*j]
```


```python
#drop last row (to clean up)
df = df[:-1]
```


```python
y = df[['class']].copy()

x = df[['# times pregnant','Glucose after 2 hours','Blood Pressure','Tricept skin fold thickness','2hr serum insulin','BMI','Diabetes pedigree function','Age']].copy()

```


```python
#Accuracy seems to be maximized with test_size=0.30

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=127)
```


```python
diabetes_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
diabetes_classifier.fit(x_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=10, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=0,
                splitter='best')




```python
pred = diabetes_classifier.predict(x_test)
```


```python
accuracy_score(y_true = y_test, y_pred = pred)
```




    0.76623376623376627




```python
##Support Vector Machine Classifier



#Convert from dataframe to lists of floats

y_array = []
x_array = []
y_test_array = []
x_test_array = []


for index in range(0,y_train.shape[0]):
    y_array.extend(list(map(int,y_train.iloc[index,:])))
for index in range(0,x_train.shape[0]):
    x_array.append(list(map(float,x_train.iloc[index,:])))

    
for index in range(0,y_test.shape[0]):
    y_test_array.extend(list(map(int,y_test.iloc[index,:])))
for index in range(0,x_test.shape[0]):
    x_test_array.append(list(map(float,x_test.iloc[index,:])))


#print(x_test)

```


```python
clf = svm.SVC()
clf.fit(x_array, y_array)  
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
test_pred = []

for index in range(0,x_test.shape[0]):
    test_pred.extend(clf.predict(x_test_array[index:index+1]))
    
#test_pred
```


```python
##Naive Bayes Classifier



gnb = GaussianNB()
pred = gnb.fit(x_array, y_array).predict(x_test_array)

match = 0
non_match = 0

for index in range(0,len(y_test_array)):
    if y_test_array[index] == pred[index]:
        match += 1 
    else:
        non_match += 1
        
print("Accuracy: " + str(match*100/(match+non_match)) + "%")

```

    Accuracy: 76.19047619047619%



```python

```
