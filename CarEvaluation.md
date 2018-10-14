

```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# CarEvaluation Data 불러오기
car_raw = pd.read_csv('C:/Users/HEO/HW1/car.csv')
```


```python
# 첫 5행 살펴보기
car_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 끝 5행 살펴보기
car_raw.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1723</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>med</td>
      <td>med</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1724</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>med</td>
      <td>high</td>
      <td>vgood</td>
    </tr>
    <tr>
      <th>1725</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1726</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>med</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>low</td>
      <td>low</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>high</td>
      <td>vgood</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 각 변수 별 결측치가 있는지,
# 변수형이 numeric인지 nominal인지 알아보기
car_raw.info()

# 모든 변수의 data가 1728개로 결측치가 없고
# 모든 변수의 타입이 object이므로 모두 nominal
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1728 entries, 0 to 1727
    Data columns (total 7 columns):
    buying      1728 non-null object
    maint       1728 non-null object
    doors       1728 non-null object
    persons     1728 non-null object
    lug_boot    1728 non-null object
    safety      1728 non-null object
    class       1728 non-null object
    dtypes: object(7)
    memory usage: 94.6+ KB
    


```python
# 변수별 막대그래프
for var in car_raw.keys():
    plt.subplots(figsize=(7,3))
    plt.title(var)
    for i in dict(car_raw.groupby([var])[var].count()):
        plt.bar(i, dict(car_raw.groupby([var])[var].count())[i])
    plt.show()
    
# 모든 변수의 속성들이 균일하게 분포되어 있음(문제없음)
```


![png](output_5_0.png)



![png](output_5_1.png)



![png](output_5_2.png)



![png](output_5_3.png)



![png](output_5_4.png)



![png](output_5_5.png)



![png](output_5_6.png)



```python
# 설명변수(x)와 반응변수(y) 분리
y = car_raw['class']
x = car_raw.drop(columns=['class'])

# 반응변수(y) 예측 결과를 순서대로 나타내기 위해 숫자 부여함
y[y == 'unacc'] = '1unacc'
y[y == 'acc'] = '2acc'
y[y == 'good'] = '3good'
y[y == 'vgood'] = '4vgood'


# 설명변수의 변수형이 문자형일 경우 classification이 되지 않으므로 dummy 변수를 만들어줌
# 각 dummy 변수는 0 또는 1의 값을 갖게 됨
# dummy 변수만 포함되어있는 x_dummy라는 새로운 데이터프레임 생성
x_dummy = x[:]
for var in x.keys():
    x_dummy = x_dummy.merge(pd.get_dummies(x[var], prefix=var), left_index=True, right_index=True)
    
x_dummy = x_dummy.drop(columns=x.keys())
x_dummy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying_high</th>
      <th>buying_low</th>
      <th>buying_med</th>
      <th>buying_vhigh</th>
      <th>maint_high</th>
      <th>maint_low</th>
      <th>maint_med</th>
      <th>maint_vhigh</th>
      <th>doors_2</th>
      <th>doors_3</th>
      <th>...</th>
      <th>doors_5more</th>
      <th>persons_2</th>
      <th>persons_4</th>
      <th>persons_more</th>
      <th>lug_boot_big</th>
      <th>lug_boot_med</th>
      <th>lug_boot_small</th>
      <th>safety_high</th>
      <th>safety_low</th>
      <th>safety_med</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# ZeroR
# 분류의 기준이 되는 attribute가 없고
# 가장 높은 확률의 class로 무조건 예측하여 정확도를 높임

from sklearn.dummy import DummyClassifier
zeroR = DummyClassifier(strategy='most_frequent')
zeroR.fit(x_dummy,y)
y_pred = zeroR.predict(x_dummy)

print('Accuracy:',zeroR.score(x_dummy,y))
pd.crosstab(y,y_pred)

# 예측 정확도는 약 70%이며
# 실제 class가 무엇이든 간에 모두 도수가 가장 높은 'unacc'로 예측함
```

    Accuracy: 0.7002314814814815
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>1unacc</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1210</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>384</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>69</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ID3

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy")
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 100%이다(과적합 가능성)
```

    Accuracy: 1.0
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
      <th>3good</th>
      <th>4vgood</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1210</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>0</td>
      <td>384</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ID3 overfitting 방지1: min_samples_leaf 높게 설정
# 최소 leaf 1000개로 지정

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1000)
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# min_samples_leaf를 지나치게 높게 설정하면 zeroR과 같은 결과를 얻는다(과소적합)
```

    Accuracy: 0.7002314814814815
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1210</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>384</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>69</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ID3 overfitting 방지1: min_samples_leaf 높게 설정
# 최소 leaf 10개로 지정

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10)
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 95%이다
```

    Accuracy: 0.9548611111111112
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
      <th>3good</th>
      <th>4vgood</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1181</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>18</td>
      <td>351</td>
      <td>9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>1</td>
      <td>6</td>
      <td>53</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ID3 overfitting 방지2: min_weight_fraction_leaf 높게 설정
# 각 class에서 비슷한 수의 샘플을 추출해 균형을 맞춘다
# 수가 많고 지배적인 class로 결과가 편향되는 것을 방지함
# 가지를 지나치게 많이 치는 것을 방지하는 pre-pruning

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy", min_weight_fraction_leaf=0.5)
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# min_weight_fraction_leaf를 최대값인 0.5로 설정하면 zeroR과 같은 결과를 얻는다(과소적합)
```

    Accuracy: 0.7002314814814815
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1210</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>384</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>69</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# C45

```


```python
# NaiveBayes

from sklearn.naive_bayes import MultinomialNB
NaiveBayes = MultinomialNB()
NaiveBayes.fit(x_dummy,y)
y_pred = NaiveBayes.predict(x_dummy)

print('Accuracy:',NaiveBayes.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 87%이다
```

    Accuracy: 0.8715277777777778
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
      <th>3good</th>
      <th>4vgood</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1163</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>87</td>
      <td>287</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>0</td>
      <td>46</td>
      <td>21</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Logistic Regression

from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(x_dummy,y)
y_pred = LogReg.predict(x_dummy)

print('Accuracy:',LogReg.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

#예측 정확도는 약 89%이다
```

    Accuracy: 0.8912037037037037
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
      <th>3good</th>
      <th>4vgood</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1162</td>
      <td>46</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>47</td>
      <td>327</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>0</td>
      <td>46</td>
      <td>21</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# MLP: hidden layer 100개(default)

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier()
MLP.fit(x_dummy,y)
y_pred = MLP.predict(x_dummy)

print('Accuracy:',MLP.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 하나의 데이터를 제외하고 모두 알맞게 예측하였다
```

    Accuracy: 0.9994212962962963
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
      <th>3good</th>
      <th>4vgood</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1210</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>0</td>
      <td>383</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# MLP: hidden layer 1개

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(1,))
MLP.fit(x_dummy,y)
y_pred = MLP.predict(x_dummy)

print('Accuracy:',MLP.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 82%이다(unacc, acc으로만 예측이 이루어졌다. 과소적합)
```

    Accuracy: 0.8234953703703703
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1131</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>92</td>
      <td>292</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>3</td>
      <td>66</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
# MLP: hidden layer 10개

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(10,))
MLP.fit(x_dummy,y)
y_pred = MLP.predict(x_dummy)

print('Accuracy:',MLP.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 95%이다
```

    Accuracy: 0.9508101851851852
    

    C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1unacc</th>
      <th>2acc</th>
      <th>3good</th>
      <th>4vgood</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1unacc</th>
      <td>1173</td>
      <td>36</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2acc</th>
      <td>15</td>
      <td>358</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3good</th>
      <td>0</td>
      <td>15</td>
      <td>48</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4vgood</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>


