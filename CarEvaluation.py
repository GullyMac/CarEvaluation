
# coding: utf-8

# In[195]:


import pandas as pd
import matplotlib.pyplot as plt


# In[206]:


# CarEvaluation Data 불러오기
car_raw = pd.read_csv('C:/Users/HEO/HW1/car.csv')


# In[208]:


# 첫 5행 살펴보기
car_raw.head()


# In[209]:


# 끝 5행 살펴보기
car_raw.tail()


# In[210]:


# 각 변수 별 결측치가 있는지,
# 변수형이 numeric인지 nominal인지 알아보기
car_raw.info()

# 모든 변수의 data가 1728개로 결측치가 없고
# 모든 변수의 타입이 object이므로 모두 nominal


# In[211]:


# 변수별 막대그래프
for var in car_raw.keys():
    plt.subplots(figsize=(7,3))
    plt.title(var)
    for i in dict(car_raw.groupby([var])[var].count()):
        plt.bar(i, dict(car_raw.groupby([var])[var].count())[i])
    plt.show()
    
# 모든 변수의 속성들이 균일하게 분포되어 있음(문제없음)


# In[212]:


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


# In[213]:


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


# In[224]:


# ID3

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy")
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 100%이다(과적합 가능성)


# In[229]:


# ID3 overfitting 방지1: min_samples_leaf 높게 설정
# 최소 leaf 1000개로 지정

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1000)
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# min_samples_leaf를 지나치게 높게 설정하면 zeroR과 같은 결과를 얻는다(과소적합)


# In[231]:


# ID3 overfitting 방지1: min_samples_leaf 높게 설정
# 최소 leaf 10개로 지정

from sklearn import tree
ID3 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10)
ID3.fit(x_dummy,y)
y_pred = ID3.predict(x_dummy)

print('Accuracy:',ID3.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 95%이다


# In[227]:


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


# In[203]:


# C45


# In[223]:


# NaiveBayes

from sklearn.naive_bayes import MultinomialNB
NaiveBayes = MultinomialNB()
NaiveBayes.fit(x_dummy,y)
y_pred = NaiveBayes.predict(x_dummy)

print('Accuracy:',NaiveBayes.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 87%이다


# In[221]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(x_dummy,y)
y_pred = LogReg.predict(x_dummy)

print('Accuracy:',LogReg.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

#예측 정확도는 약 89%이다


# In[232]:


# MLP: hidden layer 100개(default)

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier()
MLP.fit(x_dummy,y)
y_pred = MLP.predict(x_dummy)

print('Accuracy:',MLP.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 하나의 데이터를 제외하고 모두 알맞게 예측하였다


# In[236]:


# MLP: hidden layer 1개

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(1,))
MLP.fit(x_dummy,y)
y_pred = MLP.predict(x_dummy)

print('Accuracy:',MLP.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 82%이다(unacc, acc으로만 예측이 이루어졌다. 과소적합)


# In[237]:


# MLP: hidden layer 10개

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(10,))
MLP.fit(x_dummy,y)
y_pred = MLP.predict(x_dummy)

print('Accuracy:',MLP.score(x_dummy,y))
pd.crosstab(y,y_pred, rownames=['True'], colnames=['Predicted'])

# 예측 정확도는 약 95%이다

