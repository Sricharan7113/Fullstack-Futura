import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
payment = pd.read_csv("payment_data.csv")
customer = pd.read_csv("customer_data.csv")
payment.isnull().sum()
payment1 = payment.drop(columns=['prod_limit','report_date','update_date'])
payment1.info()
payment1['highest_balance'].describe()
payment1['OVD_t2'].unique()
customer
customer.isnull().sum()
data = pd.concat([payment1,customer],axis=1)
data.info()
data['OVD_sum']
sns.barplot(x=data['label'],y = data['pay_normal'])
data['pay_normal'].describe()
sns.lineplot(x=data['label'],y=data['prod_code'],)
sns.barplot(x=data['label'],y = data['highest_balance'])
sns.barplot(x=data['label'],y = data['new_balance'])
data.describe()
df1 = data[data['label']==1]
df1.describe()
from sklearn.impute import SimpleImputer
simp = SimpleImputer(missing_values = np.nan,strategy='mean')
def imp(i=""):
    simp = SimpleImputer(missing_values = np.nan,strategy='mean')
    data[i] = simp.fit_transform(data[[i]])
data.columns
for i in data.columns:
    print(imp(i))
data[data['label']==1].head()
data[data['label']==0].head()
data.head()
data.isnull().sum()
data['label'] = data['label'].astype('int64')
X = data.drop(columns=['label'],axis = 0)
Y = data['label'] 
(x_train,x_test,y_train,y_test) = train_test_split(X,Y,test_size=0.2)
log = LogisticRegression()
model = log.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc*100)
model_params ={
    'LogisticRegression':{
        'model':LogisticRegression( solver='liblinear',multi_class = 'auto'),
        'params':{
            'C':[1,5,10]
        }
    },
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'params':{
            'criterion':['gini','log_loss','entropy'],
            'splitter':['best','random']
        }
    },
    'RandomForestRegression':{
        'model':RandomForestRegressor(),
        'params':{
        'n_estimators':[100,150,200]
        }
    }
}
scores = []
for model_name , mp in model_params.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5)
    clf.fit(x_train,y_train)
    scores.append({'model': model_name,'best_score':clf.best_score_,'best_params':clf.best_params_})
pd.DataFrame(scores)

pickle_out = open("pickle_file.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

import sklearn_json as skljson
skljson.to_json(model,"model.json")
