import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
col_names = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A/G','Patient']
#Training file
df_train = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv',names=col_names,index_col=False)
# General overview of the dataset
df_train.info()
# Is there any imbalance?
df_train['Gender'].value_counts()

# Separating the target column
y = df_train['Patient']
X = df_train.drop(['Patient'],axis=1)

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
#
categorical_pipe = Pipeline([('onehot', OneHotEncoder())])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ("scaler", StandardScaler())
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, ['Gender']),('num', numerical_pipe, ['Age','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A/G'])],remainder='passthrough'
    )
#preprocessing.fit(X)
#preprocessing.transform(X)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
#from sklearn.externals import joblib

#model = RandomForestClassifier(random_state=42,n_estimators = 1000)
model = XGBClassifier(random_state=42,n_estimators=1000,max_depth=8,learning_rate=0.005)
pipeline = Pipeline(steps=[('prep',preprocessing), ('m', model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

pipeline.fit(X,y)
feature = [65,'Male',7.3,4.1,500,60,65,6,3,0.60]
col_names_test = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A/G']
testdata = pd.DataFrame([feature],columns=col_names_test)
#print(testdata.shape)
pred = pipeline.predict(testdata)
if pred ==1:
    print("Its one")
#pickle.dump(pipeline,open('model.pkl','wb'))
