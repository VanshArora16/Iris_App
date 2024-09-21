import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

import joblib


iris_df = pd.read_csv(r"Iris.csv")


iris_df = pd.read_csv(r"Iris.csv")
iris_df.sample(frac=1, random_state=40)
x = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size= .30, random_state = 40)

rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(x_train,y_train)

rfc_pred =rfc.predict(x_test)
acc_train = round(100*(rfc.score(x_train,y_train)),2)
acc_val = round(100*(rfc.score(x_test,y_test)),2)
print(f"training accuracy : {acc_train}")
print(f"testing accuracy : {acc_val}")


print(classification_report(y_test, rfc_pred))

print(confusion_matrix(y_test, rfc_pred))

print(round(100*accuracy_score(y_test, rfc_pred),2))

joblib.dump(rfc,"rf_model.sav")

import streamlit as st # help in deplying model
import  pandas as pd
import numpy as np
from prediction import predict

st.title("Classifying Iris Flowers")
st.markdown("Toy model to classify iris flowers into setosa , versicolor, virginica")

st.header("Plant Features")
col1, col2 = st.columns(2)
with col1:

    st.text("sepal characteristics")
    sepal_l = st.slider("sepal length(cm)" , 1.0, 8.0, 0.5)
    sepal_w = st.slider("sepal width(cm)" , 2.0, 4.4, 0.5)

with col2:

    st.text("petal characteristics")
    petal_l = st.slider("petal length(cm)" , 1.0, 7.0, 0.5)
    petal_w = st.slider("petal width(cm)" , 0.1, 2.5, 0.5)

if st.button("Predict iris Flower"):
    result = predict(np.array([[sepal_l,sepal_w,petal_l,petal_w]]))
    st.text(result[0])