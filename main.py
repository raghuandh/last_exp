import streamlit as st


from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


import pandas as pd
df = pd.read_csv("C:/Users/mirya/Downloads/datasets_we/iris_dataset.csv")
data = df.values
X = data[:,0:4]
Y = data[:,4]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

svn = SVC()
svn.fit(X_train, y_train)
pred = svn.predict(X_test)



st.title('ML Iris Classifier')
sp_len=st.number_input('Enter Sepal Length')
sp_wid = st.number_input('Enter Sepal Width')
pt_len = st.number_input('Enter Petal Length')
pt_wid =st.number_input('Enter Petal Widht')
feature_data = [[sp_len,sp_wid,pt_len,pt_wid]]
predict_but = st.button('Predict')
if predict_but:
    prediction=svn.predict(feature_data)
    st.write("Prediction: {}".format(prediction))









