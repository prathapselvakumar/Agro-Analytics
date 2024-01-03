import streamlit as st
import pandas as pd
import numpy as np
import ultralytics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(layout="wide",initial_sidebar_state="expanded",page_icon="🌾")
sidebar_html = """
<style>
    [data-testid="stSidebar"]{
    background-image: url(https://i.pinimg.com/564x/32/46/02/324602f9aa99c6c00892811a4398e634.jpg);
    background-size: cover;
    }
</style>    
"""
st.markdown(sidebar_html, unsafe_allow_html=True)


if st.session_state['my_input']=='success':
    # st.title("Crop Recommendation")
    weather_theme_html = """
    <div style="background-color: rgba(255, 255, 0, 0.3); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: #000000; font-size: 36px;">Welcome to the Crop Recommendation Section!</h1>
        <img src="https://img.icons8.com/?size=256&id=1521&format=png" alt="Crop Icon" style="width: 100px; height: 100px;">
        <p style="color: #000000; font-size: 18px;">Enter your Temperature , Humidity % , pH , Rain % in your area for our Recommendation</p>
    </div>
    """

    st.markdown(weather_theme_html, unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")

    background_html = """ 
    <style>
    [data-testid="stAppViewContainer"]{
    position: fixed;
    background-image: url(https://images.pexels.com/photos/1838552/pexels-photo-1838552.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1);
    background-size: cover;
    }
    [data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
    }
    </style>
    """

    st.markdown(background_html, unsafe_allow_html=True)
    def load_file():
        #path of crop recommendation dataset
        rec_data = pd.read_csv("dataset/crop_recommendation.csv")
        rec_data=rec_data.drop(['N','P','K'],axis=1)

        label_encoder = LabelEncoder()
        X = rec_data.drop(['label'],axis=1)
        y = label_encoder.fit_transform(rec_data["label"])

        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.15, random_state = 0)

        return X_train, X_test, y_train, y_test, label_encoder

    X_train, X_test, y_train, y_test, le = load_file()

    def RDF(X_train,y_train):

        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=9,random_state=128))
        pipeline.fit(X_train, y_train)

        # Test Data Metrics
        #predictions = pipeline.predict(X_test)
        #accuracy = accuracy_score(y_test, predictions)
        #print(f"Accuracy on Test Data: {accuracy*100}%")

        # Whole Data Metrics
        #predictions = pipeline.predict(X.values)
        #accuracy = accuracy_score(y, predictions)
        #print(f"Accuracy on Whole Data: {accuracy*100}%")

        return pipeline

    def XGB(X_train,y_train):
        pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state=128))
        pipeline.fit(X_train, y_train)

        # Test Data Metrics
        #predictions = knn_pipeline.predict(X_test)
        #accuracy = accuracy_score(y_test, predictions)
        #print(f"Accuracy on Test Data: {accuracy*100}%")

        # Whole Data Metrics
        #predictions = knn_pipeline.predict(X.values)
        #accuracy = accuracy_score(y, predictions)
        #print(f"Accuracy on Whole Data: {accuracy*100}%")

        return pipeline
    
    def NB(X_train, y_train):
        pipeline = make_pipeline(StandardScaler(), GaussianNB())
        pipeline.fit(X_train, y_train)
        return pipeline
    
    # def KNN(X_train, y_train):
    #     pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    #     pipeline.fit(X_train, y_train)
    #     return pipeline
    

    temp = st.number_input("Enter Temperature (in c):", format="%f")
    humi = st.number_input("Enter Humidity (in %):", format="%f")
    ph = st.number_input("Enter pH (1 - 14):",format="%f")
    rain = st.number_input("Enter Rain (in %):",format="%f")


    col1,col2 = st.columns(2)
    with col1:
        st.subheader("XGB Recommendation")
        xgb_pipeline = XGB(X_train,y_train)
        xgb_val = xgb_pipeline.predict(np.array([temp,humi,ph,rain]).reshape(1,-1))
        xgb_val = le.inverse_transform(xgb_val)
        st.write(xgb_val)

        st.subheader("Random Forest Recommendation")
        rdf_pipeline = RDF(X_train,y_train)
        val = rdf_pipeline.predict(np.array([temp,humi,ph,rain]).reshape(1,-1))
        val = le.inverse_transform(val)
        st.write(val)

        st.subheader("Naive Bayes Recommendation")
        nb_pipeline = NB(X_train, y_train)
        nb_val = nb_pipeline.predict(np.array([temp, humi, ph, rain]).reshape(1, -1))
        nb_val = le.inverse_transform(nb_val)
        st.write(nb_val)

        # st.subheader("KNN Recommendation")
        # knn_pipeline = KNN(X_train, y_train)
        # knn_val = knn_pipeline.predict(np.array([temp, humi, ph, rain]).reshape(1, -1))
        # knn_val = le.inverse_transform(knn_val)
        # st.write(knn_val)
    
    with col2:
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.image("assets/accuracycomp.jpg", caption="Accuracy comparison")
        # rotate_html = """ 
        # <style>
        # .stImage>img {
        #     transform: rotate(90deg);
        # }
        # </style>
        # """
        # st.markdown(rotate_html, unsafe_allow_html=True)

        

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Created with ❤️ by Team Byte Bay Bugs")

    st.markdown(
    """
    <style>
        .stButton>button {
            position: fixed;
            bottom: 20px;
            right: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    def reload_page():
        st.experimental_rerun()

    # Add a sample button
    if st.button("Log out"):
        st.session_state["my_input"]=""
        reload_page()

else:
    st.write("Log in to continue")