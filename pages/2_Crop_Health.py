import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="🌾")
sidebar_html = """
<style>
    [data-testid="stSidebar"]{
    background-image: url(https://i.pinimg.com/564x/32/46/02/324602f9aa99c6c00892811a4398e634.jpg);
    background-size: cover;
    }
</style>    
"""
st.markdown(sidebar_html, unsafe_allow_html=True)

# Initialize session state
if 'my_input' not in st.session_state:
    st.session_state.my_input = ''

if st.session_state.my_input == 'success':
    def save_image(file_upload, folder_path):
        file = file_upload.read()
        file_name = file_upload.name

        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the image to the folder
        with open(os.path.join(folder_path, file_name), "wb") as f:
            f.write(file)

    model_path = '/Volumes/Prathap Docs/Presentation and Study Material/Project/GAIP/AgroAnalytics/model/best.pt'

    try:
        # Load the YOLO model
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        st.stop()

    background_html = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url(https://img.freepik.com/free-photo/hydroponics-system-planting-vegetables-herbs-without-using-soil-health_1150-8154.jpg?w=1380&t=st=1703845101~exp=1703845701~hmac=672954cc2e175b8dd76645ddc7341c1d13d7ffb985fe71de9d10f6d5d685c5de);
    background-size: cover;
    }
    [data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(background_html, unsafe_allow_html=True)

    division_box_html = """
    <div style="background-color: rgba(255,255,255,0.4); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: #000000; font-size: 36px;">Welcome to the Crop Health Monitoring Section!</h1>
        <img width="96" height="96" src="https://img.icons8.com/material-sharp/96/trust--v1.png" alt="trust--v1"/>
        <p style="color: #000000; font-size: 18px;">Upload a picture of your crop's leaves and get accurate crop health results!</p>
    </div>
    """

    st.markdown(division_box_html, unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")
    uploaded_file = st.file_uploader("Upload your image here...", type=["jpg"])
    
    if uploaded_file is not None:
        save_image(uploaded_file, "temp")

        try:
            # Predict using the YOLO model
            results = model.predict(source="temp")
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                st.image(im)
        except Exception as e:
            st.error(f"Error making predictions with YOLO model: {str(e)}")

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

    if st.button("Log out"):
        st.session_state.my_input = ""
        reload_page()

else:
    st.write("Log in to continue")
