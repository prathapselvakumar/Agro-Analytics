# app.py

import streamlit as st
import mysql.connector
from passlib.hash import pbkdf2_sha256
from PIL import Image

# Function to create a MySQL connection
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="crops"
    )

# Function to create users table if not exists
def create_users_table(connection):
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            password VARCHAR(100) NOT NULL
        )
    ''')
    connection.commit()

# Function to register a new user
def register_user(connection, username, password):
    cursor = connection.cursor()
    hashed_password = pbkdf2_sha256.hash(password)
    cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, hashed_password))
    connection.commit()

# Function to authenticate a user
def authenticate_user(connection, username, password):
    cursor = connection.cursor()
    cursor.execute('SELECT password FROM users WHERE username = %s', (username,))
    result = cursor.fetchone()
    if result:
        hashed_password = result[0]
        return pbkdf2_sha256.verify(password, hashed_password)
    return False

# Streamlit app
def main():
    st.set_page_config(layout="centered",page_icon="🌾")
    st.title("Agro Analytics")
    background_style = """
        <style>
        [data-testid="stAppViewContainer"]{
        position: fixed;
        background-image: url(https://i.pinimg.com/564x/64/7c/f7/647cf73d470a70dbeb38471e6d481073.jpg);
        background-size: cover;
        }
        [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);
        }
        [data-testid="stSidebar"]{
        background-image: url(https://i.pinimg.com/564x/32/46/02/324602f9aa99c6c00892811a4398e634.jpg);
        background-size: cover;
        }
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

    connection = create_connection()
    create_users_table(connection)
    if "my_input" not in st.session_state:
        st.session_state["my_input"] = ""


    page = st.sidebar.radio("", ["Login", "Register"])

    if page == "Register":
        st.header("Register")
        new_username = st.text_input("Enter your username:")
        new_password = st.text_input("Enter your password:", type="password")

        if st.button("Register"):
            register_user(connection, new_username, new_password)
            st.success("Registration successful! Please go to the login page.")

    elif page == "Login":
        st.header("Login")
        username = st.text_input("Enter your username:")
        password = st.text_input("Enter your password:", type="password")

        if st.button("Login"):
            if authenticate_user(connection, username, password):
                st.success("Login successful!")
                # Set the login_successful attribute to True
                st.session_state["my_input"] = "success"
            else:
                st.error("Authentication failed. Invalid username or password.")
    
    centered_html = """
    <style>
        div.stApp {
            text-align: center;
        }
    </style>
    """
    st.markdown(centered_html, unsafe_allow_html=True)

    # img_path="C:/Users/mrthu/Desktop/NUS/New folder/assets/logo.jpg"
    # image_html = """
    # <div style="position: absolute; top: 10px; left: 10px;">
    #     <img src="{img_path}" alt="Logo" width="50">
    # </div>
    # """
    # st.markdown(image_html, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
