import streamlit as st
from face_auth import register_user, login_user

st.set_page_config(page_title="Face Login", layout="centered")
st.title("🔐 Face Login System")

option = st.sidebar.radio("Choose Option", ("Login", "Register"))

if option == "Register":
    name = st.text_input("Enter your name")
    if st.button("Register"):
        result = register_user(name)
        st.success(result)

elif option == "Login":
    if st.button("Scan Face to Login"):
        result = login_user()
        if result.startswith("Welcome"):
            name = result.split(",")[1].split("!")[0].strip()
            image_url = result.split("Image: ")[1]
            st.success(f"Welcome back, {name}!")
            st.image(image_url, caption=f"{name}'s face", width=250)
        else:
            st.error(result)