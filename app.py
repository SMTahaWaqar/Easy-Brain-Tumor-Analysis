import numpy as np
import google.generativeai as genai
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv

load_dotenv()

model = load_model('/kaggle/working/model.h5')

# Function to run the Keras classification model and get the predicted class
def predict_class(tumor_image):
    img = image.load_img(tumor_image,target_size=(1250,1250))
    image_array = np.asarray(img)
    img = np.expand_dims(image_array, axis=0)
    prediction = model.predict(img)

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    index_of_1 = prediction[0].tolist().index(1)
    predicted_class = classes[index_of_1]
    print("Predicted class:", predicted_class)
    return predicted_class

# Function to generate report using Gemini Pro model
def generate_report(class_name):
    # # Call Gemini Pro API to generate report
        GOOGLE_API_KEY=os.getenv('GEMINI-API-KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.0-pro-latest')
        if class_name == 'notumor':
            response = 'Congratulations! Your scan showed no signs of tumor'
            print(response)
            return response
        elif class_name == 'glioma' or class_name == 'meningioma' or class_name == 'pituitary':
            response = model.generate_content(f"A patient has received a scan result indicating signs of {class_name} brain tumor. Their doctor is unavailable. Write a 5-6 line explanation focused on reassurance and the next steps. Emphasize the need for further medical consultation. Avoid making claims about cure chances, as these depend on individual factors. Do not use overly technical language, and maintain a sensitive tone.")
            print(response)
            return response.text


st.title("NeuroSight ")
st.write("###### Disclaimer: **We do not provide any kind of professional medical advice**")

with st.sidebar:
    selected = option_menu("NeuroSight ", ["Home", 'About'],
        icons=['house-fill', 'bi-people-fill'], menu_icon="bi-file-earmark-medical-fill", default_index=0)

# Home Page
if selected == 'Home':
    st.header("Tumor Detection and Report Generation")
    st.subheader("Upload an image for classification")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Report", key="generate_report", use_container_width=True):
            report_placeholder = st.empty()
            report_placeholder.write("Generating report...")
            # Run the classification model
            class_name = predict_class(uploaded_image)

            # Generate report using Gemini Pro
            report_summary = generate_report(class_name)

            report_placeholder.empty()
           # Display the predicted class and report summary
            if class_name == 'notumor':
                st.subheader("No Tumor Detected")
            else:
                st.subheader("Detected Tumor Class")
                st.write(class_name)

            st.subheader("Report Summary")
            st.info(report_summary)

elif selected == 'About':
    st.write("## About")
    st.write("This web application uses machine learning to classify brain tumor and  Gemini 1.0-Pro-Latest to provide user-friendly non-medical explanation of the detected tumor in a summarized manner.")
    st.write("### Our Team")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Syed Riaz Ali')
        st.write('Mohammad Rafay Khan')
    with col2:
        st.write('S.M. Taha Waqar')
        st.write('Hamza bin Ashraf')
    with col3:
        st.write('Syed Maaz Ali')
