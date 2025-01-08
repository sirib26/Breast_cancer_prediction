import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import time
from preprocess import (
    encode_age, 
    convert_race_to_int,
    convert_breast_cancer_history_to_int,
    convert_age_menarche_to_int,
    convert_age_first_birth_to_int,
    convert_birads_breast_density_to_int,
    convert_current_hrt_to_int,
    convert_menopausal_status_to_int,
    convert_bmi_to_group,
    convert_biophx_to_int
)

# Load the trained model
with open("model/cancer_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/model.pkl", "rb") as file:
    model2 = pickle.load(file)


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

# App-wide styling
st.markdown(
    """
    <style>
    .main {
        background-color: #fce4ec;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .header {
        color: #e91e63;
        text-align: center;
        font-weight: bold;
    }
    .description {
        font-size: 28px;
        color: #555;
        margin-bottom: 20px;
    }
    .button {
        background-color: #e91e63;
        color: white;
        font-size: 16px;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f8bbd0;
        color: #333;
    }
    .image-container {
        text-align: center;
        margin: 20px 0;
    }
    .image-container img {
        width: 300px;
        height: auto;
        border-radius: 10px;
    }
    .description ul {
        font-size: 20px;  /* Increase font size */
    }
    .description li {
        font-size: 20px;  /* Increase font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)
page_bg_img = '''
<style>
body {
    background-image: url("https://img.freepik.com/free-vector/realistic-breast-cancer-awareness-month-background_52683-70381.jpg?semt=ais_hybrid");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# Load the CSS into the app
st.markdown(page_bg_img, unsafe_allow_html=True)

# App title
st.markdown("<h1 class='header' style='color: #AA336A;'>Breast Cancer Detection</h1>", unsafe_allow_html=True)

# Create a navigation menu for the home page
menu = ["Home", "Diagnosis", "Helpline", "About"]
selection = st.sidebar.selectbox("Select an option", menu)

# Home Page
if selection == "Home":
    # import time
    if "helpline_displayed" in st.session_state:
        del st.session_state["helpline_displayed"] 
    # Define the text to animate
    app_name = "Welcome to the Breast Cancer Detection!"
    
    # Create a character-by-character cycling effect
    import time

# Dynamic Text Animation
    app_name = "Welcome to the Breast Cancer Detection!"
    text_placeholder = st.empty()
    st.markdown(
        """
        <div class="description">
            This app provides a simple and effective way to assist in the early detection of breast cancer.
            Use the navigation menu to explore the following features:
            <ul>
                <li><strong>About:</strong> Learn more about breast cancer and this app.</li>
                <li><strong>Helpline:</strong> Get support and contact information.</li>
                <li><strong>Diagnosis:</strong> Predict the chances of breast cancer based on input features.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='image-container'><img src='https://cdn.shopify.com/s/files/1/0663/8341/3490/files/unnamed.jpg?v=1697220555' alt='Breast Cancer Awareness'></div>", unsafe_allow_html=True)

    for _ in range(5):  # Adjust the loop count for the number of animation loops
        for i in range(len(app_name) + 1):
            text_placeholder.markdown(
                f"<div style='text-align: center; font-size: 28px; font-weight: bold; color: #AA336A;'>{app_name[:i]}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.1)
    for i in range(len(app_name), -1, -1):
        text_placeholder.markdown(
            f"<div style='text-align: center; font-size: 28px; font-weight: bold; color: #AA336A;'>{app_name[:i]}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.1)


    

# About Page
elif selection == "About":
    st.header("About Breast Cancer Detection")
    st.markdown(
        """
        ### 🌟 About the Project
        Welcome to the **Breast Cancer Detection App**, a cutting-edge application designed to empower patients, caregivers, and healthcare professionals in the fight against breast cancer. Leveraging the power of **artificial intelligence** and **machine learning**, this app provides accurate and early predictions of breast cancer based on input features, helping to save lives through timely intervention.

        ### 🔍 What is Breast Cancer?
        Breast cancer is one of the most prevalent cancers worldwide, affecting millions of women and men each year. It occurs when abnormal cells in the breast multiply uncontrollably, forming a lump or mass. While the causes of breast cancer are not fully understood, factors such as genetics, lifestyle, and age contribute to its development.  
        
        #### Why Early Detection Matters:
        - **Improved Survival Rates**: Early detection can lead to more effective treatment and higher survival rates.  
        - **Better Treatment Options**: Diagnosing cancer at an early stage allows for less invasive treatments.  
        - **Awareness Saves Lives**: Regular screenings and awareness campaigns significantly reduce the risk of late-stage diagnosis.

        ### 🎯 Objective of the App
        The **primary goal** of this app is to assist in the early detection and classification of breast cancer. By analyzing patient data such as tumor size, texture, and other features, the app predicts whether the tumor is **Benign** (non-cancerous) or **Malignant** (cancerous).  
        
        #### Key Features:
        - **User-Friendly Interface**: A simple and intuitive design for patients and healthcare providers.
        - **Accurate Predictions**: Powered by a meticulously trained machine learning model.
        - **Decision Support**: Offers actionable insights to guide next steps in healthcare.

        ### 🧠 How the Machine Learning Model Works
        At the heart of this app lies a powerful **Logistic Regression model** trained on a comprehensive dataset of breast cancer cases. The dataset includes features such as:  
        - Tumor dimensions (e.g., mean radius, perimeter, and area)  
        - Cellular characteristics (e.g., texture, smoothness, and compactness)  
        - Advanced metrics (e.g., fractal dimension and symmetry)  

        The model is capable of:
        - **Binary Classification**: Determining whether the tumor is benign or malignant.  
        - **Risk Estimation**: Providing probabilities for more informed decision-making.

        ### 🌈 Our Vision
        Our vision is to create a world where every individual has access to tools that can **detect cancer early**, leading to timely treatment and improved outcomes. By combining technology with healthcare, this app aspires to bridge the gap in cancer care and bring a ray of hope to millions.

        ### 👩‍⚕️ How You Can Use This App
        - **Patients**: Enter your diagnostic data to get predictions and take the next step with confidence.  
        - **Doctors**: Use the app as a supplementary tool for patient consultations.  
        - **Researchers**: Explore how machine learning is revolutionizing healthcare.

        ### 📊 Fun Facts About Breast Cancer
        1. **Survival Rates Are Increasing**: The 5-year survival rate for early-detected breast cancer is over 90%!  
        2. **Self-Exams Matter**: Regular self-examinations help with early detection.  
        3. **Men Can Get Breast Cancer Too**: Although rare, men account for about 1% of all breast cancer cases.  

        ### ❤️ Join the Fight Against Breast Cancer
        This app is more than just a diagnostic tool; it’s part of a larger mission to raise awareness, foster early detection, and ultimately save lives. Together, we can make a difference.
        """,
        unsafe_allow_html=True,
    )


# Helpline Page
elif selection == "Helpline":
    st.markdown('<div class="helpline-header">Breast Cancer Helpline - Bangalore</div>', unsafe_allow_html=True)

    st.markdown("### Find support and care at these leading hospitals in Bangalore:")
    st.write("Here’s a list of hospitals and organizations that specialize in breast cancer detection, treatment, and support. Click to expand each section for more details.")

    # Hospital 1
    with st.expander("Tata Memorial Hospital - Bangalore"):
        st.markdown('<p class="hospital-name">Tata Memorial Hospital</p>', unsafe_allow_html=True)
        st.markdown("""
            <p class="hospital-details">
            - Address: Hosur Road, Bangalore, Karnataka<br>
            - Phone: +91-80-2659-2750<br>
            - Email: info@tmhbangalore.org<br>
            - Website: <a href="https://tmc.gov.in" target="_blank">tmc.gov.in</a><br>
            - Services: Comprehensive cancer care, early detection, and breast cancer treatment.
            </p>
            """, unsafe_allow_html=True)

    # Hospital 2
    with st.expander("Manipal Hospital - Old Airport Road"):
        st.markdown('<p class="hospital-name">Manipal Hospital</p>', unsafe_allow_html=True)
        st.markdown("""
            <p class="hospital-details">
            - Address: 98, Old Airport Road, Bangalore, Karnataka<br>
            - Phone: +91-80-2502-4444<br>
            - Email: contact@manipalhospitals.com<br>
            - Website: <a href="https://www.manipalhospitals.com" target="_blank">manipalhospitals.com</a><br>
            - Services: Breast cancer screening, mammography, and specialized oncology care.
            </p>
            """, unsafe_allow_html=True)

    # Hospital 3
    with st.expander("Apollo Hospital - Bannerghatta Road"):
        st.markdown('<p class="hospital-name">Apollo Hospital</p>', unsafe_allow_html=True)
        st.markdown("""
            <p class="hospital-details">
            - Address: 154/11, Bannerghatta Main Rd, Bengaluru, Karnataka<br>
            - Phone: +91-80-2630-4050<br>
            - Website: <a href="https://www.apollohospitals.com" target="_blank">apollohospitals.com</a><br>
            - Services: Advanced oncology treatments, breast cancer surgery, and counseling.
            </p>
            """, unsafe_allow_html=True)
        # Hospital 4
    with st.expander("HCG Cancer Centre - K.R. Road, Bangalore"):
        st.markdown('<p class="hospital-name">HCG Cancer Centre</p>', unsafe_allow_html=True)
        st.markdown("""
            <p class="hospital-details">
            - Address: #8, P Kalinga Rao Road, Sampangiram Nagar, K.R. Road, Bengaluru, Karnataka<br>
            - Phone: +91-80-4020-6000<br>
            - Website: <a href="https://www.hcgoncology.com/cancer-centers/hcg-cancer-centre-k-r-road-bengaluru/" target="_blank">hcgoncology.com</a><br>
            - Services: Comprehensive cancer care, advanced diagnostics, and personalized treatment plans.
            </p>
            """, unsafe_allow_html=True)


    # Additional Tip Section
    st.markdown("""
    ### General Tips:
    - Conduct regular self-examinations for early detection.
    - Follow up with your healthcare provider if you experience symptoms such as lumps, pain, or discharge from the nipple.
    """, unsafe_allow_html=True)   

# Diagnosis Page
elif selection == "Diagnosis":
    st.header("Diagnosis")

    # Add option to check the type of cancer or chances of cancer
    diagnosis_option = st.selectbox("Select an option:", ["Check Type of Cancer", "Check Chances of Cancer"])

    # Check Type of Cancer (Existing functionality)
    if diagnosis_option == "Check Type of Cancer":
        st.subheader("Enter Patient Details for Type of Cancer Prediction")

        # Define feature column names
        column_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
            'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
            'area error', 'smoothness error', 'compactness error', 'concavity error', 
            'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 
            'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
            'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]

        # Create input fields for the features with descriptive names
        features = []
        for feature_name in column_names:
            feature_value = st.number_input(f"{feature_name}")
            features.append(feature_value)

        # Prediction button
        if st.button("Predict Type of Cancer"):
            input_data = np.array([features])
            input_data_reshaped = input_data.reshape(1, -1)
            input_data_df = pd.DataFrame(input_data, columns=column_names)

            # Prediction
            prediction = model.predict(input_data_df)

            # Display the result
            if prediction[0] == 0:
                st.success("The result is Malignant!")
            else:
                st.success("The result is Benign!")


    # Check Chances of Cancer (New option)
    elif diagnosis_option == "Check Chances of Cancer":
        st.subheader("Enter Your Details for Cancer Risk Prediction")

        # Input for Year
        year = st.number_input("Enter Current Year", min_value=1900, max_value=2100, value=2023)
        
        # Age Input with validation
        age = st.number_input("Age (Years)", min_value=18, max_value=120, value=36, step=1)
        if age < 18:
            st.warning("Please enter an age of 18 or older for more accurate predictions.")

        # Race/Ethnicity selection
        race_eth = st.selectbox("Race/Ethnicity", 
                                ["Non-Hispanic White", "Non-Hispanic Black", 
                                "Asian/Pacific Islander", "Native American", 
                                "Hispanic", "Other/Mixed", "Unknown"])
        race_encoded = convert_race_to_int(race_eth)

        # Family History (first-degree relative with breast cancer)
        family_history = st.selectbox("Do you have a first-degree relative with breast cancer?", 
                                    ["No", "Yes"])
        family_history_encoded = convert_breast_cancer_history_to_int(family_history)

        # Age at Menarche
        age_menarche = st.number_input("Age at Menarche (First Period)", min_value=0, max_value=20, value=12)
        age_menarche_encoded = convert_age_menarche_to_int(
            "Age <12" if age_menarche < 12 else 
            ("Age 12-13" if age_menarche <= 13 else "Age >14")
        )

        # Age at First Birth with edge case handling for -1 (Nulliparous)
        age_first_birth = st.number_input("Age at First Birth (enter -1 if no children)", min_value=-1, max_value=60, value=0)
        if age_first_birth == -1:
            first_birth_encoded = convert_age_first_birth_to_int("Nulliparous")
        else:
            first_birth_encoded = convert_age_first_birth_to_int(
                "Age < 20" if age_first_birth < 20 else 
                ("Age 20-24" if age_first_birth < 25 else 
                ("Age 25-29" if age_first_birth < 30 else "Age > 30"))
            )

        # BIRADS Breast Density
        BIRADS_breast_density = st.selectbox("Breast Density (BIRADS classification)", 
                                            ["Almost entirely fat", "Scattered fibroglandular densities", 
                                            "Heterogeneously dense", "Extremely dense", 
                                            "Unknown or different measurement system"])
        birads_breast_density_encoded = convert_birads_breast_density_to_int(BIRADS_breast_density)

        # Hormone Replacement Therapy (HRT) status
        current_hrt = st.selectbox("Are you currently on Hormone Replacement Therapy (HRT)?", 
                                ["No", "Yes"])
        current_hrt_encoded = convert_current_hrt_to_int(current_hrt)

        # Menopausal Status
        menopaus = st.selectbox("Menopausal Status", 
                                ["Pre- or peri-menopausal", "Post-menopausal", 
                                "Surgical menopause", "Unknown"])
        menopausal_status_encoded = convert_menopausal_status_to_int(menopaus)

        # BMI Group (A grouping for BMI categories)
        bmi = st.number_input("What is your current BMI (Body Mass Index)?", min_value=0.0, max_value=100.0, value=24.0, step=0.1)
        bmi_group_encoded = convert_bmi_to_group(bmi)

        # Biopsy History
        biophx = st.selectbox("Have you had a breast biopsy in the past?", ["No", "Yes"])
        biophx_encoded = convert_biophx_to_int(biophx)

        # Prepare the input data for the prediction
        input_data = {
            "year": year,
            "age_group_5_years": encode_age(age),
            "race_eth": race_encoded,
            "first_degree_hx": family_history_encoded,
            "age_menarche": age_menarche_encoded,
            "age_first_birth": first_birth_encoded,
            "BIRADS_breast_density": birads_breast_density_encoded,
            "current_hrt": current_hrt_encoded,
            "menopaus": menopausal_status_encoded,
            "bmi_group": bmi_group_encoded,
            "biophx": biophx_encoded,
            "count": 1  # Placeholder for count (you can process this as needed)
        }

        # Prediction button
        if st.button("Predict Risk of Breast Cancer"):
            # Make prediction using the second model (model2)
            input_df = pd.DataFrame([input_data]) 
            pred = model2.predict(input_df)

            # Output Result with Probability and Additional Information
            if pred == 1:
                st.write(f"According to the input data, there is a higher risk of contracting breast cancer.")
                st.warning("Please consult with a healthcare provider for further evaluation and screening.")
            else:
                st.write(f"The model indicates a lower chance of contracting breast cancer.")
                st.success("However, it's still essential to maintain regular check-ups and screenings.")
            
            # Display additional information (optional)
            st.info("The prediction is based on your provided data. A healthcare professional can provide a more accurate evaluation.")

            # Optional: Display model confidence if available
            prob = model2.predict_proba(input_df)  # Uncomment if model supports probability
            st.write(f"Confidence level: {prob[0][1]*100:.2f}%")