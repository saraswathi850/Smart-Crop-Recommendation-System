import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="CropSense AI",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("🌾 CropSense AI")
st.sidebar.write(
    "A smart crop recommendation system using Machine Learning "
    "to help farmers choose the best crop based on soil and climate."
)

st.sidebar.divider()
st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Inputs:** Soil + Climate")
st.sidebar.write("**Output:** Best Crop")

# ---------------------------------
# Main Title
# ---------------------------------
st.title("🌱 Smart Crop Recommendation")
st.caption(
    "Make data-driven farming decisions for better yield and sustainability"
)

st.divider()

# ---------------------------------
# Load Dataset & Train Model
# ---------------------------------
df = pd.read_csv("Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X, y)

# ---------------------------------
# User Inputs
# ---------------------------------
st.subheader("🌍 Enter Soil & Climate Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (kg/ha)", 0, 140, 60)
    P = st.slider("Phosphorus (kg/ha)", 0, 140, 55)
    K = st.slider("Potassium (kg/ha)", 0, 140, 60)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

with col2:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Analyze & Recommend"):
    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=X.columns
    )

    prediction = model.predict(input_data)[0]

    st.divider()

    # 🔥 BIG, BOLD, CLEAR OUTPUT
    st.success("🌾 Recommended Crop")
    st.header(prediction.upper())

    st.write(
        f"Based on the given soil nutrients and climate conditions, "
        f"**{prediction}** is the most suitable crop for optimal yield."
    )

    st.info(
        f"📊 Input Summary\n\n"
        f"- Nitrogen: {N} kg/ha\n"
        f"- Temperature: {temperature} °C\n"
        f"- Rainfall: {rainfall} mm"
    )

# ---------------------------------
# Footer
# ---------------------------------
st.divider()
st.caption(
    "🌍 Growing smarter farms with data, not guesswork 🌱\n\n"
    "**CropSense AI — Intelligence rooted in agriculture**"
)