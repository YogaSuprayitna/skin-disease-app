import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model dan label kelas
model = load_model('model_dt_kulit.h5')
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.title("🩺 Deteksi Penyakit Kulit dengan CNN")
st.write("Upload gambar kulit atau gunakan kamera untuk prediksi penyakit kulit.")

# Opsi input: Upload atau Kamera
input_method = st.radio("Pilih metode input gambar:", ["📁 Upload Gambar", "📷 Kamera"])

img = None

if input_method == "📁 Upload Gambar":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

elif input_method == "📷 Kamera":
    camera_image = st.camera_input("Ambil foto kulit")
    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")

# Jika gambar tersedia, tampilkan dan prediksi
if img is not None:
    st.image(img, caption="Gambar yang dipilih", use_column_width=True)

    img_resized = img.resize((200, 200))  # Sesuaikan dengan input model
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(prediction[0])) * 100

    st.markdown(f"### 🔍 Prediksi: **{predicted_label.capitalize()}**")
    st.markdown(f"📊 Tingkat keyakinan: **{confidence:.2f}%**")
