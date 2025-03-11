import streamlit as st
import torch             
import pickle
import os
import gdown


# Path Model & Tokenizer
MODEL_PATH = "./dataset/indobert_finetuned.pkl"
TOKENIZER_PATH = "./dataset/indobert_tokenizer.pkl"

# Pastikan folder tujuan ada
os.makedirs("dataset", exist_ok=True)

# Cek apakah file sudah ada sebelum mengunduh
if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(MODEL_PATH):
    st.warning("Model atau tokenizer tidak ditemukan. Mengunduh file...")
    
    # Download file pertama jika belum ada
    if not os.path.exists(TOKENIZER_PATH):
        try:
            url = "https://drive.google.com/uc?id=1JH6NXQygYGDA9e_U4HhXYVAuIFZVwAfd"
            gdown.download(url, TOKENIZER_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error downloading tokenizer: {e}")
    
    # Download file kedua jika belum ada
    if not os.path.exists(MODEL_PATH):
        try:
            url1 = "https://drive.google.com/uc?id=1sVxpucTDccVjYOnoy7h5-e9eSS4x_pIe"
            gdown.download(url1, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
else:
    st.success("Model dan tokenizer sudah tersedia.")

# Load Tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load Model
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Load model dan tokenizer saat startup
tokenizer = load_tokenizer(TOKENIZER_PATH)
model = load_model(MODEL_PATH)

# Mapping label
label_dict = {
    0: {"label": "Pesan Biasa", "color": "#4CAF50"},
    1: {"label": "Pesan Spam", "color": "#FF5722"},
    2: {"label": "Pesan Promosi", "color": "#FFC107"},
}

# Streamlit UI
st.title("Klasifikasi Pesan dengan IndoBERT")
st.write("Masukkan teks untuk diklasifikasikan:")

# Input teks dari pengguna
user_input = st.text_area("Tulis pesan di sini", "")

if st.button("Prediksi"):
    if user_input:
        # Tokenisasi input
        tokens = tokenizer.encode_plus(user_input, padding=True, truncation=True, return_tensors="pt")
        
        # Prediksi Model
        with torch.no_grad():
            output = model(**tokens)
            prediction = torch.argmax(output.logits, dim=1).item()
        
        hasil = label_dict.get(prediction, {"label": "Kategori Tidak Diketahui", "color": "#9E9E9E"})
        
        # Tampilkan hasil prediksi
        st.markdown(f"<h3 style='color: {hasil['color']};'>{hasil['label']}</h3>", unsafe_allow_html=True)
    else:
        st.warning("Teks tidak boleh kosong!")
