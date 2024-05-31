import streamlit as st
import requests

# Streamlit uygulaması başlıkları
st.title('Gemma 2B Text Generation')
st.subheader('Input the text and parameters to generate responses.')

# Form oluşturuluyor
with st.form("text_gen_form"):
    # Kullanıcıdan alınacak girdiler
    input_text = st.text_area("Enter text", value="", help="Input the text you want to generate from.")
    max_new_tokens = st.number_input("Max New Tokens", min_value=10, max_value=3096, value=50, step=10,
                                     help="Maximum new tokens to generate.")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                            help="Controls randomness in generation.")
    top_k = st.number_input("Top K", min_value=0, max_value=100, value=50,
                            help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.92, step=0.01,
                      help="Nucleus sampling: selects tokens with cumulative probability of `top_p` or higher.")
    submit_button = st.form_submit_button("Generate")

# Butona basıldığında yapılacak işlem
if submit_button:
    # API'ye gönderilecek data
    data = {
        "text": input_text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    }

    # Flask API endpoint
    url = "http://127.0.0.1:8081/gemma-generate"  # Flask servisinizin IP adresi ve portunu girin.

    # POST request ile data'yı gönder
    response = requests.post(url, json=data)

    # Yanıtı al ve ekranda göster
    if response.status_code == 200:
        result = response.json().get("response")
        st.write("Generated Text:", result)
    else:
        st.error("Failed to generate text. Server responded with an error.")