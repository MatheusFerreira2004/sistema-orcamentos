import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
from google import genai
from pyairtable import Api

# ==========================================
# 1. CONFIGURAÇÕES SEGURAS (SECRETS)
# ==========================================
# O sistema vai buscar as chaves escondidas na nuvem
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    AIRTABLE_TOKEN = st.secrets["AIRTABLE_TOKEN"]
    AIRTABLE_BASE_ID = st.secrets["AIRTABLE_BASE_ID"]
    AIRTABLE_TABLE_NAME = "Catalogo"
except Exception:
    st.error("Erro: As chaves de API não foram configuradas nos Secrets do Streamlit.")
    st.stop()

# ==========================================
# 2. INICIALIZAÇÃO DOS CLIENTES
# ==========================================
client = genai.Client(api_key=GEMINI_API_KEY)
api_airtable = Api(AIRTABLE_TOKEN)
tabela_precos = api_airtable.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

# ==========================================
# 3. INTERFACE DO SISTEMA
# ==========================================
st.set_page_config(page_title="Bot de Orçamentos", layout="wide")
st.title("💡 Sistema Híbrido: Visão + IA")
st.markdown("---")

uploaded_file = st.file_uploader("1. Carregue a prancha/planta", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    st.subheader("2. Recorte o símbolo que deseja contar:")
    cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00')
    
    threshold = st.slider("Sensibilidade da Busca", 0.5, 0.99, 0.8, 0.01)

    if st.button("Iniciar Contagem", type="primary"):
        crop_array = np.array(cropped_img)
        crop_cv2 = cv2.cvtColor(crop_array, cv2.COLOR_RGB2BGR)
        crop_gray = cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(img_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        h, w = crop_gray.shape
        count, pontos = 0, []
        img_result = img_cv2.copy()
        
        for pt in zip(*loc[::-1]):
            if not any(np.linalg.norm(np.array(pt) - np.array(p)) < min(w, h)/2 for p in pontos):
                pontos.append(pt)
                cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                count += 1

        st.success(f"Foram encontrados **{count}** itens na planta!")
        st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), use_column_width=False, output_format="PNG")
        st.session_state['quantidade_itens'] = count

if 'quantidade_itens' in st.session_state:
    st.markdown("---")
    item_busca = st.text_input("Nome do item para buscar no Airtable:")
    
    if st.button("Buscar e Calcular") and item_busca:
        with st.spinner("Buscando no catálogo..."):
            records = tabela_precos.all()
            item_encontrado = None
            for record in records:
                campos = record.get('fields', {})
                if item_busca.lower() in str(campos.get('Nome', '')).lower():
                    item_encontrado = campos.get('Nome')
                    preco_unitario = float(campos.get('Preco', 0))
                    break
            
            if item_encontrado:
                total = st.session_state['quantidade_itens'] * preco_unitario
                st.metric("Total Estimado", f"R$ {total:.2f}")
            else:
                st.warning("Item não encontrado.")          
