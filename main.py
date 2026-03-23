import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
from streamlit_cropper import st_cropper
import requests
import urllib.parse

# Configuração da página (Sempre a primeira linha de código Streamlit)
st.set_page_config(layout="wide", page_title="Orçamento IA - Goiânia")

# --- FUNÇÃO DE LIMPEZA DE SECRETS ---
def get_clean_secret(key_name):
    try:
        # Remove espaços, aspas e quebras de linha invisíveis
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

# Carregamento seguro
AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# --- TESTE DE CONEXÃO IMEDIATO ---
if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
    st.error("⚠️ Erro: Chaves do Airtable não encontradas nos Secrets do Streamlit.")
    st.stop()

# --- INTERFACE DE UPLOAD ---
uploaded_file = st.file_uploader("Suba sua Planta (PDF ou Imagem)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Processamento de PDF (Fix fundo branco)
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")

        # Layout em Colunas
        col_crop, col_res = st.columns([2, 1])

        with col_crop:
            st.subheader("1. Selecione o Símbolo")
            # st_cropper para capturar o alvo
            cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00', aspect_ratio=None)

        with col_res:
            st.subheader("2. Configurações")
            # Aumentei o padrão para 0.92 para resolver os "10 extras"
            sensibilidade = st.slider("Precisão do Robô (0.95 = Máxima)", 0.80, 0.99, 0.92, 0.01)
            btn_contar = st.button("CONTAR E GERAR ORÇAMENTO")

        if btn_contar:
            # --- LÓGICA DE CONTAGEM (OpenCV) ---
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            temp_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            h, w = temp_cv.shape[:2]
            
            res = cv2.matchTemplate(img_cv, temp_cv, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= sensibilidade)
            
            # Filtro para não contar o mesmo item várias vezes (NMS simples)
            pontos = []
            for pt in zip(*loc[::-1]):
                if not any(abs(pt[0]-p[0]) < w/2 and abs(pt[1]-p[1]) < h/2 for p in pontos):
                    pontos.append(pt)
            
            total_itens = len(pontos)
            st.success(f"Encontrados {total_itens} itens na planta.")

            # --- BUSCA NO AIRTABLE (CORREÇÃO DO 403) ---
            with st.spinner("Consultando preços no Airtable..."):
                table_encoded = urllib.parse.quote(AIRTABLE_TABLE_NAME)
                url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_encoded}"
                
                headers = {
                    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
                    "Content-Type": "application/json"
                }

                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])
                    st.info(f"Banco de dados '{AIRTABLE_TABLE_NAME}' acessado com sucesso!")
                    
                    # Aqui você pode adicionar a lógica de filtro por nome do produto
                    # conforme a necessidade do seu catálogo.
                else:
                    st.error(f"Erro {response.status_code} no Airtable.")
                    st.write(f"Detalhe do erro: {response.text}")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
