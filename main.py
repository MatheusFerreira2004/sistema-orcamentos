import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
from streamlit_cropper import st_cropper
import requests
import urllib.parse

# 1. CONFIGURAÇÃO INICIAL
st.set_page_config(layout="wide", page_title="Sistema de Orçamento IA")

# Carregamento seguro dos Secrets
try:
    # .strip() remove espaços invisíveis que causam o erro 403
    api_key = st.secrets["AIRTABLE_API_KEY"].strip()
    base_id = st.secrets["AIRTABLE_BASE_ID"].strip()
    table_name = st.secrets["AIRTABLE_TABLE_NAME"].strip()
    secrets_ok = True
except:
    st.error("Erro nos Secrets. Verifique as chaves no painel do Streamlit.")
    secrets_ok = False

st.title("Contador & Orçamentista IA 🚀")

# 2. UPLOAD E CONVERSÃO DE PDF/IMG
uploaded_file = st.file_uploader("Suba o arquivo do projeto", type=["pdf", "jpg", "png"])

if uploaded_file and secrets_ok:
    # Processamento de Imagem (PDF para 4K RGB)
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0) # Carrega a primeira página por padrão
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    # 3. INTERFACE DE RECORTE
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Selecione o Símbolo na Planta")
        cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
    
    with col2:
        st.subheader("Alvo Selecionado")
        if cropped_img:
            st.image(cropped_img)
            sensibilidade = st.slider("Precisão do Robô", 0.50, 0.99, 0.90, 0.01)
            btn_contar = st.button("CONTAR AGORA")

    # 4. LÓGICA DE CONTAGEM E BUSCA NO AIRTABLE
    if 'btn_contar' in locals() and btn_contar:
        # Contagem OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        h, w = temp_cv.shape[:2]
        
        res = cv2.matchTemplate(img_cv, temp_cv, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= sensibilidade)
        pontos = list(zip(*loc[::-1]))
        
        # Desenhar resultados
        total = len(pontos)
        st.success(f"Encontrados: {total} itens.")
        
        # --- BLOCO DE CONEXÃO AIRTABLE (CORREÇÃO 403) ---
        st.subheader("Consultando Preços no Airtable...")
        
        # Codifica o nome da tabela para a URL (Ex: transforma espaços em %20)
        table_encoded = urllib.parse.quote(table_name)
        url = f"https://api.airtable.com/v0/{base_id}/{table_encoded}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                st.write("✅ Conexão com Airtable estabelecida!")
                # Aqui você pode adicionar a lógica de busca por nome que já tínhamos
            else:
                st.error(f"Erro {response.status_code}: Verifique se o Token tem acesso à base '{base_id}'")
                st.info("Dica: No Airtable, o Token precisa ter o Scope 'data.records:read' e a Base adicionada em 'Access'.")
                
        except Exception as e:
            st.error(f"Falha na requisição: {e}")
