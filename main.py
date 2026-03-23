import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
from streamlit_cropper import st_cropper

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA (DEVE SER A LINHA 1 APÓS OS IMPORTS)
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA")

st.title("Sistema de Orçamento Automatizado - IA")

# ==========================================
# 2. UPLOAD E LEITURA (PDF OU IMAGEM)
# ==========================================
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG ou PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Se o cliente subir um PDF
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            
            # Pergunta a página se o PDF for gigante
            if len(doc) > 1:
                page_num = st.number_input(f"O projeto tem {len(doc)} pranchas. Qual página deseja analisar?", min_value=1, max_value=len(doc), value=1) - 1
            else:
                page_num = 0
                
            page = doc.load_page(page_num)
            
            # Tela de carregamento enquanto converte para 4K
            with st.spinner("Extraindo planta em Alta Resolução (300 DPI)..."):
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4)) 
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
        # Se o cliente subir um JPG/PNG normal
        else:
            image = Image.open(uploaded_file)
        
        st.success("Planta processada com sucesso!")
        
        # ==========================================
        # 3. O SEU CÓDIGO CONTINUA A PARTIR DAQUI
        # ==========================================
        # NÃO APAGUE o seu código daqui para baixo! 
        # É aqui que entra o seu st_cropper, a barra de sensibilidade, 
        # a busca do Airtable e o cálculo matemático.

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
