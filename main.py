import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
import requests
import urllib.parse
import gc
import pandas as pd

# ==========================================
# 1. CONFIGURAÇÃO E SECRETS
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA - Precisão Máxima")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

if 'carrinho' not in st.session_state:
    st.session_state['carrinho'] = []
if 'resultado_contagem' not in st.session_state:
    st.session_state['resultado_contagem'] = None
if 'qtd_encontrada' not in st.session_state:
    st.session_state['qtd_encontrada'] = 0

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# ==========================================
# 2. UPLOAD E PROCESSAMENTO
# ==========================================
uploaded_file = st.file_uploader("Suba a sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)
            # Matrix 2.0 para equilíbrio entre velocidade e nitidez
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            doc.close()
        else:
            img_pil = Image.open(uploaded_file).convert("RGB")

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Interface de Seleção
        largura_display = 1100
        fator = largura_display / img_pil.size[0]
        altura_display = int(img_pil.size[1] * fator)
        img_res = img_pil.resize((largura_display, altura_display), Image.Resampling.LANCZOS)

        st.subheader("1. Clique no símbolo que deseja contar")
        from streamlit_image_coordinates import streamlit_image_coordinates
        coords = streamlit_image_coordinates(img_res, key="clique_v1")

        if coords:
            x_real = int(coords["x"] / fator)
            y_real = int(coords["y"] / fator)

            st.markdown("---")
            col_cfg, col_alvo = st.columns([2, 1])

            with col_cfg:
                # O retorno dos controles que você prefere:
                box_size = st.slider("Tamanho da captura (Ajuste o quadrado no símbolo)", 5, 80, 25)
                precision = st.slider("Precisão", 0.50, 0.99, 0.80)
                
                btn_contar = st.button("🔍 Iniciar Contagem", type="primary")

            # Recorte do Template
            y1, y2 = max(0, y_real - box_size), min(img_cv.shape[0], y_real + box_size)
            x1, x2 = max(0, x_real - box_size), min(img_cv.shape[1], x_real + box_size)
            template_gray = img_gray[y1:y2, x1:x2]

            with col_alvo:
                st.image(cv2.cvtColor(img_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), caption="Alvo Capturado", width=120)

            if btn_contar:
                with st.spinner("Contando símbolos na planta inteira..."):
                    pontos = []
                    # Rotações para garantir que conte símbolos em qualquer posição
                    for angulo in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                        t = template_gray if angulo is None else cv2.rotate(template_gray, angulo)
                        res = cv2.matchTemplate(img_gray, t, cv2.TM_CCOEFF_NORMED)
                        loc = np.where(res >= precision)
                        
                        raio = max(t.shape) / 2
                        for pt in zip(*loc[::-1]):
                            if not any(abs(pt[0]-p[0]) < raio and abs(pt[1]-p[1]) < raio for p in pontos):
                                pontos.append(pt)
                                cv2.rectangle(img_cv, pt, (pt[0]+t.shape[1], pt[1]+t.shape[0]), (0,0,255), 3)

                    st.session_state['qtd_encontrada'] = len(pontos)
                    # Redimensionar resultado para não pesar no navegador
                    res_viz = cv2.resize(img_cv, (1000, int(img_cv.shape[0]*(1000/img_cv.shape[1]))))
                    st.session_state['resultado_contagem'] = cv2.cvtColor(res_viz, cv2.COLOR_BGR2RGB)

        # ==========================================
        # 3. RESULTADOS E AIRTABLE
        # ==========================================
        if st.session_state['resultado_contagem'] is not None:
            st.image(st.session_state['resultado_contagem'], caption="Resultado da Detecção")
            st.success(f"Encontrados: {st.session_state['qtd_encontrada']} itens")

            col_air1, col_air2 = st.columns(2)
            with col_air1:
                nome_item = st.text_input("Nome exato ou parte do nome no Airtable")
            with col_air2:
                if st.button("🛒 Adicionar ao Orçamento") and nome_item:
                    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
                    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"
                    
                    try:
                        resp = requests.get(url, headers=headers).json()
                        preco = 0.0
                        for r in resp.get('records', []):
                            if nome_item.lower() in r['fields'].get('Nome', '').lower():
                                preco = float(r['fields'].get('Preco', 0))
                                nome_item = r['fields'].get('Nome', nome_item)
                                break
                        
                        st.session_state['carrinho'].append({
                            "Item": nome_item,
                            "Quantidade": st.session_state['qtd_encontrada'],
                            "Preço Unit.": preco,
                            "Subtotal": preco * st.session_state['qtd_encontrada']
                        })
                        st.success(f"{nome_item} adicionado!")
                    except:
                        st.error("Erro ao acessar o Airtable. Verifique os Secrets.")

        # Tabela Final
        if st.session_state['carrinho']:
            st.markdown("---")
            st.subheader("📋 Resumo do Orçamento")
            df = pd.DataFrame(st.session_state['carrinho'])
            st.table(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV", csv, "orcamento.csv", "text/csv")

    except Exception as e:
        st.error(f"Erro: {e}")
