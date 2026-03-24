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
st.set_page_config(layout="wide", page_title="Orçamento IA - Alta Precisão")

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

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# ==========================================
# 2. UPLOAD E CONVERSÃO
# ==========================================
uploaded_file = st.file_uploader("Suba a sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Lendo PDF de alta resolução..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) # Aumentamos a escala para 3x para mais precisão
                img_pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                doc.close()
        else:
            img_pil = Image.open(uploaded_file).convert("RGB")

        # Preparação das imagens
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        st.info("💡 Como usar: 1. Clique no símbolo na planta. 2. Ajuste a precisão. 3. Clique em Contar.")

        # Exibição para o clique
        largura_display = 1200
        fator = largura_display / img_pil.size[0]
        altura_display = int(img_pil.size[1] * fator)
        img_res = img_pil.resize((largura_display, altura_display), Image.Resampling.LANCZOS)

        from streamlit_image_coordinates import streamlit_image_coordinates
        coords = streamlit_image_coordinates(img_res, key="clique")

        if coords:
            x_real = int(coords["x"] / fator)
            y_real = int(coords["y"] / fator)

            # Recorte do Template (Alvo)
            box = 30
            y1, y2 = max(0, y_real-box), min(img_cv.shape[0], y_real+box)
            x1, x2 = max(0, x_real-box), min(img_cv.shape[1], x_real+box)
            template_gray = img_gray[y1:y2, x1:x2]

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(cv2.cvtColor(img_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), caption="Símbolo Alvo", width=100)
            
            with col2:
                threshold = st.slider("Sensibilidade do Robô (0.80 é o ideal)", 0.50, 0.98, 0.80, 0.01)
                
                if st.button("🔍 Iniciar Contagem em Toda a Planta", type="primary"):
                    with st.spinner("O robô está analisando cada pixel da planta..."):
                        pontos = []
                        # Motor de Rotação 360º para detectar símbolos em qualquer posição
                        for angulo in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                            t = template_gray if angulo is None else cv2.rotate(template_gray, angulo)
                            res = cv2.matchTemplate(img_gray, t, cv2.TM_CCOEFF_NORMED)
                            loc = np.where(res >= threshold)
                            
                            raio_exclusao = max(t.shape) / 2
                            for pt in zip(*loc[::-1]):
                                if not any(abs(pt[0]-p[0]) < raio_exclusao and abs(pt[1]-p[1]) < raio_exclusao for p in pontos):
                                    pontos.append(pt)
                                    cv2.rectangle(img_cv, pt, (pt[0] + t.shape[1], pt[1] + t.shape[0]), (0, 0, 255), 3)

                        st.success(f"✅ Sucesso! Encontrados {len(pontos)} itens.")
                        
                        # Resultado Visual
                        res_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        st.image(res_final, caption="Mapa de Detecção", use_container_width=True)

                        # Integração Airtable
                        st.markdown("---")
                        nome_busca = st.text_input("Qual o nome deste item no seu Catálogo do Airtable?")
                        
                        if st.button("💰 Consultar Preço e Adicionar ao Orçamento") and nome_busca:
                            headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
                            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"
                            
                            try:
                                r = requests.get(url, headers=headers).json()
                                preco = 0.0
                                nome_real = nome_busca
                                
                                for rec in r.get('records', []):
                                    if nome_busca.lower() in rec['fields'].get('Nome', '').lower():
                                        preco = float(rec['fields'].get('Preco', 0))
                                        nome_real = rec['fields'].get('Nome', nome_busca)
                                        break
                                
                                st.session_state['carrinho'].append({
                                    "Produto": nome_real,
                                    "Quantidade": len(pontos),
                                    "Preço Unitário": preco,
                                    "Total": preco * len(pontos)
                                })
                                st.success(f"Item {nome_real} adicionado!")
                            except:
                                st.error("Erro ao conectar com o Airtable.")

        # Tabela de Orçamento Final
        if st.session_state['carrinho']:
            st.markdown("---")
            st.header("📋 Orçamento Consolidado")
            df = pd.DataFrame(st.session_state['carrinho'])
            st.dataframe(df, use_container_width=True)
            
            total_geral = df["Total"].sum()
            st.subheader(f"Custo Total: R$ {total_geral:,.2f}")
            
            st.download_button("📥 Baixar Excel", df.to_csv(index=False).encode('utf-8'), "orcamento_ia.csv", "text/csv")

    except Exception as e:
        st.error(f"Erro de processamento: {e}")
