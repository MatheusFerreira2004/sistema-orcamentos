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
import base64
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURAÇÃO E FUNÇÕES DE SUPORTE
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA v2.1")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

def get_image_base64(img):
    """Converte imagem PIL para Base64 para evitar erro de image_to_url"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None
if 'carrinho' not in st.session_state:
    st.session_state['carrinho'] = []
if 'produto_atual' not in st.session_state:
    st.session_state['produto_atual'] = None

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# ==========================================
# 2. UPLOAD E PROCESSAMENTO
# ==========================================
uploaded_file = st.file_uploader("Suba a sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("A extrair PDF..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_high_res = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                doc.close()
                del doc, page, pix
                gc.collect()
        else:
            image_high_res = Image.open(uploaded_file).convert("RGB")

        # Redimensionamento para exibição
        largura_tela = 1000 
        fator_escala = largura_tela / float(image_high_res.size[0])
        altura_tela = int(float(image_high_res.size[1]) * float(fator_escala))
        image_display = image_high_res.resize((largura_tela, altura_tela), Image.Resampling.LANCZOS)

        # Bypass para o erro de image_to_url
        img_b64 = get_image_base64(image_display)
        bg_data = f"data:image/png;base64,{img_b64}"

        st.markdown("---")
        st.subheader("1. Isole a área de projeto (Desenhe um retângulo)")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_color=bg_data,
            update_streamlit=True,
            height=altura_tela,
            width=largura_tela,
            drawing_mode="rect",
            key="canvas_main",
        )

        roi_coords = None
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                rect = objects[-1]
                roi_coords = (
                    int(rect["left"] / fator_escala),
                    int(rect["top"] / fator_escala),
                    int(rect["width"] / fator_escala),
                    int(rect["height"] / fator_escala)
                )
                st.success("✅ Área isolada!")

        # ==========================================
        # 3. CAPTURA E CONTAGEM
        # ==========================================
        st.markdown("---")
        st.subheader("2. Clique no símbolo que deseja contar")
        
        coords = streamlit_image_coordinates(image_display, key="clique_contagem")

        if coords and roi_coords:
            x_real = int(coords["x"] / fator_escala)
            y_real = int(coords["y"] / fator_escala)

            img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            box_size = st.slider("Tamanho da captura", 5, 80, 25)
            y1, y2 = max(0, y_real - box_size), min(img_cv.shape[0], y_real + box_size)
            x1, x2 = max(0, x_real - box_size), min(img_cv.shape[1], x_real + box_size)
            
            template_gray = img_gray[y1:y2, x1:x2]
            st.image(cv2.cvtColor(img_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), caption="Alvo", width=100)

            threshold = st.slider("Precisão", 0.50, 0.99, 0.80)

            if st.button("🔍 Iniciar Contagem na ROI"):
                with st.spinner("A processar..."):
                    x_r, y_r, w_r, h_r = roi_coords
                    roi_gray = img_gray[y_r:y_r+h_r, x_r:x_r+w_r]
                    
                    pontos = []
                    rotations = [template_gray, 
                                 cv2.rotate(template_gray, cv2.ROTATE_90_CLOCKWISE),
                                 cv2.rotate(template_gray, cv2.ROTATE_180),
                                 cv2.rotate(template_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)]
                    
                    raio = max(template_gray.shape) / 2
                    for temp in rotations:
                        res = cv2.matchTemplate(roi_gray, temp, cv2.TM_CCOEFF_NORMED)
                        loc = np.where(res >= threshold)
                        for pt in zip(*loc[::-1]):
                            pt_g = (pt[0] + x_r, pt[1] + y_r)
                            if not any(abs(pt_g[0]-p[0]) < raio and abs(pt_g[1]-p[1]) < raio for p in pontos):
                                pontos.append(pt_g)
                                cv2.rectangle(img_cv, pt_g, (pt_g[0]+temp.shape[1], pt_g[1]+temp.shape[0]), (0,0,255), 4)

                    st.session_state['total_itens'] = len(pontos)
                    res_small = cv2.resize(img_cv, (1200, int(img_cv.shape[0]*(1200/img_cv.shape[1]))))
                    st.session_state['mapa_resultado'] = cv2.cvtColor(res_small, cv2.COLOR_BGR2RGB)
                    st.rerun()

        # ==========================================
        # 4. AIRTABLE E CARRINHO
        # ==========================================
        if st.session_state['mapa_resultado'] is not None:
            st.image(st.session_state['mapa_resultado'], caption="Resultado da Contagem")
            
            product_search = st.text_input("Nome do item no Airtable")
            if st.button("🛒 Adicionar ao Orçamento") and product_search:
                # Lógica simplificada de busca
                headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
                url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"
                resp = requests.get(url, headers=headers).json()
                
                price = 0.0
                for r in resp.get('records', []):
                    if product_search.lower() in r['fields'].get('Nome', '').lower():
                        price = float(r['fields'].get('Preco', 0))
                        break
                
                st.session_state['carrinho'].append({
                    "Produto": product_search,
                    "Quantidade": st.session_state['total_itens'],
                    "Unitário": price,
                    "Subtotal": price * st.session_state['total_itens']
                })
                st.success("Adicionado!")

        # Exibição do Carrinho Final
        if st.session_state['carrinho']:
            st.markdown("### 📋 Orçamento Atual")
            df = pd.DataFrame(st.session_state['carrinho'])
            st.table(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Excel (CSV)", csv, "orcamento.csv", "text/csv")

    except Exception as e:
        st.error(f"Erro Crítico: {e}")
