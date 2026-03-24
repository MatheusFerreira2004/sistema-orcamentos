import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
import requests
import urllib.parse
from streamlit_image_coordinates import streamlit_image_coordinates

# CONFIG
st.set_page_config(layout="wide", page_title="Orçamento IA - Goiânia")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

st.title("Sistema de Orçamento Automatizado - IA 🚀")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
    st.error("⚠️ Erro: Chaves do Airtable não encontradas.")
    st.stop()

# MEMÓRIA
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None

# UPLOAD
uploaded_file = st.file_uploader("Suba sua Planta", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # PDF alta resolução
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(6, 6))
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")

        st.subheader("1. Clique no símbolo que deseja contar")

        # CLIQUE NA IMAGEM
        coords = streamlit_image_coordinates(image)

        if coords:
            x, y = coords["x"], coords["y"]
            st.write(f"📍 Coordenada selecionada: {x}, {y}")

            # Converter imagem
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Capturar cor
            b, g, r = img_cv[y, x]
            pixel = np.uint8([[[b, g, r]]])
            hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)

            h, s, v = hsv_pixel[0][0]

            st.write(f"🎨 Cor HSV detectada: H={h}, S={s}, V={v}")

            # SLIDERS DE AJUSTE
            st.subheader("Ajuste fino da detecção")
            h_range = st.slider("Range de Hue", 5, 30, 10)
            s_min = st.slider("Saturação mínima", 0, 255, 50)
            v_min = st.slider("Valor mínimo", 0, 255, 50)
            area_min = st.slider("Área mínima", 10, 500, 80)

            lower = np.array([max(0, h - h_range), s_min, v_min])
            upper = np.array([min(179, h + h_range), 255, 255])

            if st.button("🔍 Detectar símbolos"):
                with st.spinner("Analisando..."):

                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)

                    # Limpeza
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    count = 0
                    img_result = img_cv.copy()

                    for cnt in contours:
                        area = cv2.contourArea(cnt)

                        if area > area_min:
                            count += 1
                            x,y,w,h = cv2.boundingRect(cnt)
                            cv2.rectangle(img_result, (x,y), (x+w,y+h), (0,255,0), 3)

                    st.session_state['total_itens'] = count
                    st.session_state['mapa_resultado'] = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # RESULTADO
        if st.session_state['mapa_resultado'] is not None:
            st.success(f"✅ Encontrados {st.session_state['total_itens']} itens")
            st.image(st.session_state['mapa_resultado'])

            st.markdown("---")
            st.subheader("2. Gerar Orçamento")

            product_search = st.text_input("Nome do produto no Airtable")
            if st.button("💰 Calcular Orçamento") and product_search:

                table_encoded = urllib.parse.quote(AIRTABLE_TABLE_NAME)
                url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_encoded}"
                headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    records = response.json().get("records", [])
                    found = None

                    for record in records:
                        nome = record.get("fields", {}).get("Nome", "")
                        if product_search.lower() in nome.lower():
                            found = record
                            break

                    if found:
                        preco = found["fields"].get("Preco", 0)
                        try:
                            preco = float(preco)
                        except:
                            preco = 0

                        total = preco * st.session_state['total_itens']

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Preço Unitário", f"R$ {preco:,.2f}")
                        col2.metric("Quantidade", st.session_state['total_itens'])
                        col3.metric("Total", f"R$ {total:,.2f}")
                    else:
                        st.warning("Produto não encontrado.")
                else:
                    st.error("Erro ao conectar com Airtable.")

    except Exception as e:
        st.error(f"Erro: {e}")
