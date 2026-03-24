import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
import requests
import urllib.parse
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA v2.5")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

# SESSION
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None
if 'carrinho' not in st.session_state:
    st.session_state['carrinho'] = []

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# ==========================================
# UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Suba a sua Planta", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:

    if 'img_high' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            st.session_state['img_high'] = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            doc.close()
        else:
            st.session_state['img_high'] = Image.open(uploaded_file).convert("RGB")

        st.session_state['last_file'] = uploaded_file.name

    image_high_res = st.session_state['img_high']

    largura = 1000
    fator = largura / image_high_res.size[0]
    altura = int(image_high_res.size[1] * fator)

    image_display = image_high_res.resize((largura, altura), Image.Resampling.LANCZOS)

    st.markdown("---")

    # ==========================================
    # ROI VISUAL + CANVAS
    # ==========================================
    st.subheader("1. Selecione a área da planta")
    st.info("💡 Primeiro olhe a imagem. Depois desenhe o retângulo EXATAMENTE na mesma posição abaixo.")

    # imagem visível
    st.image(image_display, use_container_width=True)

    st.markdown("### ✏️ Agora desenhe aqui (mesma posição da imagem acima)")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.15)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_color="rgba(0,0,0,0)",
        update_streamlit=True,
        height=altura,
        width=largura,
        drawing_mode="rect",
        key="canvas_ok",
    )

    roi_coords = None

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][-1]

        roi_coords = (
            int(rect["left"] / fator),
            int(rect["top"] / fator),
            int(rect["width"] / fator),
            int(rect["height"] / fator)
        )

        st.success("Área selecionada!")

    # ==========================================
    # CLIQUE + DETECÇÃO
    # ==========================================
    if roi_coords:
        st.markdown("---")
        st.subheader("2. Clique no símbolo")

        coords = streamlit_image_coordinates(image_display, key="click")

        if coords:
            x_real = int(coords["x"] / fator)
            y_real = int(coords["y"] / fator)

            img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            box = 25
            template = gray[
                max(0, y_real-box):y_real+box,
                max(0, x_real-box):x_real+box
            ]

            st.image(
                cv2.cvtColor(img_cv[y_real-box:y_real+box, x_real-box:x_real+box], cv2.COLOR_BGR2RGB),
                width=80,
                caption="Símbolo"
            )

            threshold = st.slider("Sensibilidade", 0.5, 0.95, 0.8)

            if st.button("🔍 Contar símbolos"):
                xr, yr, wr, hr = roi_coords
                roi = gray[yr:yr+hr, xr:xr+wr]

                pontos = []

                for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    t = template if rot is None else cv2.rotate(template, rot)
                    res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)

                    raio = max(t.shape) / 2

                    for pt in zip(*loc[::-1]):
                        ptg = (pt[0]+xr, pt[1]+yr)

                        if not any(abs(ptg[0]-p[0]) < raio and abs(ptg[1]-p[1]) < raio for p in pontos):
                            pontos.append(ptg)
                            cv2.rectangle(img_cv, ptg, (ptg[0]+t.shape[1], ptg[1]+t.shape[0]), (0,0,255), 3)

                st.session_state['total_itens'] = len(pontos)

                img_small = cv2.resize(img_cv, (1000, int(img_cv.shape[0]*(1000/img_cv.shape[1]))))
                st.session_state['mapa_resultado'] = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

                st.rerun()

    # ==========================================
    # RESULTADO + ORÇAMENTO
    # ==========================================
    if st.session_state['mapa_resultado'] is not None:
        st.image(st.session_state['mapa_resultado'])
        st.success(f"Quantidade: {st.session_state['total_itens']}")

        nome = st.text_input("Produto")

        if st.button("Adicionar") and nome:
            headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"

            data = requests.get(url, headers=headers).json()

            preco = next(
                (float(r['fields'].get('Preco', 0)) for r in data['records']
                 if nome.lower() in r['fields'].get('Nome', '').lower()),
                0
            )

            st.session_state['carrinho'].append({
                "Item": nome,
                "Qtd": st.session_state['total_itens'],
                "Unitário": preco,
                "Total": preco * st.session_state['total_itens']
            })

    # ==========================================
    # CARRINHO
    # ==========================================
    if st.session_state['carrinho']:
        st.markdown("---")
        df = pd.DataFrame(st.session_state['carrinho'])
        st.table(df)

        st.download_button(
            "📥 Baixar CSV",
            df.to_csv(index=False).encode(),
            "orcamento.csv",
            "text/csv"
        )
