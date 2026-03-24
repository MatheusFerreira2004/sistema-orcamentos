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
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURAÇÃO
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA v2.4")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

# Sessão
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None
if 'carrinho' not in st.session_state:
    st.session_state['carrinho'] = []

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# ==========================================
# 2. UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Suba a sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

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

    largura_canvas = 1000
    fator_escala = largura_canvas / float(image_high_res.size[0])
    altura_canvas = int(float(image_high_res.size[1]) * fator_escala)

    image_display = image_high_res.resize(
        (largura_canvas, altura_canvas),
        Image.Resampling.LANCZOS
    )

    # 🔥 CORREÇÃO DEFINITIVA
    image_bytes = pil_to_bytes(image_display)
    image_safe = Image.open(io.BytesIO(image_bytes))

    st.markdown("---")

    # ==========================================
    # 3. ROI
    # ==========================================
    st.subheader("1. Selecione a área de projeto (Ignore legendas)")
    st.info("💡 Desenhe um retângulo sobre a planta")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.1)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=image_safe,  # 🔥 AQUI ESTÁ O FIX
        update_streamlit=True,
        height=altura_canvas,
        width=largura_canvas,
        drawing_mode="rect",
        key="canvas_ok",
    )

    roi_coords = None

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][-1]

        roi_coords = (
            int(rect["left"] / fator_escala),
            int(rect["top"] / fator_escala),
            int(rect["width"] / fator_escala),
            int(rect["height"] / fator_escala)
        )

        st.success("Área selecionada com sucesso!")

    # ==========================================
    # 4. CLIQUE E DETECÇÃO
    # ==========================================
    if roi_coords:
        st.markdown("---")
        st.subheader("2. Clique no símbolo que deseja contar")

        coords = streamlit_image_coordinates(image_display, key="click_coord")

        if coords:
            x_real = int(coords["x"] / fator_escala)
            y_real = int(coords["y"] / fator_escala)

            img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            box = 25

            y1 = max(0, y_real - box)
            y2 = min(img_cv.shape[0], y_real + box)
            x1 = max(0, x_real - box)
            x2 = min(img_cv.shape[1], x_real + box)

            template_gray = img_gray[y1:y2, x1:x2]

            st.image(
                cv2.cvtColor(img_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
                caption="Alvo Selecionado",
                width=80
            )

            threshold = st.slider("Sensibilidade", 0.50, 0.95, 0.80)

            if st.button("🔍 Iniciar Contagem"):

                xr, yr, wr, hr = roi_coords
                roi_gray = img_gray[yr:yr+hr, xr:xr+wr]

                pontos = []

                for angulo in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:

                    t = template_gray if angulo is None else cv2.rotate(template_gray, angulo)

                    res = cv2.matchTemplate(roi_gray, t, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)

                    raio = max(t.shape) / 2

                    for pt in zip(*loc[::-1]):
                        pt_g = (pt[0] + xr, pt[1] + yr)

                        if not any(abs(pt_g[0]-p[0]) < raio and abs(pt_g[1]-p[1]) < raio for p in pontos):
                            pontos.append(pt_g)

                            cv2.rectangle(
                                img_cv,
                                pt_g,
                                (pt_g[0]+t.shape[1], pt_g[1]+t.shape[0]),
                                (0, 0, 255),
                                4
                            )

                st.session_state['total_itens'] = len(pontos)

                res_viz = cv2.resize(
                    img_cv,
                    (1000, int(img_cv.shape[0]*(1000/img_cv.shape[1])))
                )

                st.session_state['mapa_resultado'] = cv2.cvtColor(res_viz, cv2.COLOR_BGR2RGB)

                st.rerun()

    # ==========================================
    # 5. RESULTADO + AIRTABLE
    # ==========================================
    if st.session_state['mapa_resultado'] is not None:

        st.image(st.session_state['mapa_resultado'], caption="Resultado Final")
        st.success(f"Quantidade: {st.session_state['total_itens']}")

        prod_nome = st.text_input("Buscar Nome no Airtable")

        if st.button("Confirmar Item") and prod_nome:

            headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"

            dados = requests.get(url, headers=headers).json()

            preco = next(
                (
                    float(r['fields'].get('Preco', 0))
                    for r in dados.get('records', [])
                    if prod_nome.lower() in r['fields'].get('Nome', '').lower()
                ),
                0.0
            )

            st.session_state['carrinho'].append({
                "Item": prod_nome,
                "Qtd": st.session_state['total_itens'],
                "Unitário": preco,
                "Total": preco * st.session_state['total_itens']
            })

            st.success("Item adicionado!")

    # ==========================================
    # 6. CARRINHO
    # ==========================================
    if st.session_state['carrinho']:

        st.markdown("---")

        df = pd.DataFrame(st.session_state['carrinho'])
        st.table(df)

        st.download_button(
            "📥 Salvar Orçamento",
            df.to_csv(index=False).encode('utf-8'),
            "orcamento.csv",
            "text/csv"
        )
