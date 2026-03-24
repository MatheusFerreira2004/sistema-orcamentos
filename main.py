import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
from streamlit_image_coordinates import streamlit_image_coordinates

# ================================
# CONFIG
# ================================
st.set_page_config(layout="wide")
st.title("Sistema Inteligente de Leitura de Plantas 🚀")

# ================================
# MEMÓRIA
# ================================
if "boxes" not in st.session_state:
    st.session_state["boxes"] = []

if "pre_map_img" not in st.session_state:
    st.session_state["pre_map_img"] = None

if "result_img" not in st.session_state:
    st.session_state["result_img"] = None

if "click" not in st.session_state:
    st.session_state["click"] = None

# ================================
# UPLOAD
# ================================
uploaded_file = st.file_uploader("Suba sua planta", type=["pdf","png","jpg","jpeg"])

if uploaded_file:

    # ================================
    # CARREGAR IMAGEM (QUALIDADE BOA SEM EXAGERO)
    # ================================
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(3,3))  # 🔥 equilíbrio ideal
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ================================
    # EXIBIÇÃO SEM CORTE (FIXA E NÍTIDA)
    # ================================
    st.subheader("1. Visualização da planta")

    # largura fixa (NÃO corta)
    max_width = 1400
    scale = min(max_width / image.width, 1.0)

    display_w = int(image.width * scale)
    display_h = int(image.height * scale)

    image_display = image.resize(
        (display_w, display_h),
        Image.Resampling.LANCZOS
    )

    coords = streamlit_image_coordinates(image_display, key="map")

    st.image(image_display)  # 🔥 SEM use_container_width

    # ================================
    # SALVAR CLIQUE
    # ================================
    if coords:
        x_real = int(coords["x"] / scale)
        y_real = int(coords["y"] / scale)

        st.session_state["click"] = (x_real, y_real)
        st.success(f"Clique salvo: {x_real}, {y_real}")

    # ================================
    # PRÉ-MAPEAMENTO
    # ================================
    st.subheader("2. Pré-mapeamento")

    if st.button("🔍 Mapear elementos"):

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 1)

        # remover linhas longas
        lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength=120,maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(thresh,(x1,y1),(x2,y2),0,10)

        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        result = img_cv.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 80 or area > 2000:
                continue

            x,y,w,h = cv2.boundingRect(cnt)

            aspect = max(w,h)/(min(w,h)+1)
            if aspect > 4:
                continue

            # ignorar legenda (parte inferior)
            if y > img_cv.shape[0] * 0.85:
                continue

            boxes.append((x,y,w,h))

            cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),2)

        st.session_state["boxes"] = boxes

        result_small = cv2.resize(
            result,
            (display_w, display_h)
        )

        st.session_state["pre_map_img"] = result_small

        st.success(f"{len(boxes)} elementos detectados")

    # ================================
    # MOSTRAR PRÉ-MAPEAMENTO
    # ================================
    if st.session_state["pre_map_img"] is not None:
        st.image(cv2.cvtColor(st.session_state["pre_map_img"], cv2.COLOR_BGR2RGB))

    # ================================
    # DETECÇÃO POR SIMILARIDADE
    # ================================
    st.subheader("3. Detectar iguais")

    if st.session_state["click"] and st.session_state["boxes"]:

        if st.button("🎯 Detectar iguais ao clicado"):

            x_click, y_click = st.session_state["click"]

            selected_box = None

            for (x,y,w,h) in st.session_state["boxes"]:
                if x < x_click < x+w and y < y_click < y+h:
                    selected_box = (x,y,w,h)
                    break

            if selected_box:

                x_ref, y_ref, w_ref, h_ref = selected_box

                result = img_cv.copy()
                count = 0

                for (x,y,w,h) in st.session_state["boxes"]:

                    if abs(w - w_ref) > 10:
                        continue

                    if abs(h - h_ref) > 10:
                        continue

                    ratio1 = w / (h+1)
                    ratio2 = w_ref / (h_ref+1)

                    if abs(ratio1 - ratio2) > 0.3:
                        continue

                    count += 1

                    cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),3)

                result_small = cv2.resize(result, (display_w, display_h))

                st.session_state["result_img"] = result_small

                st.success(f"{count} itens encontrados")

            else:
                st.warning("Clique não corresponde a um símbolo")

    # ================================
    # RESULTADO FINAL
    # ================================
    if st.session_state["result_img"] is not None:
        st.image(cv2.cvtColor(st.session_state["result_img"], cv2.COLOR_BGR2RGB))
