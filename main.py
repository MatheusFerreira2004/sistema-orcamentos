import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("IA de Orçamento Inteligente 🚀")

# MEMÓRIA DE SÍMBOLOS
if "symbols" not in st.session_state:
    st.session_state["symbols"] = []

uploaded_file = st.file_uploader("Suba a planta", type=["pdf","png","jpg","jpeg"])

if uploaded_file:

    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(6,6))
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    # ZOOM
    zoom = st.slider("Zoom", 0.5, 2.5, 1.5)
    image_disp = image.resize(
        (int(image.width*zoom), int(image.height*zoom)),
        Image.Resampling.LANCZOS
    )

    coords = streamlit_image_coordinates(image_disp)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # =====================================
    # APRENDER NOVO SÍMBOLO
    # =====================================
    st.subheader("1. Ensinar símbolo")

    nome = st.text_input("Nome do símbolo (ex: Embutido)")

    if coords and st.button("Salvar símbolo"):
        x = int(coords["x"]/zoom)
        y = int(coords["y"]/zoom)

        region = img_cv[y-5:y+5, x-5:x+5]
        avg = np.mean(region, axis=(0,1)).astype(int)

        hsv = cv2.cvtColor(np.uint8([[avg]]), cv2.COLOR_BGR2HSV)
        h,s,v = map(int, hsv[0][0])

        st.session_state["symbols"].append({
            "name": nome,
            "h": h,
            "range": 15,
            "min_area": 100,
            "max_area": 600
        })

        st.success(f"Símbolo '{nome}' salvo!")

    # =====================================
    # LISTA DE SÍMBOLOS
    # =====================================
    st.subheader("Símbolos cadastrados")

    for s in st.session_state["symbols"]:
        st.write(s)

    # =====================================
    # DETECTAR TODOS
    # =====================================
    if st.button("🔍 Detectar tudo automaticamente"):

        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        result = img_cv.copy()

        total_geral = {}

        # remover linhas
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

        mask_global = np.zeros(gray.shape, dtype=np.uint8)

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(mask_global,(x1,y1),(x2,y2),255,10)

        for symbol in st.session_state["symbols"]:

            lower = np.array([max(0, symbol["h"]-symbol["range"]), 50, 50])
            upper = np.array([min(179, symbol["h"]+symbol["range"]), 255, 255])

            mask = cv2.inRange(hsv_img, lower, upper)

            # remover linhas da máscara
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_global))

            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < symbol["min_area"] or area > symbol["max_area"]:
                    continue

                x,y,w,h = cv2.boundingRect(cnt)

                aspect = max(w,h)/(min(w,h)+1)
                if aspect > 3:
                    continue

                count += 1
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(result, symbol["name"], (x,y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

            total_geral[symbol["name"]] = count

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        st.success(total_geral)
