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
st.title("Sistema de Orçamento Inteligente 🚀")

# ================================
# MEMÓRIA
# ================================
if "result_img" not in st.session_state:
    st.session_state["result_img"] = None

# ================================
# UPLOAD
# ================================
uploaded_file = st.file_uploader("Suba sua planta", type=["pdf","png","jpg","jpeg"])

if uploaded_file:

    # ================================
    # PROCESSAMENTO EM ALTA QUALIDADE
    # ================================
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(5,5))
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h_img, w_img = img_cv.shape[:2]

    # ================================
    # VIEWPORT (ANTI-TRAVAMENTO)
    # ================================
    st.subheader("Navegação da planta")

    zoom = st.slider("Zoom", 1.0, 5.0, 2.0)

    view_w = int(w_img / zoom)
    view_h = int(h_img / zoom)

    x_pos = st.slider("Mover Horizontal", 0, max(0, w_img - view_w), 0)
    y_pos = st.slider("Mover Vertical", 0, max(0, h_img - view_h), 0)

    viewport = img_cv[y_pos:y_pos+view_h, x_pos:x_pos+view_w]
    viewport_display = cv2.resize(viewport, (800, 600))

    coords = streamlit_image_coordinates(viewport_display)

    st.image(cv2.cvtColor(viewport_display, cv2.COLOR_BGR2RGB))

    # ================================
    # DETECÇÃO POR CLIQUE
    # ================================
    if coords:

        click_x = int(coords["x"] * (view_w / 800)) + x_pos
        click_y = int(coords["y"] * (view_h / 600)) + y_pos

        # média de cor (mais estável)
        region = img_cv[click_y-5:click_y+5, click_x-5:click_x+5]
        avg_color = np.mean(region, axis=(0,1)).astype(int)

        hsv_pixel = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)
        h,s,v = map(int, hsv_pixel[0][0])

        st.write(f"HSV: {h}, {s}, {v}")

        # ================================
        # AJUSTES
        # ================================
        col1, col2 = st.columns(2)

        with col1:
            h_range = st.slider("Hue Range", 5, 40, 15)
            min_area = st.slider("Área mínima", 50, 500, 120)

        with col2:
            max_area = st.slider("Área máxima", 200, 2000, 800)
            aspect_limit = st.slider("Filtro linha (aspect ratio)", 2.0, 6.0, 3.0)

        lower = np.array([max(0, h-h_range), 50, 50])
        upper = np.array([min(179, h+h_range), 255, 255])

        # ================================
        # DETECTAR
        # ================================
        if st.button("🔍 Detectar símbolos"):

            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # suavização
            mask = cv2.GaussianBlur(mask, (7,7), 0)
            _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

            # morfologia
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

            # ================================
            # REMOVER LINHAS (PERFIL LED)
            # ================================
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

            if lines is not None:
                for line in lines:
                    x1,y1,x2,y2 = line[0]
                    cv2.line(mask,(x1,y1),(x2,y2),0,10)

            # ================================
            # CONTORNOS
            # ================================
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            centers = []
            result = img_cv.copy()

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < min_area or area > max_area:
                    continue

                x,y,w,h = cv2.boundingRect(cnt)

                # filtro linha
                aspect = max(w,h)/(min(w,h)+1)
                if aspect > aspect_limit:
                    continue

                cx = x + w//2
                cy = y + h//2

                # evitar duplicados
                duplicate = False
                for px,py in centers:
                    if abs(cx-px)<20 and abs(cy-py)<20:
                        duplicate = True
                        break

                if duplicate:
                    continue

                centers.append((cx,cy))
                count += 1

                cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),3)

            # ================================
            # EXIBIÇÃO LEVE
            # ================================
            result_small = cv2.resize(result, (1000, int(1000*h_img/w_img)))

            st.session_state["result_img"] = result_small
            st.success(f"{count} itens encontrados")

    # ================================
    # RESULTADO FINAL
    # ================================
    if st.session_state["result_img"] is not None:
        st.image(cv2.cvtColor(st.session_state["result_img"], cv2.COLOR_BGR2RGB))
