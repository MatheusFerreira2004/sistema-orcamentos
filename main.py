import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import requests
import urllib.parse
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURAÇÃO INICIAL
# ==========================================
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
    st.error("⚠️ Erro: Chaves do Airtable não encontradas nos Secrets.")
    st.stop()

# ==========================================
# 2. MEMÓRIA DO SISTEMA
# ==========================================
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None

# ==========================================
# 3. UPLOAD DE ARQUIVO
# ==========================================
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Extração de PDF em Alta Resolução
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Processando PDF em alta resolução..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(6, 6))
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")
        st.subheader("1. Clique no símbolo que deseja contar na planta abaixo:")
        st.info("Dica: Clique bem no centro da cor (ex: no meio do azul do embutido).")

        # ==========================================
        # 4. CAPTURA DE COORDENADA E COR
        # ==========================================
        coords = streamlit_image_coordinates(image, key="mapa_clique")

        if coords:
            x, y = coords["x"], coords["y"]
            st.write(f"📍 Coordenada selecionada: X={x}, Y={y}")

            # Converte a imagem para o formato do OpenCV
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Captura a cor exata do pixel clicado
            b, g, r = img_cv[y, x]
            pixel = np.uint8([[[b, g, r]]])
            hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)

            # CONVERSÃO CRÍTICA PARA INT (Evita erro matemático de Underflow)
            h, s, v = int(hsv_pixel[0][0][0]), int(hsv_pixel[0][0][1]), int(hsv_pixel[0][0][2])

            st.write(f"🎨 Cor detectada pelo Robô: Hue={h}, Sat={s}, Val={v}")

            # ==========================================
            # 5. AJUSTE FINO E DETECÇÃO
            # ==========================================
            st.subheader("Ajuste fino da detecção")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                h_range = st.slider("Tolerância de Cor", 5, 40, 15)
            with col2:
                s_min = st.slider("Ignorar Tons Cinza", 0, 255, 50)
            with col3:
                v_min = st.slider("Ignorar Tons Escuros", 0, 255, 50)
            with col4:
                area_min = st.slider("Área Mínima (Tamanho)", 10, 500, 80)

            # Cria a "janela" de cores aceitáveis
            lower = np.array([max(0, h - h_range), s_min, v_min])
            upper = np.array([min(179, h + h_range), 255, 255])

            if st.button("🔍 Detectar símbolos na planta"):
                with st.spinner("Varrendo a planta por cor..."):

                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)

                    # Limpeza Morfológica (Junta pedaços separados por linhas pretas)
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    # Encontra os contornos (os objetos azuis isolados)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    count = 0
                    img_result = img_cv.copy()

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > area_min:
                            count += 1
                            x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                            # Desenha um retângulo vermelho para destacar bem na tela
                            cv2.rectangle(img_result, (x_box, y_box), (x_box+w_box, y_box+h_box), (0, 0, 255), 6)

                    # Salva o resultado na memória
                    st.session_state['total_itens'] = count
                    st.session_state['mapa_resultado'] = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # ==========================================
        # 6. EXIBIÇÃO DO RESULTADO E ORÇAMENTO
        # ==========================================
        if st.session_state['mapa_resultado'] is not None:
            st.success(f"✅ O robô encontrou {st.session_state['total_itens']} itens com essa cor e tamanho!")
            st.image(st.session_state['mapa_resultado'], use_column_width=False)

            st.markdown("---")
            st.subheader("2. Gerar Orçamento 💰")

            product_search = st.text_input("Qual o nome do produto cadastrado no Airtable?")
            
            if st.button("Calcular Orçamento") and product_search:
                with st.spinner("Consultando banco de dados..."):
                    table_encoded = urllib.parse.quote(AIRTABLE_TABLE_NAME)
                    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_encoded}"
                    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

                    response = requests.get(url, headers=headers)

                    if response.status_code == 200:
                        records = response.json().get("records", [])
                        found = None
                        search_term = product_search.lower().strip()

                        for record in records:
                            nome = record.get("fields", {}).get("Nome", "")
                            if search_term in nome.lower():
                                found = record
                                break

                        if found:
                            nome_real = found["fields"].get("Nome", "Produto")
                            preco_str = found["fields"].get("Preco", found["fields"].get("Preço", 0))
                            
                            try:
                                preco = float(preco_str)
                            except:
                                preco = 0.0

                            total = preco * st.session_state['total_itens']

                            st.success(f"Produto localizado: **{nome_real}**")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Preço Unitário (R$)", f"{preco:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                            col2.metric("Quantidade na Planta", st.session_state['total_itens'])
                            col3.metric("Total Estimado (R$)", f"{total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                        else:
                            st.warning(f"❌ Produto '{product_search}' não encontrado na tabela.")
                    else:
                        st.error(f"Erro ao conectar com Airtable. Código: {response.status_code}")

    except Exception as e:
        st.error(f"Ocorreu um erro no processamento: {e}")
