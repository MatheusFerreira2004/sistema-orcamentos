import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
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

if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None

# ==========================================
# 2. UPLOAD E REDIMENSIONAMENTO (O Zoom Responsivo)
# ==========================================
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Processando PDF em alta resolução..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(6, 6))
                image_high_res = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        else:
            image_high_res = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")
        st.subheader("1. Clique no símbolo que deseja contar na planta abaixo:")
        
        # --- Evita que a imagem quebre a tela, mas mantém a alta resolução para o robô ---
        largura_tela = 1000 
        fator_escala = largura_tela / float(image_high_res.size[0])
        altura_tela = int(float(image_high_res.size[1]) * float(fator_escala))
        image_display = image_high_res.resize((largura_tela, altura_tela), Image.Resampling.LANCZOS)

        # ==========================================
        # 3. CAPTURA DE COORDENADA E COR (Média Segura)
        # ==========================================
        coords = streamlit_image_coordinates(image_display, key="mapa_clique")

        if coords:
            # Converte clique da tela para a imagem gigante
            x_real = int(coords["x"] / fator_escala)
            y_real = int(coords["y"] / fator_escala)

            img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)

            # Usa a "Média de Cor" do código novo, que é uma ideia excelente para evitar pegar um pixel preto por acidente
            region = img_cv[max(0, y_real-5):min(img_cv.shape[0], y_real+5), max(0, x_real-5):min(img_cv.shape[1], x_real+5)]
            avg_color = np.mean(region, axis=(0,1)).astype(int)

            pixel = np.uint8([[avg_color]])
            hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)

            h, s, v = int(hsv_pixel[0][0][0]), int(hsv_pixel[0][0][1]), int(hsv_pixel[0][0][2])

            st.write(f"🎨 HSV detectado (Média 10x10px): H={h}, S={s}, V={v}")

            # ==========================================
            # 4. PAINEL DE AJUSTES (OS SLIDERS VOLTARAM!)
            # ==========================================
            st.subheader("2. Ajustes da detecção geométrica")
            st.info("💡 **Dica para perfis de LED:** Se for um item longo, aumente a **Área máxima** e a **Proporção máxima**.")

            col1, col2 = st.columns(2)
            with col1:
                h_range = st.slider("Range de Hue (Tolerância de Cor)", 5, 40, 15) # Aumentei o padrão para 15
                s_min = st.slider("Saturação mínima (Ignora cinza)", 0, 255, 40) # Baixei para 40
                v_min = st.slider("Valor mínimo (Ignora escuro)", 0, 255, 40)
            with col2:
                min_area = st.slider("Área mínima (Poeira)", 10, 500, 80)
                max_area = st.slider("Área máxima (Legendas/Blocos)", 100, 15000, 1500) # Limite bem maior para perfis
                ratio_min = st.slider("Proporção mínima (larg/alt)", 0.1, 1.0, 0.3) # Começa menor
                ratio_max = st.slider("Proporção máxima (larg/alt)", 1.0, 20.0, 5.0) # Permite itens longos (perfil led)
                extent_min = st.slider("Preenchimento mínimo", 0.1, 1.0, 0.4)

            lower = np.array([max(0, h - h_range), s_min, v_min])
            upper = np.array([min(179, h + h_range), 255, 255])

            if st.button("🔍 Detectar símbolos de forma inteligente"):
                with st.spinner("Analisando com Inteligência de Contornos..."):

                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)

                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    count = 0
                    img_result = img_cv.copy()

                    for cnt in contours:
                        area = cv2.contourArea(cnt)

                        if area < min_area or area > max_area:
                            continue

                        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)

                        if h_box == 0:
                            continue
                            
                        # A Proporção agora aceita itens compridos (se você ajustar no slider)
                        ratio = float(w_box) / float(h_box)
                        # Para perfis longos, o w pode ser menor que o h (linha vertical), então verificamos os dois lados
                        ratio_invertido = float(h_box) / float(w_box)
                        
                        if not ((ratio_min < ratio < ratio_max) or (ratio_min < ratio_invertido < ratio_max)):
                            continue

                        rect_area = w_box * h_box
                        if rect_area == 0:
                            continue
                            
                        extent = float(area) / float(rect_area)
                        if extent < extent_min:
                            continue

                        count += 1
                        cv2.rectangle(img_result, (x_box, y_box), (x_box+w_box, y_box+h_box), (0, 0, 255), 6)

                    st.session_state['total_itens'] = count
                    st.session_state['mapa_resultado'] = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # ==========================================
        # 5. RESULTADO E INTEGRAÇÃO AIRTABLE
        # ==========================================
        if st.session_state['mapa_resultado'] is not None:
            st.success(f"✅ Encontrados {st.session_state['total_itens']} itens legítimos!")
            st.image(st.session_state['mapa_resultado'], use_column_width=True)

            st.markdown("---")
            st.subheader("3. Gerar Orçamento 💰")

            product_search = st.text_input("Nome do produto no Airtable (ex: Spot Embutido)")

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
