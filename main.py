import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz
import io
from streamlit_cropper import st_cropper
import requests
import urllib.parse

# 1. Configuração da página
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

# --- INICIALIZAR MEMÓRIA DO ROBÔ ---
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'contagem_feita' not in st.session_state:
    st.session_state['contagem_feita'] = False
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None

# --- UPLOAD ---
uploaded_file = st.file_uploader("Suba sua Planta (PDF ou Imagem)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")

        # --- ÁREA DE RECORTE ---
        col_crop, col_res = st.columns([2, 1])

        with col_crop:
            st.subheader("1. Selecione o Símbolo")
            cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00', aspect_ratio=None)

        with col_res:
            st.subheader("2. Alvo e Configurações")
            # DEVOLVENDO A PRÉ-VISUALIZAÇÃO!
            if cropped_img:
                st.image(cropped_img, caption="Símbolo a ser buscado")
                
            sensibilidade = st.slider("Precisão do Robô (0.95 = Máxima)", 0.80, 0.99, 0.93, 0.01)
            btn_contar = st.button("CONTAR ITENS NA PLANTA 🔍")

        # --- LÓGICA DE CONTAGEM ---
        if btn_contar:
            with st.spinner("Analisando a planta..."):
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                temp_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
                h, w = temp_cv.shape[:2]
                
                res = cv2.matchTemplate(img_cv, temp_cv, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= sensibilidade)
                
                pontos = []
                img_result = img_cv.copy()
                
                for pt in zip(*loc[::-1]):
                    if not any(abs(pt[0]-p[0]) < w/2 and abs(pt[1]-p[1]) < h/2 for p in pontos):
                        pontos.append(pt)
                        cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 4)
                
                # Salva na memória
                st.session_state['total_itens'] = len(pontos)
                st.session_state['contagem_feita'] = True
                st.session_state['mapa_resultado'] = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # --- EXIBIR RESULTADOS E BUSCA NO AIRTABLE ---
        if st.session_state['contagem_feita']:
            st.success(f"✅ Encontrados {st.session_state['total_itens']} itens na planta!")
            st.image(st.session_state['mapa_resultado'], caption="Mapa de Contagem (Verde = Encontrado)", use_column_width=False)
            
            st.markdown("---")
            st.subheader("3. Gerar Orçamento 💰")
            
            # DEVOLVENDO A BARRA DE PESQUISA!
            product_search = st.text_input("Qual o nome do material no Airtable? (ex: Spot Embutido)")
            btn_orcamento = st.button("Buscar Preço e Calcular Total")

            if btn_orcamento and product_search:
                with st.spinner(f"Buscando '{product_search}' no Airtable..."):
                    table_encoded = urllib.parse.quote(AIRTABLE_TABLE_NAME)
                    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_encoded}"
                    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
                    
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        records = response.json().get("records", [])
                        found_product = None
                        search_term = product_search.strip().lower()
                        
                        # Busca Inteligente
                        for record in records:
                            nome_bd = record.get("fields", {}).get("Nome", "")
                            if search_term in nome_bd.lower():
                                found_product = record
                                actual_name = nome_bd
                                break
                        
                        if found_product:
                            fields = found_product.get("fields", {})
                            preco_str = fields.get("Preco", fields.get("Preço", 0.0))
                            
                            try:
                                preco_float = float(preco_str)
                            except:
                                preco_float = 0.0
                            
                            total_estimate = st.session_state['total_itens'] * preco_float
                            
                            # Exibe o painel financeiro
                            st.subheader(f"Orçamento: {actual_name}")
                            col_p, col_q, col_t = st.columns(3)
                            col_p.metric("Preço Unitário (R$)", f"{preco_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                            col_q.metric("Quantidade na Planta", st.session_state['total_itens'])
                            col_t.metric("Total Estimado (R$)", f"{total_estimate:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                        else:
                            st.warning(f"❌ Item '{product_search}' não encontrado na aba '{AIRTABLE_TABLE_NAME}'. Verifique como está escrito lá.")
                    else:
                        st.error(f"Erro {response.status_code} de conexão com o Airtable.")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
