import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
from streamlit_cropper import st_cropper
import requests
import urllib.parse

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E SECRETS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Orçamento IA - Goiânia")

try:
    airtable_api_key = st.secrets["AIRTABLE_API_KEY"]
    airtable_base_id = st.secrets["AIRTABLE_BASE_ID"]
    airtable_table_name = st.secrets.get("AIRTABLE_TABLE_NAME", "Estoque Luz Cor")
    secrets_ok = True
except (KeyError, FileNotFoundError):
    st.sidebar.error("⚠️ Secrets do Airtable não configurados. O cálculo não funcionará.")
    secrets_ok = False

# ==============================================================================
# 2. MENU LATERAL E TÍTULO
# ==============================================================================
with st.sidebar:
    st.header("Configurações da IA")
    # Slider para sensibilidade
    sensitivity = st.slider("Sensibilidade do Robô (CV2)", min_value=0.5, max_value=0.99, value=0.75, step=0.01)
    st.markdown("---")
    st.write("Orçamento Automatizado v1.1")

st.title("Sistema de Orçamento Automatizado - IA 🚀")
st.markdown("Plataforma piloto para escritórios de arquitetura e engenharia.")

# ==============================================================================
# 3. UPLOAD E PROCESSAMENTO DO ARQUIVO (PDF COM RGB FIX)
# ==============================================================================
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG ou PNG)", type=["pdf", "jpg", "png", "jpeg"])

image = None 

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Extraindo planta em Alta Resolução (4K/300 DPI) para análise..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                
                if len(doc) > 1:
                    page_num = st.number_input(f"Este PDF tem {len(doc)} pranchas. Qual página deseja usar?", min_value=1, max_value=len(doc), value=1) - 1
                else:
                    page_num = 0
                    
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4)) 
                img_data = pix.tobytes("png")
                
                # Força o fundo branco para evitar erro no st_cropper
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
            st.success(f"Prancha {page_num + 1} carregada com sucesso!")
                
        else:
            image = Image.open(uploaded_file)
            st.success("Planta carregada com sucesso!")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

# ==============================================================================
# 4. ÁREA DE RECORTE E CONTAGEM
# ==============================================================================
if image is not None:
    st.header("1. Selecione o Símbolo do Material")
    
    col_cropper, col_preview = st.columns([2, 1])
    
    with col_cropper:
        st.write("Arraste o quadrado verde para envolver apenas UM símbolo (justinho).")
        cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00', aspect_ratio=None, key="cropper")
    
    with col_preview:
        st.subheader("Pré-visualização do Alvo")
        if cropped_img:
            st.image(cropped_img, caption="Símbolo Alvo")
            start_counting = st.button("Iniciar Contagem de Itens 🔍")
        else:
            st.info("Aguardando seleção na planta...")

    # --- Lógica de Contagem OpenCV ---
    if 'start_counting' in locals() and start_counting and cropped_img:
        st.header("2. Resultado da Contagem")
        
        with st.spinner("Robô analisando a planta... (isso pode levar alguns segundos)"):
            img_raw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            template = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            
            h, w = template.shape[:2]
            
            res = cv2.matchTemplate(img_raw, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= sensitivity)
            
            matches = []
            img_result = img_raw.copy()
            
            for pt in zip(*loc[::-1]):
                matches.append(pt)
                cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 4)

            total_counted = len(matches)
            
            st.success(f"O robô encontrou **{total_counted}** itens idênticos ao selecionado!")
            st.markdown("**Mapa de Resultados (Zoom ativado):**")
            st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption=f"Total: {total_counted}", use_column_width=False, output_format="PNG")
            
            st.session_state['last_count'] = total_counted
            st.session_state['counting_done'] = True

# ==============================================================================
# 5. BUSCA INTELIGENTE E CÁLCULO FINANCEIRO (AIRTABLE)
# ==============================================================================
if image is not None and st.session_state.get('counting_done', False) and secrets_ok:
    st.header("3. Gerar Orçamento (Busca no Banco de Dados)")
    
    total_counted = st.session_state.get('last_count', 0)
    
    st.write("Digite o nome cadastrado no Airtable (pode ser parcial).")
    product_search = st.text_input("Qual material é esse?")
    calculate_button = st.button("Buscar e Calcular Valor Total 💰")
    
    if calculate_button and product_search:
        with st.spinner("Buscando preço no Airtable..."):
            
            # --- CORREÇÃO DO ERRO 403 (URL ENCODE) ---
            table_safe_name = urllib.parse.quote(airtable_table_name)
            url = f"https://api.airtable.com/v0/{airtable_base_id}/{table_safe_name}"
            
            headers = {"Authorization": f"Bearer {airtable_api_key}"}
            
            try:
                response = requests.get(url, headers=headers)
                
                if response.status_code == 403:
                    st.error("❌ Erro 403: O Token funciona, mas não tem permissão para acessar esta Base. Verifique o 'Access' nas configurações do Token no Airtable.")
                else:
                    response.raise_for_status()
                    data = response.json()
                    records = data.get("records", [])
                    
                    found_product = None
                    search_term_lower = product_search.strip().lower()
                    
                    for record in records:
                        fields = record.get("fields", {})
                        product_name = fields.get("Nome", "")
                        
                        if search_term_lower in product_name.lower():
                            found_product = record
                            actual_product_name = product_name
                            break
                    
                    if found_product:
                        fields = found_product.get("fields", {})
                        # Tenta buscar "Preco" ou "Preço" (para evitar erros de digitação no Airtable)
                        product_price = fields.get("Preco", fields.get("Preço", 0.0))
                        
                        try:
                            price_float = float(product_price)
                        except:
                            price_float = 0.0
                        
                        total_estimate = total_counted * price_float
                        
                        st.markdown("---")
                        st.subheader(f"Orçamento para: {actual_product_name}")
                        col_p, col_q, col_t = st.columns(3)
                        with col_p:
                            st.metric("Preço Unitário (R$)", f"{price_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                        with col_q:
                            st.metric("Quantidade", f"{total_counted}")
                        with col_t:
                            st.metric("Total Estimado (R$)", f"{total_estimate:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                        st.success("Orçamento gerado com sucesso!")
                    else:
                        st.error(f"❌ Item '{product_search}' não encontrado na tabela '{airtable_table_name}'. Verifique como está escrito no Airtable.")
            
            except Exception as e:
                st.error(f"Erro ao conectar com Airtable: {e}")
