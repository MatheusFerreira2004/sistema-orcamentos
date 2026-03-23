import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
from streamlit_cropper import st_cropper
import requests
import json

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E SECRETS (ORDEM É CRÍTICA)
# ==============================================================================
# Deve ser o PRIMEIRO comando do Streamlit
st.set_page_config(layout="wide", page_title="Orçamento IA - Goiânia")

# --- Configuração de Acesso ao Airtable (Configurar no Streamlit Cloud) ---
# Você deve criar uma seção [[secrets]] no painel do app com:
# AIRTABLE_API_KEY = "seu_token"
# AIRTABLE_BASE_ID = "seu_id"
# AIRTABLE_TABLE_NAME = "Catalogo" # Nome exato da sua tabela
try:
    airtable_api_key = st.secrets["AIRTABLE_API_KEY"]
    airtable_base_id = st.secrets["AIRTABLE_BASE_ID"]
    airtable_table_name = st.secrets.get("AIRTABLE_TABLE_NAME", "Catalogo")
    secrets_ok = True
except (KeyError, FileNotFoundError):
    st.sidebar.error("⚠️ Secrets do Airtable não configurados. O cálculo não funcionará.")
    secrets_ok = False

# ==============================================================================
# 2. MENU LATERAL E TÍTULO
# ==============================================================================
with st.sidebar:
    st.header("Configurações da IA")
    # Slider para sensibilidade (Ajuste para contar melhor)
    sensitivity = st.slider("Sensibilidade do Robô (CV2)", min_value=0.5, max_value=0.99, value=0.75, step=0.01)
    st.markdown("---")
    st.write("Orçamento Automatizado v1.0")

st.title("Sistema de Orçamento Automatizado - IA 🚀")
st.markdown("Plataforma piloto para escritórios de arquitetura e engenharia em Goiânia.")

# ==============================================================================
# 3. UPLOAD E PROCESSAMENTO DO ARQUIVO (PDF COM RGB FIX)
# ==============================================================================
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG ou PNG)", type=["pdf", "jpg", "png", "jpeg"])

image = None # Inicializa a variável

if uploaded_file is not None:
    try:
        # --- TRATAMENTO DE PDF ---
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Extraindo planta em Alta Resolução (4K/300 DPI) para análise..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                
                # Seleção de página se o PDF for grande
                if len(doc) > 1:
                    page_num = st.number_input(f"Este PDF tem {len(doc)} pranchas. Qual página deseja usar?", min_value=1, max_value=len(doc), value=1) - 1
                else:
                    page_num = 0
                    
                page = doc.load_page(page_num)
                
                # Renderiza em 4x (4K aprox) para manter qualidade técnica
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4)) 
                img_data = pix.tobytes("png")
                
                # --- CORREÇÃO CRÍTICA AQUI (.convert("RGB")) ---
                # Isso troca o fundo transparente do PDF por branco sólido, 
                # resolvendo o problema da planta não carregar.
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
            st.success(f"Prancha {page_num + 1} carregada com sucesso!")
                
        # --- TRATAMENTO DE IMAGENS NORMAIS ---
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
    
    # Cria colunas para o layout WIDE
    col_cropper, col_preview = st.columns([2, 1])
    
    with col_cropper:
        st.write("Arraste o quadrado verde para envolver apenas UM símbolo (justinho).")
        # st_cropper retorna a imagem recortada em alta resolução
        cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00', aspect_ratio=None, key="cropper")
    
    with col_preview:
        st.subheader("Pré-visualização do Alvo")
        if cropped_img:
            # Mostra o "molde" que a IA vai procurar
            st.image(cropped_img, caption="Símbolo Alvo")
            start_counting = st.button("Iniciar Contagem de Itens 🔍")
        else:
            st.info("Aguardando seleção na planta...")

    # --- Lógica de Contagem OpenCV ---
    if 'start_counting' in locals() and start_counting and cropped_img:
        st.header("2. Resultado da Contagem")
        
        with st.spinner("Robô analisando a planta... (isso pode levar alguns segundos)"):
            # Converte PIL Images para OpenCV format (BGR)
            img_raw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            template = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            
            h, w = template.shape[:2]
            
            # Template Matching
            res = cv2.matchTemplate(img_raw, template, cv2.TM_CCOEFF_NORMED)
            
            # Filtra resultados pela sensibilidade definida no sidebar
            loc = np.where(res >= sensitivity)
            
            # Lista para guardar matches e imagem resultante
            matches = []
            img_result = img_raw.copy()
            
            for pt in zip(*loc[::-1]): # Troca (y,x) para (x,y)
                matches.append(pt)
                # Desenha retângulo verde BGR (0, 255, 0)
                cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 4)

            total_counted = len(matches)
            
            # --- Exibe o Resultado da Contagem ---
            st.success(f"O robô encontrou **{total_counted}** itens idênticos ao selecionado!")
            
            # --- Exibe a Imagem Resultante (FIX: use_column_width=False, output_format="PNG" para qualidade 4K) ---
            st.markdown("**Mapa de Resultados (Zoom ativado):**")
            st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption=f"Total: {total_counted}", use_column_width=False, output_format="PNG")
            
            # Guarda a quantidade na sessão para usar no próximo passo
            st.session_state['last_count'] = total_counted
            st.session_state['counting_done'] = True

    else:
        # st.info("Selecione o símbolo e clique em 'Iniciar Contagem'.")
        pass

# ==============================================================================
# 5. BUSCA INTELIGENTE E CÁLCULO FINANCEIRO (AIRTABLE)
# ==============================================================================
# Só aparece se a contagem foi feita e os segredos estão configurados
if image is not None and st.session_state.get('counting_done', False) and secrets_ok:
    st.header("3. Gerar Orçamento (Busca no Banco de Dados)")
    
    total_counted = st.session_state.get('last_count', 0)
    
    # Busca Inteligente (Parcial e Insensível a Maiúsculas)
    st.write("Digite o nome cadastrado no Airtable (pode ser parcial). Ex: Digite 'Interruptor Simples' ou 'Interruptor Baixo'")
    product_search = st.text_input("Qual material é esse?")
    calculate_button = st.button("Buscar e Calcular Valor Total 💰")
    
    if calculate_button and product_search:
        with st.spinner("Buscando preço no Airtable..."):
            # --- Busca no Airtable ---
            url = f"https://api.airtable.com/v0/{airtable_base_id}/{airtable_table_name}"
            headers = {"Authorization": f"Bearer {airtable_api_key}"}
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status() # Lança exceção se houver erro HTTP
                data = response.json()
                records = data.get("records", [])
                
                found_product = None
                search_term_lower = product_search.strip().lower()
                
                # --- Busca Inteligente (pela coluna 'Nome') ---
                for record in records:
                    fields = record.get("fields", {})
                    product_name = fields.get("Nome", "")
                    
                    if search_term_lower in product_name.lower():
                        found_product = record
                        actual_product_name = product_name # Nome real cadastrado
                        break # Para na primeira ocorrência
                
                if found_product:
                    fields = found_product.get("fields", {})
                    # Tenta pegar 'Preco' (sem cedilha)
                    product_price = fields.get("Preco", 0.0)
                    
                    # --- Matemática Financeira (FLOAT) ---
                    try:
                        price_float = float(product_price)
                    except (ValueError, TypeError):
                        price_float = 0.0
                    
                    if price_float == 0.0:
                        st.warning(f"O preço cadastrado para '{actual_product_name}' é zero. Verifique no Airtable.")
                    
                    total_estimate = total_counted * price_float
                    
                    # --- Exibe o Orçamento Profissional ---
                    st.markdown("---")
                    st.subheader(f"Orçamento para: {actual_product_name}")
                    col_p, col_q, col_t = st.columns(3)
                    with col_p:
                        # Formatação Brasileira R$
                        formatted_price = f"{price_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        st.metric("Preço Unitário cadastrado (R$)", formatted_price)
                    with col_q:
                        st.metric("Quantidade Contada", f"{total_counted}")
                    with col_t:
                        # Formatação Brasileira R$ do Total
                        formatted_total = f"{total_estimate:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        st.metric("Orçamento Total estimado (R$)", formatted_total, help="Total = Preço x Quantidade")
                    st.success("Cálculo realizado com sucesso!")
                    
                else:
                    st.error(f"❌ Item '{product_search}' não encontrado no banco de dados. Verifique a coluna 'Nome' no Airtable.")
            
            except Exception as e:
                st.error(f"Erro genérico ao calcular: {e}")
