import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
from google import genai
from pyairtable import Api
import json

# ==========================================
# 1. CONFIGURAÇÕES E CHAVES DE API
# ==========================================
# Suas chaves já formatadas corretamente com as aspas:
GEMINI_API_KEY = "AIzaSyBzW5g9eGYfwJzOK2skZZsL1QEO0IPWd08"
AIRTABLE_TOKEN = "patM8RWo9vtME2k6s.c6924d385ad00d34ac603ceaff4b01e57d40ef4ed4ca94de8a5b24efb20"
AIRTABLE_BASE_ID = "appLhGvQWWr3E5Ow6"
AIRTABLE_TABLE_NAME = "Catalogo"

# ==========================================
# 2. INICIALIZAÇÃO DOS CLIENTES
# ==========================================
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    api_airtable = Api(AIRTABLE_TOKEN)
    tabela_precos = api_airtable.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
except Exception as e:
    st.error(f"Erro na configuração das chaves: {e}")

# ==========================================
# 3. INTERFACE DO SISTEMA (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Bot de Orçamentos", layout="wide")
st.title("💡 Sistema Híbrido: Visão + IA")
st.markdown("---")

uploaded_file = st.file_uploader("1. Carregue a prancha/planta (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preparar a imagem
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    st.subheader("2. Recorte o símbolo que deseja contar:")
    st.info("Arraste o quadrado verde para envolver apenas uma unidade do item (ex: uma luminária).")
    
    # Ferramenta de recorte
cropped_img = st_cropper(image, realtime_update=True, box_color='#00FF00')
    
    
    # Ajuste de sensibilidade da visão computacional
threshold = st.slider("Sensibilidade da Busca (Ajuste se não encontrar tudo)", min_value=0.5, max_value=0.99, value=0.8, step=0.01)

if st.button("Iniciar Contagem", type="primary"):
        # Converter o recorte para OpenCV
        crop_array = np.array(cropped_img)
        crop_cv2 = cv2.cvtColor(crop_array, cv2.COLOR_RGB2BGR)
        crop_gray = cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2GRAY)

        # Buscar pela planta toda
        res = cv2.matchTemplate(img_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        # Contar e marcar na imagem principal
        h, w = crop_gray.shape
        count = 0
        img_result = img_cv2.copy()
        
        # Filtro para evitar contar o mesmo símbolo duas vezes
        pontos = []
        for pt in zip(*loc[::-1]):
            if not any(np.linalg.norm(np.array(pt) - np.array(p)) < min(w, h)/2 for p in pontos):
                pontos.append(pt)
                cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                count += 1

        st.success(f"Foram encontrados **{count}** itens na planta!")
        st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="Mapa de itens localizados", use_column_width=True)

        # ==========================================
        # 4. ORÇAMENTO (AIRTABLE)
        # ==========================================
        st.markdown("---")
        st.subheader("3. Gerar Orçamento Automático")
        
        # Guardar a contagem na memória do sistema
        st.session_state['quantidade_itens'] = count

# Se a contagem já foi feita, exibe a busca
if 'quantidade_itens' in st.session_state:
    item_busca = st.text_input("Nome do item para buscar no Airtable (ex: Spot LED, Tomada):")
    
    if st.button("Buscar e Calcular") and item_busca:
        with st.spinner("Conectando ao catálogo Airtable..."):
            try:
                # Busca todos os itens (simplificado)
                records = tabela_precos.all()
                item_encontrado = None
                preco_unitario = 0.0
                
                for record in records:
                    campos = record.get('fields', {})
                    # IMPORTANTE: Se as colunas no seu Airtable tiverem nomes diferentes, troque "Nome" e "Preco" abaixo:
                    nome_airtable = str(campos.get('Nome', '')).lower()
                    if item_busca.lower() in nome_airtable:
                        item_encontrado = campos.get('Nome')
                        preco_unitario = float(campos.get('Preco', 0)) # ou 'Valor', dependendo de como você nomeou
                        break
                
                if item_encontrado:
                    quantidade = st.session_state['quantidade_itens']
                    total = quantidade * preco_unitario
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Item Localizado", item_encontrado)
                    col2.metric("Quantidade", f"{quantidade} un.")
                    col3.metric("Preço Unitário", f"R$ {preco_unitario:.2f}")
                    
                    st.success(f"### Valor Total Estimado: R$ {total:.2f}")
                else:
                    st.warning("Item não encontrado no catálogo. Verifique o nome digitado.")
            except Exception as e:
                st.error(f"Erro ao acessar o Airtable. Verifique se a tabela se chama 'Catalogo' e se possui colunas chamadas 'Nome' e 'Preco'. Detalhe: {e}")