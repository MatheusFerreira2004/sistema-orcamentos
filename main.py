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
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURAÇÃO INICIAL E MEMÓRIA
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

# Inicialização das Memórias do Robô
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None
if 'carrinho' not in st.session_state:
    st.session_state['carrinho'] = []
if 'produto_atual' not in st.session_state:
    st.session_state['produto_atual'] = None

# ==========================================
# 2. UPLOAD E RESOLUÇÃO ULTRA-LEVE
# ==========================================
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Extraindo PDF (Modo de Baixo Consumo de Memória)..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_high_res = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                
                del doc, page, pix
                gc.collect() 
        else:
            image_high_res = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")
        st.subheader("1. Clique BEM NO CENTRO do símbolo que deseja contar:")
        
        largura_tela = 1000 
        fator_escala = largura_tela / float(image_high_res.size[0])
        altura_tela = int(float(image_high_res.size[1]) * float(fator_escala))
        image_display = image_high_res.resize((largura_tela, altura_tela), Image.Resampling.LANCZOS)

        # ==========================================
        # 3. CAPTURA DE FORMA (MOTOR EM TONS DE CINZA)
        # ==========================================
        coords = streamlit_image_coordinates(image_display, key="mapa_clique")

        if coords:
            x_real = int(coords["x"] / fator_escala)
            y_real = int(coords["y"] / fator_escala)

            img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) 

            st.subheader("2. Ajuste o recorte do seu alvo")
            st.info("🚨 O quadrado abaixo deve conter APENAS o símbolo, sem pegar linhas vizinhas.")
            
            box_size = st.slider("Tamanho da área de captura", 5, 80, 20)

            y1, y2 = max(0, y_real - box_size), min(img_cv.shape[0], y_real + box_size)
            x1, x2 = max(0, x_real - box_size), min(img_cv.shape[1], x_real + box_size)
            
            template_color = img_cv[y1:y2, x1:x2]
            template_gray = img_gray[y1:y2, x1:x2]

            col_target, col_config = st.columns([1, 3])
            
            with col_target:
                st.image(cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB), caption="Símbolo Capturado", width=150)
            
            with col_config:
                threshold = st.slider("Precisão da Forma (0.90 = Idêntico)", 0.50, 0.99, 0.85, 0.01)

                if st.button("🔍 Procurar em Todos os Ângulos na Planta"):
                    with st.spinner("Calculando geometria com Inteligência Leve..."):
                        
                        pontos = []
                        img_result = img_cv.copy() 
                        
                        rotations_gray = [
                            template_gray,
                            cv2.rotate(template_gray, cv2.ROTATE_90_CLOCKWISE),
                            cv2.rotate(template_gray, cv2.ROTATE_180),
                            cv2.rotate(template_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        ]
                        
                        raio_seguranca = max(template_gray.shape[0], template_gray.shape[1]) / 2

                        for rot_template_gray in rotations_gray:
                            res = cv2.matchTemplate(img_gray, rot_template_gray, cv2.TM_CCOEFF_NORMED)
                            loc = np.where(res >= threshold)
                            h_tmpl, w_tmpl = rot_template_gray.shape[:2]
                            
                            for pt in zip(*loc[::-1]):
                                if not any(abs(pt[0]-p[0]) < raio_seguranca and abs(pt[1]-p[1]) < raio_seguranca for p in pontos):
                                    pontos.append(pt)
                                    cv2.rectangle(img_result, pt, (pt[0] + w_tmpl, pt[1] + h_tmpl), (0, 0, 255), 4)
                            
                            del res, loc
                            gc.collect()
                        
                        fator_reducao = 1500 / float(img_result.shape[1])
                        nova_largura = 1500
                        nova_altura = int(img_result.shape[0] * fator_reducao)
                        img_result_small = cv2.resize(img_result, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

                        st.session_state['total_itens'] = len(pontos)
                        st.session_state['mapa_resultado'] = cv2.cvtColor(img_result_small, cv2.COLOR_BGR2RGB)
                        
                        st.session_state['produto_atual'] = None

                        del img_cv, img_gray, img_result, image_high_res
                        gc.collect()

        # ==========================================
        # 4. RESULTADO E INTEGRAÇÃO AIRTABLE
        # ==========================================
        if st.session_state['mapa_resultado'] is not None:
            st.success(f"✅ O robô encontrou {st.session_state['total_itens']} símbolos na prancha inteira!")
            st.image(st.session_state['mapa_resultado'], use_container_width=True)

            st.markdown("---")
            st.subheader("3. Refino e Busca no Banco de Dados 💰")

            col_desc, col_busca = st.columns([1, 2])
            
            with col_desc:
                desconto = st.number_input("Descontar símbolos da legenda:", min_value=0, max_value=20, value=1)
                total_final = max(0, st.session_state['total_itens'] - desconto)
                st.info(f"Total real para orçamento: **{total_final} itens**")

            with col_busca:
                product_search = st.text_input("Nome do produto no Airtable (ex: Interruptor)")

            if st.button("Buscar Preço no Airtable") and product_search:
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

                            total_orcamento = preco * total_final

                            st.session_state['produto_atual'] = {
                                "Produto": nome_real,
                                "Quantidade": total_final,
                                "Preço Unitário (R$)": preco,
                                "Subtotal (R$)": total_orcamento
                            }
                        else:
                            st.warning(f"❌ Produto '{product_search}' não encontrado na tabela.")
                    else:
                        st.error(f"Erro ao conectar com Airtable. Código: {response.status_code}")

            # ==========================================
            # 5. EXIBE O PRODUTO E O BOTÃO DE ADICIONAR
            # ==========================================
            if st.session_state['produto_atual']:
                prod = st.session_state['produto_atual']
                st.success(f"Produto localizado: **{prod['Produto']}**")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Preço Unitário (R$)", f"{prod['Preço Unitário (R$)']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                col2.metric("Quantidade Real", prod['Quantidade'])
                col3.metric("Total Estimado (R$)", f"{prod['Subtotal (R$)']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🛒 Adicionar item ao Orçamento Geral", type="primary"):
                    st.session_state['carrinho'].append(prod)
                    st.session_state['produto_atual'] = None 
                    st.success("Item adicionado com sucesso! Role a página para baixo para ver o Orçamento Geral.")
                    st.rerun() 

        # ==========================================
        # 6. O CARRINHO DE ORÇAMENTO E EXCEL (TABELA FINAL)
        # ==========================================
        st.markdown("---")
        st.header("📋 Orçamento Geral do Projeto")
        
        if len(st.session_state['carrinho']) > 0:
            df_carrinho = pd.DataFrame(st.session_state['carrinho'])
            
            st.dataframe(
                df_carrinho.style.format({
                    "Preço Unitário (R$)": "{:.2f}",
                    "Subtotal (R$)": "{:.2f}"
                }), 
                use_container_width=True
            )

            total_geral = df_carrinho["Subtotal (R$)"].sum()
            st.subheader(f"💰 CUSTO TOTAL: R$ {total_geral:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            # --- A MÁGICA DO EXCEL ---
            col_excel, col_limpar = st.columns([1, 1])
            
            with col_excel:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_carrinho.to_excel(writer, index=False, sheet_name='Orçamento')
                
                st.download_button(
                    label="📊 Baixar Orçamento em Excel",
                    data=buffer.getvalue(),
                    file_name="orcamento_projeto.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

            with col_limpar:
                if st.button("🗑️ Limpar Orçamento e Começar de Novo"):
                    st.session_state['carrinho'] = []
                    st.rerun()
        else:
            st.info("O seu carrinho está vazio. Escaneie os símbolos na planta e adicione-os aqui.")

    except Exception as e:
        st.error(f"Ocorreu um erro no processamento: {e}")
