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
import base64
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURAÇÃO E SUPORTE
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA v2.2")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except: return None

def get_image_base64(img):
    """Converte imagem para Base64 otimizado para não travar o navegador"""
    buffered = io.BytesIO()
    # Convertemos para JPEG com qualidade 60% para o fundo do canvas não ficar preto
    img.convert("RGB").save(buffered, format="JPEG", quality=60)
    return base64.b64encode(buffered.getvalue()).decode()

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

if 'total_itens' not in st.session_state: st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state: st.session_state['mapa_resultado'] = None
if 'carrinho' not in st.session_state: st.session_state['carrinho'] = []

st.title("Sistema de Orçamento Automatizado - IA 🚀")

# ==========================================
# 2. UPLOAD E PROCESSAMENTO
# ==========================================
uploaded_file = st.file_uploader("Suba a sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    # Salvamos a imagem em alta resolução no cache para o robô usar depois
    if 'img_high' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                st.session_state['img_high'] = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                doc.close()
            else:
                st.session_state['img_high'] = Image.open(uploaded_file).convert("RGB")
            st.session_state['last_file'] = uploaded_file.name
        except Exception as e:
            st.error(f"Erro no upload: {e}")

    image_high_res = st.session_state['img_high']
    
    # Redimensionamento para o Canvas (LARGURA FIXA PARA PERFORMANCE)
    largura_canvas = 1000
    fator_escala = largura_canvas / float(image_high_res.size[0])
    altura_canvas = int(float(image_high_res.size[1]) * fator_escala)
    image_display = image_high_res.resize((largura_canvas, altura_canvas), Image.Resampling.LANCZOS)

    # Gerar Base64 leve (JPEG) para o fundo
    bg_data = f"data:image/jpeg;base64,{get_image_base64(image_display)}"

    st.markdown("---")
    st.subheader("1. Selecione a área de projeto (Ignore legendas)")

    # Instrução para o usuário
    st.info("💡 Desenhe um retângulo sobre a parte da planta que contém os desenhos.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.1)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=image_display, # Tentamos imagem direta primeiro
        update_streamlit=True,
        height=altura_canvas,
        width=largura_canvas,
        drawing_mode="rect",
        key="canvas_v2",
    )

    # Se a imagem ainda ficar preta, o plano B entra em ação automaticamente via background_color
    # (O Streamlit Drawable Canvas às vezes prefere um ou outro dependendo da versão)

    roi_coords = None
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][-1]
        roi_coords = (
            int(rect["left"] / fator_escala),
            int(rect["top"] / fator_escala),
            int(rect["width"] / fator_escala),
            int(rect["height"] / fator_escala)
        )

    # ==========================================
    # 3. CLIQUE E CONTAGEM
    # ==========================================
    if roi_coords:
        st.markdown("---")
        st.subheader("2. Clique no símbolo que deseja contar")
        
        # Exibimos a imagem novamente para o clique (fora do canvas para evitar conflito)
        coords = streamlit_image_coordinates(image_display, key="click_coord")

        if coords:
            x_real = int(coords["x"] / fator_escala)
            y_real = int(coords["y"] / fator_escala)

            img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            box = 25
            y1, y2 = max(0, y_real-box), min(img_cv.shape[0], y_real+box)
            x1, x2 = max(0, x_real-box), min(img_cv.shape[1], x_real+box)
            
            template_gray = img_gray[y1:y2, x1:x2]
            
            col_target, col_btn = st.columns([1, 2])
            with col_target:
                st.image(cv2.cvtColor(img_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), caption="Alvo", width=80)
            
            with col_btn:
                threshold = st.slider("Precisão", 0.50, 0.95, 0.80)
                if st.button("🔍 Contar Agora"):
                    xr, yr, wr, hr = roi_coords
                    roi_gray = img_gray[yr:yr+hr, xr:xr+wr]
                    
                    pontos = []
                    # Rotações para pegar símbolos em qualquer direção
                    for angulo in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                        t = template_gray if angulo is None else cv2.rotate(template_gray, angulo)
                        res = cv2.matchTemplate(roi_gray, t, cv2.TM_CCOEFF_NORMED)
                        loc = np.where(res >= threshold)
                        
                        raio = max(t.shape)/2
                        for pt in zip(*loc[::-1]):
                            pt_g = (pt[0] + xr, pt[1] + yr)
                            if not any(abs(pt_g[0]-p[0]) < raio and abs(pt_g[1]-p[1]) < raio for p in pontos):
                                pontos.append(pt_g)
                                cv2.rectangle(img_cv, pt_g, (pt_g[0]+t.shape[1], pt_g[1]+t.shape[0]), (0,0,255), 4)
                    
                    st.session_state['total_itens'] = len(pontos)
                    # Reduz imagem final para não dar erro de memória no browser
                    res_viz = cv2.resize(img_cv, (1000, int(img_cv.shape[0]*(1000/img_cv.shape[1]))))
                    st.session_state['mapa_resultado'] = cv2.cvtColor(res_viz, cv2.COLOR_BGR2RGB)
                    st.rerun()

    # ==========================================
    # 4. RESULTADOS E AIRTABLE
    # ==========================================
    if st.session_state['mapa_resultado'] is not None:
        st.image(st.session_state['mapa_resultado'], caption="Localizações encontradas")
        st.success(f"Encontrados: {st.session_state['total_itens']}")
        
        prod_nome = st.text_input("Nome do Produto no Airtable")
        if st.button("Confirmar e Adicionar") and prod_nome:
            # Busca simples no Airtable
            headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"
            dados = requests.get(url, headers=headers).json()
            
            preco = 0.0
            for r in dados.get('records', []):
                if prod_nome.lower() in r['fields'].get('Nome', '').lower():
                    preco = float(r['fields'].get('Preco', 0))
                    break
            
            st.session_state['carrinho'].append({
                "Item": prod_nome,
                "Qtd": st.session_state['total_itens'],
                "Unitário": preco,
                "Total": preco * st.session_state['total_itens']
            })
            st.success("Adicionado ao carrinho!")

    if st.session_state['carrinho']:
        st.markdown("---")
        st.subheader("🛒 Carrinho de Orçamento")
        df = pd.DataFrame(st.session_state['carrinho'])
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Excel (CSV)", csv, "orcamento.csv", "text/csv")
