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

from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONFIGURAÇÃO INICIAL E MEMÓRIA
# ==========================================
st.set_page_config(layout="wide", page_title="Orçamento IA - Luz Cor e Design")

def get_clean_secret(key_name):
    try:
        return st.secrets[key_name].strip().replace('"', '').replace("'", "")
    except:
        return None

AIRTABLE_API_KEY = get_clean_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_clean_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_clean_secret("AIRTABLE_TABLE_NAME")

st.title("Sistema de Orçamento Automatizado - Luz Cor e Design 🚀")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
    st.error("⚠️ Erro: Chaves do Airtable não encontradas nos Secrets.")
    st.stop()

# Inicialização das Memórias
for key in ['total_itens', 'mapa_resultado', 'carrinho', 'produto_atual', 'biblioteca']:
    if key not in st.session_state:
        st.session_state[key] = {} if key == 'biblioteca' else ([] if key == 'carrinho' else None)
        
if 'total_itens' not in st.session_state or st.session_state['total_itens'] is None:
    st.session_state['total_itens'] = 0

if 'modo_imagem' not in st.session_state:
    st.session_state['modo_imagem'] = 'binarizado'

# ==========================================
# 2. FUNÇÕES AUXILIARES E DE VISÃO COMPUTACIONAL
# ==========================================
def encode_template(img_cv):
    _, buffer = cv2.imencode('.png', img_cv)
    return base64.b64encode(buffer).decode()

def decode_template(b64_str):
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def preprocessar_imagem(img_gray, modo):
    """Aplica CLAHE e o filtro escolhido pelo usuário"""
    # 🔥 CLAHE: Equaliza sombras e melhora o contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    if modo == 'binarizado':
        return cv2.adaptiveThreshold(img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif modo == 'nitidez':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img_clahe, -1, kernel)
    else:
        return img_clahe

def auto_canny(image, sigma=0.33):
    """Calcula os limites do Canny dinamicamente baseado na iluminação da planta"""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def buscar_produto_airtable(product_search):
    table_encoded = urllib.parse.quote(AIRTABLE_TABLE_NAME)
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_encoded}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, response.status_code
    
    search_term = product_search.lower().strip()
    for record in response.json().get("records", []):
        if search_term in record.get("fields", {}).get("Nome", "").lower():
            return record, 200
    return None, 200

def contar_simbolos(img_processada, template_processado, template_original_cv, threshold, raio_seguranca):
    """Motor central de busca com Auto-Canny e Validação de Região"""
    # Transforma as imagens pré-processadas em bordas (Canny)
    edges_global = auto_canny(img_processada)
    template_edges = auto_canny(template_processado)

    img_result_color = template_original_cv.copy() if template_original_cv is not None else cv2.cvtColor(img_processada, cv2.COLOR_GRAY2BGR)
    pontos = []
    
    rotations = [
        template_edges,
        cv2.rotate(template_edges, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(template_edges, cv2.ROTATE_180),
        cv2.rotate(template_edges, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for rot_template in rotations:
        res = cv2.matchTemplate(edges_global, rot_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        # 🔥 BLINDAGEM ANTI-CRASH (Trava de CPU) 🔥
        if len(loc[0]) > 3000:
            st.error(f"⚠️ Muitos candidatos encontrados ({len(loc[0])}). O robô abortou para não travar o site! Aumente a Precisão para refinar a busca.")
            break

        h_t, w_t = rot_template.shape[:2]

        for pt in zip(*loc[::-1]):
            x, y = pt
            # 🔥 Score Local (Validação de Região)
            y_end, x_end = min(y + h_t, res.shape[0]), min(x + w_t, res.shape[1])
            region_score = np.mean(res[y:y_end, x:x_end])

            # Só aceita se a região toda fizer sentido (Evita contar linhas cruzadas isoladas)
            if region_score >= (threshold * 0.85):
                if not any(np.linalg.norm(np.array(pt)-np.array(p)) < raio_seguranca for p in pontos):
                    pontos.append(pt)
                    cv2.rectangle(img_result_color, pt, (x + w_t, y + h_t), (0, 0, 255), 4)

        del res, loc
        gc.collect()

    return pontos, img_result_color

def otimizar_imagem_memoria(img_pil):
    """🔥 BLINDAGEM ANTI-CRASH (Trava de RAM) 🔥"""
    MAX_WIDTH = 3000
    if img_pil.size[0] > MAX_WIDTH:
        fator = MAX_WIDTH / float(img_pil.size[0])
        altura = int(float(img_pil.size[1]) * fator)
        st.toast("A planta era muito pesada. Foi otimizada para evitar travamentos!")
        return img_pil.resize((MAX_WIDTH, altura), Image.Resampling.LANCZOS)
    return img_pil

# ==========================================
# 3. ABAS PRINCIPAIS
# ==========================================
aba_contagem, aba_biblioteca, aba_orcamento = st.tabs([
    "📐 Contar Símbolos",
    "📚 Biblioteca de Símbolos",
    "💰 Orçamento Geral"
])

# ==========================================
# ABA 1: CONTAGEM
# ==========================================
with aba_contagem:
    with st.expander("⚙️ Configurações de Processamento de Imagem", expanded=False):
        st.session_state['modo_imagem'] = st.radio(
            "Modo de pré-processamento:",
            options=['binarizado', 'nitidez', 'cinza'],
            format_func=lambda x: {
                'binarizado': '🟢 Binarizado — Melhor para plantas impressas/escaneadas sujas',
                'nitidez':    '🟡 Realce de bordas — Útil para plantas digitais de baixa qualidade',
                'cinza':      '⚪ Somente cinza (com CLAHE) — Melhor para plantas digitais limpas (Recomendado)'
            }[x],
            index=2
        )

    uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG, PNG)", type=["pdf", "jpg", "png", "jpeg"], key="up_main")

    if uploaded_file:
        try:
            with st.spinner("Carregando e otimizando..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    pdf_bytes = uploaded_file.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    total_paginas = len(doc)

                    pagina_idx = st.selectbox(f"📄 Selecione a página ({total_paginas} páginas no PDF):", options=list(range(total_paginas)), format_func=lambda i: f"Página {i + 1}")
                    page = doc.load_page(pagina_idx)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image_high_res = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    del doc, page, pix
                    gc.collect()
                else:
                    image_high_res = Image.open(uploaded_file).convert("RGB")

                # Trava de RAM
                image_high_res = otimizar_imagem_memoria(image_high_res)

            st.markdown("---")
            st.subheader("1. Clique BEM NO CENTRO do símbolo que deseja contar:")

            largura_tela = 1000
            fator_escala = largura_tela / float(image_high_res.size[0])
            altura_tela = int(float(image_high_res.size[1]) * fator_escala)
            image_display = image_high_res.resize((largura_tela, altura_tela), Image.Resampling.LANCZOS)

            coords = streamlit_image_coordinates(image_display, key="mapa_clique")

            if coords:
                x_real, y_real = int(coords["x"] / fator_escala), int(coords["y"] / fator_escala)

                img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                img_processada = preprocessar_imagem(img_gray, st.session_state['modo_imagem'])

                st.subheader("2. Ajuste o recorte e a precisão")
                box_size = st.slider("Tamanho do Recorte", 5, 80, 25)

                y1, y2 = max(0, y_real - box_size), min(img_cv.shape[0], y_real + box_size)
                x1, x2 = max(0, x_real - box_size), min(img_cv.shape[1], x_real + box_size)

                template_color = img_cv[y1:y2, x1:x2]
                template_processado = img_processada[y1:y2, x1:x2]

                col_target, col_config = st.columns([1, 3])
                with col_target:
                    st.image(cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB), caption="Alvo", width=120)

                with col_config:
                    threshold = st.slider("Precisão da Busca (0.75 a 0.85 recomendado)", 0.50, 0.99, 0.80, 0.01)
                    col_buscar, col_salvar = st.columns(2)

                    with col_buscar:
                        if st.button("🔍 Iniciar Varredura Inteligente", type="primary"):
                            with st.spinner("Analisando geometria..."):
                                raio_seguranca = max(template_processado.shape[0], template_processado.shape[1]) / 2
                                pontos, img_result = contar_simbolos(img_processada, template_processado, img_cv, threshold, raio_seguranca)

                                fator_reducao = 1500 / float(img_result.shape[1])
                                img_result_small = cv2.resize(img_result, (1500, int(img_result.shape[0] * fator_reducao)), interpolation=cv2.INTER_AREA)

                                st.session_state['total_itens'] = len(pontos)
                                st.session_state['mapa_resultado'] = cv2.cvtColor(img_result_small, cv2.COLOR_BGR2RGB)
                                st.session_state['produto_atual'] = None

                                del img_cv, img_gray, img_processada, img_result, image_high_res
                                gc.collect()

                    with col_salvar:
                        with st.popover("💾 Salvar na Biblioteca"):
                            nome_simbolo = st.text_input("Nome do símbolo", placeholder="ex: Spot Direcionável")
                            produto_vinculado = st.text_input("Produto Airtable", placeholder="ex: Embutido")
                            if st.button("✅ Confirmar"):
                                if nome_simbolo.strip():
                                    st.session_state['biblioteca'][nome_simbolo] = {
                                        "template_b64": encode_template(template_color),
                                        "produto_airtable": produto_vinculado.strip(),
                                        "threshold": threshold,
                                        "box_size": box_size
                                    }
                                    st.success("Símbolo guardado!")
                                else:
                                    st.warning("De um nome ao símbolo.")

            if st.session_state['mapa_resultado'] is not None:
                st.success(f"✅ Encontrados {st.session_state['total_itens']} itens!")
                st.image(st.session_state['mapa_resultado'], use_container_width=True)

                st.markdown("---")
                col_desc, col_busca = st.columns([1, 2])
                with col_desc:
                    desconto = st.number_input("Descontar da legenda:", 0, 20, 1)
                    total_final = max(0, st.session_state['total_itens'] - desconto)
                with col_busca:
                    product_search = st.text_input("Buscar Airtable (ex: Fita LED)")

                if st.button("Consultar Preço") and product_search:
                    with st.spinner("Consultando banco..."):
                        found, status = buscar_produto_airtable(product_search)
                        if found:
                            preco = float(found["fields"].get("Preco", found["fields"].get("Preço", 0)))
                            st.session_state['produto_atual'] = {
                                "Produto": found["fields"].get("Nome", "Produto"),
                                "Quantidade": total_final,
                                "Preço Unitário (R$)": preco,
                                "Subtotal (R$)": preco * total_final
                            }
                        else:
                            st.error("Produto não encontrado.")

                if st.session_state['produto_atual']:
                    p = st.session_state['produto_atual']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Unitário", f"R$ {p['Preço Unitário (R$)']:.2f}")
                    col2.metric("Qtd Real", p['Quantidade'])
                    col3.metric("Subtotal", f"R$ {p['Subtotal (R$)']:.2f}")
                    
                    if st.button("🛒 Adicionar ao Orçamento", type="primary"):
                        st.session_state['carrinho'].append(p)
                        st.session_state['produto_atual'] = None
                        st.success("Adicionado! Veja a aba de Orçamento.")
                        st.rerun()

        except Exception as e:
            st.error(f"Erro: {e}")

# ==========================================
# ABA 2: BIBLIOTECA
# ==========================================
with aba_biblioteca:
    st.header("📚 Biblioteca de Símbolos")
    if not st.session_state['biblioteca']:
        st.warning("Salve símbolos na aba de contagem para usá-los aqui.")
    else:
        uploaded_lib = st.file_uploader("Suba a planta para aplicar os símbolos:", type=["pdf", "jpg", "png"], key="up_lib")
        image_lib = None

        if uploaded_lib:
            with st.spinner("Carregando planta..."):
                if uploaded_lib.name.lower().endswith(".pdf"):
                    doc_lib = fitz.open(stream=uploaded_lib.read(), filetype="pdf")
                    pag_lib = st.selectbox("Página:", options=list(range(len(doc_lib))), format_func=lambda i: f"Página {i + 1}")
                    pix_lib = doc_lib.load_page(pag_lib).get_pixmap(matrix=fitz.Matrix(2, 2))
                    image_lib = Image.open(io.BytesIO(pix_lib.tobytes("png"))).convert("RGB")
                    del doc_lib, pix_lib
                    gc.collect()
                else:
                    image_lib = Image.open(uploaded_lib).convert("RGB")
                
                # Trava de RAM também na Biblioteca
                image_lib = otimizar_imagem_memoria(image_lib)

        st.markdown("---")
        for nome, dados in list(st.session_state['biblioteca'].items()):
            col_img, col_info, col_acoes = st.columns([1, 3, 2])
            with col_img:
                st.image(cv2.cvtColor(decode_template(dados['template_b64']), cv2.COLOR_BGR2RGB), width=90)
            with col_info:
                st.markdown(f"**{nome}**")
                st.caption(f"Produto: {dados.get('produto_airtable', 'Nenhum')}")
            with col_acoes:
                if image_lib:
                    thr_lib = st.slider("Precisão", 0.50, 0.99, float(dados.get('threshold', 0.80)), key=f"thr_{nome}")
                    if st.button(f"🔍 Contar", key=f"btn_{nome}"):
                        with st.spinner(f"Analisando '{nome}'..."):
                            img_cv_lib = cv2.cvtColor(np.array(image_lib), cv2.COLOR_RGB2BGR)
                            img_gray_lib = cv2.cvtColor(img_cv_lib, cv2.COLOR_BGR2GRAY)
                            img_proc_lib = preprocessar_imagem(img_gray_lib, st.session_state['modo_imagem'])
                            
                            tmpl_gray = cv2.cvtColor(decode_template(dados['template_b64']), cv2.COLOR_BGR2GRAY)
                            tmpl_proc = preprocessar_imagem(tmpl_gray, st.session_state['modo_imagem'])

                            pts_lib, res_lib = contar_simbolos(img_proc_lib, tmpl_proc, img_cv_lib, thr_lib, max(tmpl_proc.shape)/2)
                            
                            st.image(cv2.cvtColor(cv2.resize(res_lib, (1500, int(res_lib.shape[0]*(1500/res_lib.shape[1])))), cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.success(f"✅ {len(pts_lib)} encontrados.")
                            
                            if dados.get('produto_airtable') and len(pts_lib) > 0:
                                desc = st.number_input("Desconto:", 0, 10, 1, key=f"d_{nome}")
                                if st.button("🛒 Adicionar ao Orçamento", key=f"add_{nome}"):
                                    fnd, _ = buscar_produto_airtable(dados['produto_airtable'])
                                    if fnd:
                                        prc = float(fnd["fields"].get("Preco", 0))
                                        st.session_state['carrinho'].append({
                                            "Produto": fnd["fields"].get("Nome"),
                                            "Quantidade": max(0, len(pts_lib) - desc),
                                            "Preço Unitário (R$)": prc,
                                            "Subtotal (R$)": prc * max(0, len(pts_lib) - desc)
                                        })
                                        st.success("Adicionado!")
                                        st.rerun()

                if st.button("🗑️ Apagar", key=f"del_{nome}"):
                    del st.session_state['biblioteca'][nome]
                    st.rerun()

# ==========================================
# ABA 3: ORÇAMENTO GERAL
# ==========================================
with aba_orcamento:
    st.header("📋 Orçamento do Projeto")
    if st.session_state['carrinho']:
        df = pd.DataFrame(st.session_state['carrinho'])
        st.dataframe(df.style.format({"Preço Unitário (R$)": "{:.2f}", "Subtotal (R$)": "{:.2f}"}), use_container_width=True)
        
        st.subheader(f"💰 CUSTO TOTAL: R$ {df['Subtotal (R$)'].sum():,.2f}")
        
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, index=False)
        st.download_button("📊 Baixar Excel", buf.getvalue(), "orcamento.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        st.markdown("---")
        for i, item in enumerate(st.session_state['carrinho']):
            c1, c2 = st.columns([4, 1])
            c1.write(f"{item['Produto']} — {item['Quantidade']} un. — R$ {item['Subtotal (R$)']:.2f}")
            if c2.button("✕", key=f"rem_{i}"):
                st.session_state['carrinho'].pop(i)
                st.rerun()
                
        if st.button("🗑️ Limpar Tudo", type="primary"):
            st.session_state['carrinho'] = []
            st.rerun()
    else:
        st.info("O carrinho está vazio.")
