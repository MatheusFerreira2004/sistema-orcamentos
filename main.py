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
import json
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
if 'total_itens' not in st.session_state:
    st.session_state['total_itens'] = 0
if 'mapa_resultado' not in st.session_state:
    st.session_state['mapa_resultado'] = None
if 'carrinho' not in st.session_state:
    st.session_state['carrinho'] = []
if 'produto_atual' not in st.session_state:
    st.session_state['produto_atual'] = None
# NOVO: biblioteca de símbolos persistente na sessão
if 'biblioteca' not in st.session_state:
    st.session_state['biblioteca'] = {}
# NOVO: modo de processamento de imagem
if 'modo_imagem' not in st.session_state:
    st.session_state['modo_imagem'] = 'binarizado'

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================

def encode_template(img_cv):
    """Converte template BGR para base64 para salvar na biblioteca."""
    _, buffer = cv2.imencode('.png', img_cv)
    return base64.b64encode(buffer).decode()

def decode_template(b64_str):
    """Recupera template BGR a partir de base64."""
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def preprocessar_imagem(img_gray, modo):
    """
    NOVO (Fase 1): Pré-processa a imagem em tons de cinza antes do matching.
    - 'cinza': comportamento original
    - 'binarizado': binarização adaptativa — melhor para plantas com variação de brilho/impressão
    - 'nitidez': realce de bordas — útil para plantas escaneadas com baixa qualidade
    """
    if modo == 'binarizado':
        return cv2.adaptiveThreshold(
            img_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif modo == 'nitidez':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img_gray, -1, kernel)
    else:
        return img_gray

def buscar_produto_airtable(product_search):
    """Busca produto no Airtable e retorna o registro encontrado ou None."""
    table_encoded = urllib.parse.quote(AIRTABLE_TABLE_NAME)
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_encoded}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, response.status_code
    records = response.json().get("records", [])
    search_term = product_search.lower().strip()
    for record in records:
        nome = record.get("fields", {}).get("Nome", "")
        if search_term in nome.lower():
            return record, 200
    return None, 200

def contar_simbolos(img_processada, template_processado, template_original_cv, threshold, raio_seguranca):
    """
    Executa o template matching em 4 rotações e retorna (pontos, img_resultado).
    Agora recebe imagem pré-processada separada da original para o desenho dos retângulos.
    """
    img_result = cv2.cvtColor(img_processada, cv2.COLOR_GRAY2BGR) if len(img_processada.shape) == 2 else img_processada.copy()
    # Usar a imagem colorida original para desenhar os retângulos vermelhos
    img_result_color = template_original_cv.copy() if template_original_cv is not None else img_result

    pontos = []
    rotations = [
        template_processado,
        cv2.rotate(template_processado, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(template_processado, cv2.ROTATE_180),
        cv2.rotate(template_processado, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for rot_template in rotations:
        res = cv2.matchTemplate(img_processada, rot_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        h_tmpl, w_tmpl = rot_template.shape[:2]

        for pt in zip(*loc[::-1]):
            if not any(
                abs(pt[0] - p[0]) < raio_seguranca and abs(pt[1] - p[1]) < raio_seguranca
                for p in pontos
            ):
                pontos.append(pt)
                cv2.rectangle(img_result_color, pt, (pt[0] + w_tmpl, pt[1] + h_tmpl), (0, 0, 255), 4)

        del res, loc
        gc.collect()

    return pontos, img_result_color

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

    # NOVO (Fase 1): Seletor de modo de processamento com explicação
    with st.expander("⚙️ Configurações de Processamento de Imagem", expanded=False):
        st.session_state['modo_imagem'] = st.radio(
            "Modo de pré-processamento:",
            options=['binarizado', 'nitidez', 'cinza'],
            format_func=lambda x: {
                'binarizado': '🟢 Binarizado — Melhor para plantas impressas/escaneadas (recomendado)',
                'nitidez':    '🟡 Realce de bordas — Útil para plantas digitais de baixa qualidade',
                'cinza':      '⚪ Somente cinza — Comportamento original'
            }[x],
            index=0
        )

    uploaded_file = st.file_uploader(
        "Suba sua Planta (PDF, JPG, PNG)",
        type=["pdf", "jpg", "png", "jpeg"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                with st.spinner("Abrindo PDF..."):
                    pdf_bytes = uploaded_file.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    total_paginas = len(doc)

                # NOVO (Fase 1): Seletor de página — suporte multi-page
                pagina_idx = st.selectbox(
                    f"📄 Selecione a página ({total_paginas} páginas no PDF):",
                    options=list(range(total_paginas)),
                    format_func=lambda i: f"Página {i + 1}"
                )

                with st.spinner(f"Extraindo página {pagina_idx + 1}..."):
                    page = doc.load_page(pagina_idx)
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
            # CAPTURA DE FORMA
            # ==========================================
            coords = streamlit_image_coordinates(image_display, key="mapa_clique")

            if coords:
                x_real = int(coords["x"] / fator_escala)
                y_real = int(coords["y"] / fator_escala)

                img_cv = cv2.cvtColor(np.array(image_high_res), cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

                # NOVO (Fase 1): Aplicar pré-processamento escolhido
                img_processada = preprocessar_imagem(img_gray, st.session_state['modo_imagem'])

                st.subheader("2. Ajuste o recorte do seu alvo")
                st.info("🚨 O quadrado abaixo deve conter APENAS o símbolo, sem pegar linhas vizinhas.")

                box_size = st.slider("Tamanho da área de captura", 5, 80, 20)

                y1, y2 = max(0, y_real - box_size), min(img_cv.shape[0], y_real + box_size)
                x1, x2 = max(0, x_real - box_size), min(img_cv.shape[1], x_real + box_size)

                template_color = img_cv[y1:y2, x1:x2]
                # NOVO (Fase 1): template também pré-processado
                template_processado = img_processada[y1:y2, x1:x2]

                col_target, col_config = st.columns([1, 3])

                with col_target:
                    st.image(
                        cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB),
                        caption="Símbolo Capturado",
                        width=150
                    )

                with col_config:
                    threshold = st.slider("Precisão da Forma (0.90 = Idêntico)", 0.50, 0.99, 0.85, 0.01)

                    col_buscar, col_salvar = st.columns(2)

                    with col_buscar:
                        if st.button("🔍 Procurar em Todos os Ângulos na Planta"):
                            with st.spinner("Calculando..."):
                                raio_seguranca = max(template_processado.shape[0], template_processado.shape[1]) / 2
                                pontos, img_result = contar_simbolos(
                                    img_processada,
                                    template_processado,
                                    img_cv,
                                    threshold,
                                    raio_seguranca
                                )

                                fator_reducao = 1500 / float(img_result.shape[1])
                                nova_largura = 1500
                                nova_altura = int(img_result.shape[0] * fator_reducao)
                                img_result_small = cv2.resize(
                                    img_result, (nova_largura, nova_altura),
                                    interpolation=cv2.INTER_AREA
                                )

                                st.session_state['total_itens'] = len(pontos)
                                st.session_state['mapa_resultado'] = cv2.cvtColor(img_result_small, cv2.COLOR_BGR2RGB)
                                st.session_state['produto_atual'] = None

                                del img_cv, img_gray, img_processada, img_result, image_high_res
                                gc.collect()

                    # NOVO (Fase 2): Botão salvar na biblioteca
                    with col_salvar:
                        with st.popover("💾 Salvar na Biblioteca"):
                            st.markdown("**Salvar este símbolo para reutilizar:**")
                            nome_simbolo = st.text_input(
                                "Nome do símbolo",
                                placeholder="ex: Interruptor Baixo h=70cm",
                                key="nome_novo_simbolo"
                            )
                            produto_vinculado = st.text_input(
                                "Produto no Airtable (opcional)",
                                placeholder="ex: Interruptor",
                                key="produto_novo_simbolo"
                            )
                            if st.button("✅ Confirmar salvamento"):
                                if nome_simbolo.strip():
                                    st.session_state['biblioteca'][nome_simbolo] = {
                                        "template_b64": encode_template(template_color),
                                        "produto_airtable": produto_vinculado.strip(),
                                        "threshold": threshold,
                                        "box_size": box_size
                                    }
                                    st.success(f"Símbolo '{nome_simbolo}' salvo!")
                                else:
                                    st.warning("Informe um nome para o símbolo.")

            # ==========================================
            # RESULTADO E INTEGRAÇÃO AIRTABLE
            # ==========================================
            if st.session_state['mapa_resultado'] is not None:
                st.success(f"✅ O robô encontrou {st.session_state['total_itens']} símbolos na prancha!")
                st.image(st.session_state['mapa_resultado'], use_container_width=True)

                st.markdown("---")
                st.subheader("3. Refino e Busca no Banco de Dados 💰")

                col_desc, col_busca = st.columns([1, 2])

                with col_desc:
                    desconto = st.number_input(
                        "Descontar símbolos da legenda:",
                        min_value=0, max_value=20, value=1
                    )
                    total_final = max(0, st.session_state['total_itens'] - desconto)
                    st.info(f"Total real para orçamento: **{total_final} itens**")

                with col_busca:
                    product_search = st.text_input("Nome do produto no Airtable (ex: Interruptor)")

                if st.button("Buscar Preço no Airtable") and product_search:
                    with st.spinner("Consultando banco de dados..."):
                        found, status = buscar_produto_airtable(product_search)

                        if status != 200:
                            st.error(f"Erro ao conectar com Airtable. Código: {status}")
                        elif found:
                            nome_real = found["fields"].get("Nome", "Produto")
                            preco_str = found["fields"].get("Preco", found["fields"].get("Preço", 0))
                            try:
                                preco = float(preco_str)
                            except:
                                preco = 0.0

                            st.session_state['produto_atual'] = {
                                "Produto": nome_real,
                                "Quantidade": total_final,
                                "Preço Unitário (R$)": preco,
                                "Subtotal (R$)": preco * total_final
                            }
                        else:
                            st.warning(f"❌ Produto '{product_search}' não encontrado na tabela.")

                if st.session_state['produto_atual']:
                    prod = st.session_state['produto_atual']
                    st.success(f"Produto localizado: **{prod['Produto']}**")

                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Preço Unitário (R$)",
                        f"{prod['Preço Unitário (R$)']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    )
                    col2.metric("Quantidade Real", prod['Quantidade'])
                    col3.metric(
                        "Total Estimado (R$)",
                        f"{prod['Subtotal (R$)']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🛒 Adicionar item ao Orçamento Geral", type="primary"):
                        st.session_state['carrinho'].append(prod)
                        st.session_state['produto_atual'] = None
                        st.success("Item adicionado! Veja na aba 💰 Orçamento Geral.")
                        st.rerun()

        except Exception as e:
            st.error(f"Ocorreu um erro no processamento: {e}")

# ==========================================
# ABA 2: BIBLIOTECA DE SÍMBOLOS (NOVA - Fase 2)
# ==========================================
with aba_biblioteca:
    st.header("📚 Biblioteca de Símbolos Salvos")
    st.info(
        "Salve os símbolos uma vez e reutilize em qualquer planta. "
        "Ideal para os símbolos padrão da Luz Cor e Design: interruptores, luminárias, perfis LED."
    )

    if len(st.session_state['biblioteca']) == 0:
        st.warning("Nenhum símbolo salvo ainda. Capture um símbolo na aba '📐 Contar Símbolos' e clique em 'Salvar na Biblioteca'.")
    else:
        st.markdown(f"**{len(st.session_state['biblioteca'])} símbolo(s) salvos:**")
        st.markdown("---")

        # Upload de planta para uso via biblioteca
        uploaded_lib = st.file_uploader(
            "Suba a planta para usar com os símbolos da biblioteca:",
            type=["pdf", "jpg", "png", "jpeg"],
            key="upload_biblioteca"
        )

        image_lib = None
        if uploaded_lib:
            if uploaded_lib.name.lower().endswith(".pdf"):
                pdf_bytes_lib = uploaded_lib.read()
                doc_lib = fitz.open(stream=pdf_bytes_lib, filetype="pdf")
                total_pag_lib = len(doc_lib)
                pag_lib = st.selectbox(
                    f"Página ({total_pag_lib} disponíveis):",
                    options=list(range(total_pag_lib)),
                    format_func=lambda i: f"Página {i + 1}",
                    key="pag_lib"
                )
                page_lib = doc_lib.load_page(pag_lib)
                pix_lib = page_lib.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_lib = Image.open(io.BytesIO(pix_lib.tobytes("png"))).convert("RGB")
                del doc_lib, page_lib, pix_lib
                gc.collect()
            else:
                image_lib = Image.open(uploaded_lib).convert("RGB")

        # Listar símbolos da biblioteca
        for nome, dados in list(st.session_state['biblioteca'].items()):
            with st.container():
                col_img, col_info, col_acoes = st.columns([1, 3, 2])

                with col_img:
                    tmpl_cv = decode_template(dados['template_b64'])
                    tmpl_rgb = cv2.cvtColor(tmpl_cv, cv2.COLOR_BGR2RGB)
                    st.image(tmpl_rgb, width=90, caption=nome)

                with col_info:
                    st.markdown(f"**{nome}**")
                    if dados.get('produto_airtable'):
                        st.caption(f"Produto vinculado: `{dados['produto_airtable']}`")
                    else:
                        st.caption("Sem produto vinculado")
                    st.caption(
                        f"Precisão padrão: {dados.get('threshold', 0.85):.2f} | "
                        f"Box: {dados.get('box_size', 20)}px"
                    )

                with col_acoes:
                    # NOVO (Fase 2): Contar direto da biblioteca
                    if image_lib is not None:
                        threshold_lib = st.slider(
                            "Precisão",
                            0.50, 0.99,
                            float(dados.get('threshold', 0.85)),
                            0.01,
                            key=f"thr_{nome}"
                        )
                        if st.button(f"🔍 Contar na planta", key=f"contar_{nome}"):
                            with st.spinner(f"Contando '{nome}'..."):
                                img_cv_lib = cv2.cvtColor(np.array(image_lib), cv2.COLOR_RGB2BGR)
                                img_gray_lib = cv2.cvtColor(img_cv_lib, cv2.COLOR_BGR2GRAY)
                                img_proc_lib = preprocessar_imagem(img_gray_lib, st.session_state['modo_imagem'])

                                tmpl_gray = cv2.cvtColor(decode_template(dados['template_b64']), cv2.COLOR_BGR2GRAY)
                                tmpl_proc = preprocessar_imagem(tmpl_gray, st.session_state['modo_imagem'])

                                raio = max(tmpl_proc.shape[0], tmpl_proc.shape[1]) / 2
                                pontos_lib, img_res_lib = contar_simbolos(
                                    img_proc_lib, tmpl_proc, img_cv_lib,
                                    threshold_lib, raio
                                )

                                fator_r = 1500 / float(img_res_lib.shape[1])
                                img_res_small = cv2.resize(
                                    img_res_lib,
                                    (1500, int(img_res_lib.shape[0] * fator_r)),
                                    interpolation=cv2.INTER_AREA
                                )
                                resultado_rgb = cv2.cvtColor(img_res_small, cv2.COLOR_BGR2RGB)

                                total_lib = len(pontos_lib)
                                st.success(f"✅ {total_lib} ocorrências de '{nome}'")
                                st.image(resultado_rgb, use_container_width=True)

                                # Se tem produto vinculado, oferecer adicionar ao carrinho direto
                                if dados.get('produto_airtable') and total_lib > 0:
                                    desconto_lib = st.number_input(
                                        "Descontar legenda:", min_value=0, max_value=10,
                                        value=1, key=f"desc_{nome}"
                                    )
                                    total_real_lib = max(0, total_lib - desconto_lib)

                                    if st.button(
                                        f"💰 Buscar preço e adicionar ao orçamento",
                                        key=f"add_{nome}", type="primary"
                                    ):
                                        found_lib, status_lib = buscar_produto_airtable(dados['produto_airtable'])
                                        if found_lib:
                                            preco_lib = 0.0
                                            try:
                                                preco_lib = float(
                                                    found_lib["fields"].get(
                                                        "Preco",
                                                        found_lib["fields"].get("Preço", 0)
                                                    )
                                                )
                                            except:
                                                pass
                                            st.session_state['carrinho'].append({
                                                "Produto": found_lib["fields"].get("Nome", dados['produto_airtable']),
                                                "Quantidade": total_real_lib,
                                                "Preço Unitário (R$)": preco_lib,
                                                "Subtotal (R$)": preco_lib * total_real_lib
                                            })
                                            st.success("Item adicionado ao orçamento!")
                                            st.rerun()
                                        else:
                                            st.warning("Produto não encontrado no Airtable.")

                    # Botão excluir símbolo da biblioteca
                    if st.button("🗑️ Remover", key=f"del_{nome}"):
                        del st.session_state['biblioteca'][nome]
                        st.rerun()

                st.markdown("---")

# ==========================================
# ABA 3: ORÇAMENTO GERAL
# ==========================================
with aba_orcamento:
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
        st.subheader(
            f"💰 CUSTO TOTAL: R$ {total_geral:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )

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

        # NOVO: Botão remover item individual do carrinho
        st.markdown("---")
        st.subheader("Remover item específico:")
        for i, item in enumerate(st.session_state['carrinho']):
            col_item, col_rem = st.columns([4, 1])
            with col_item:
                st.write(f"{item['Produto']} — {item['Quantidade']} un. — R$ {item['Subtotal (R$)']:.2f}")
            with col_rem:
                if st.button("✕", key=f"rem_{i}"):
                    st.session_state['carrinho'].pop(i)
                    st.rerun()
    else:
        st.info("O seu carrinho está vazio. Use as abas anteriores para adicionar itens.")
