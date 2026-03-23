import fitz  # PyMuPDF
from PIL import Image
import io

# Configuração do Upload para aceitar PDF e Imagens
uploaded_file = st.file_uploader("Suba sua Planta (PDF, JPG ou PNG)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    # LÓGICA PARA TRATAR PDF
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        # Se o PDF tiver mais de uma página, permite escolher
        if len(doc) > 1:
            page_num = st.number_input(f"Este PDF tem {len(doc)} páginas. Qual deseja usar?", min_value=1, max_value=len(doc), value=1) - 1
        else:
            page_num = 0
            
        page = doc.load_page(page_num)
        
        # AQUI ESTÁ O SEGREDO: Renderizamos em 300 DPI (matrix 4x4) para não perder qualidade
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4)) 
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
    
    # LÓGICA PARA IMAGENS NORMAIS
    else:
        image = Image.open(uploaded_file)

    # A partir daqui, o código segue normal usando a variável 'image'
    st.write("Planta carregada com sucesso!")
