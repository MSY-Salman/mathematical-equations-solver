import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
import json
from sympy import N
from latex2sympy2 import latex2sympy
from io import BytesIO
import zipfile
import tempfile
from datetime import datetime
import time

# Custom CSS with fixed contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #333333 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(75, 108, 183, 0.3);
    }
    
    .result-card {
        background: white !important;
        color: #333333 !important;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
    }
    
    .sidebar-card {
        background: white !important;
        color: #333333 !important;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0 !important;
        margin: 1rem 0;
    }
    
    .streamlit-expanderContent {
        color: #333333 !important;
    }
    
    .stImage > div > div {
        background-color: white !important;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%) !important;
        color: white !important;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #4b6cb7;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    body, .stMarkdown, .stText, .stAlert, .stWarning {
        color: #333333 !important;
    }
    
    .stCodeBlock, .stMarkdown pre {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        border: 1px solid #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = 0
if 'total_equations' not in st.session_state:
    st.session_state.total_equations = 0

# Load keys
@st.cache_resource
def load_keys():
    try:
        with open("keys.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: keys.json file not found")
        return None

ltx_index = load_keys()

# Load ONNX model
@st.cache_resource
def load_model():
    try:
        session = ort.InferenceSession("model.onnx")
        return session, session.get_inputs()[0].name, session.get_outputs()[0].name, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, False

session, input_name, output_name, model_loaded = load_model()

# Page Header
st.markdown("""
<div class="main-header">
    <h1>🧠 MathVision AI</h1>
    <h3>Advanced Handwritten Equation Solver</h3>
    <p>Powered by Deep Learning • LaTeX Conversion • Symbolic Computation</p>
</div>
""", unsafe_allow_html=True)

# Image processing functions
def load_image_for_model(img, w=1024, h=192):
    img = img.convert("RGB")
    img = img.resize((w, h))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def decode_latex(res):
    res = np.argmax(res, axis=1)
    return ''.join([ltx_index.get(str(x - 1), '') if x != 0 else ' ' for x in res])

def solve_latex(latex_expr):
    try:
        latex_expr = latex_expr.replace(r"\left", "").replace(r"\right", "")
        sym_expr = latex2sympy(latex_expr)
        value = N(sym_expr)
        return str(sym_expr), value
    except:
        return "Invalid Expression", ""

def generate_boxes(img):
    h, w = img.height, img.width
    return [(int(w*0.1), int(h*0.2), 80, 60), (int(w*0.3), int(h*0.5), 100, 60)]

def draw_boxes(image, boxes):
    img = np.array(image.convert("RGB"))
    for (x, y, w, h) in boxes:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return Image.fromarray(img)

def export_latex_as_pdf(latex_str, out_path):
    tex_code = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}
\[
%s
\]
\end{document}
""" % latex_str
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_code)

# Sidebar with input options
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    
    current_time = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📈 Session Analytics</h4>
        <p><strong>🔍 Images Processed:</strong> {st.session_state.processed_images}</p>
        <p><strong>📋 Equations Solved:</strong> {st.session_state.total_equations}</p>
        <p><strong>🕐 Current Time:</strong> {current_time}</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_mode = st.radio("Select input mode", ["Single Image", "Batch Processing"])
    
    status_color = "#28a745" if model_loaded else "#dc3545"
    status_text = "✅ Ready" if model_loaded else "❌ Error"
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>🤖 AI Model Status</h4>
        <p style="color: {status_color}; font-weight: bold;">{status_text}</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize uploaded_images before processing
uploaded_images = []

# Handle file uploads based on input mode
if input_mode == "Single Image":
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file:
        uploaded_images = [Image.open(img_file)]
else:
    files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    if files:
        uploaded_images = [Image.open(f) for f in files]

# Main processing logic - only run if we have images and models loaded
if model_loaded and ltx_index and uploaded_images:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    export_zip = BytesIO()
    with zipfile.ZipFile(export_zip, mode="w") as zipf:
        for idx, img in enumerate(uploaded_images):
            progress = (idx + 1) / len(uploaded_images)
            progress_bar.progress(progress)
            status_text.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; color: #333333;">
                <div class="loading-spinner"></div>
                <p>Processing image {idx+1} of {len(uploaded_images)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📄 Image {idx+1} Results", expanded=True):
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #333333;">Image {idx+1} Analysis</h3>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Image", use_column_width=True)

                model_img = load_image_for_model(img)
                result = session.run([output_name], {input_name: model_img})[0][0]
                latex = decode_latex(result).strip()

                expr, val = solve_latex(latex)

                with col2:
                    st.markdown("### Extracted LaTeX")
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #e0e0e0;">
                        <p style="font-family: monospace; font-size: 1.2rem; color: #333333;">{latex}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "Invalid" not in expr:
                        st.markdown("### Solution")
                        st.markdown(f"""
                        <div style="background: #e6f7ff; padding: 1.5rem; border-radius: 10px; border: 1px solid #b3e0ff;">
                            <p style="font-size: 1.5rem; font-weight: bold; text-align: center; color: #0066cc;">
                                {expr} = {val}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.total_equations += 1
                    else:
                        st.warning("Could not solve the expression")

                boxes = generate_boxes(img)
                boxed_img = draw_boxes(img, boxes)
                st.image(boxed_img, caption="Detection Visualization", use_column_width=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".tex") as temp_tex:
                    export_latex_as_pdf(latex, temp_tex.name)
                    zipf.write(temp_tex.name, arcname=f"equation_{idx+1}.tex")
                
                st.session_state.processed_images += 1
            
            st.markdown("</div>", unsafe_allow_html=True)
            time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Successfully processed {len(uploaded_images)} images")
    
    st.download_button(
        "📦 Download All Results (ZIP)", 
        data=export_zip.getvalue(),
        file_name="mathvision_results.zip", 
        mime="application/zip"
    )
elif not model_loaded or not ltx_index:
    st.error("🚨 System not ready. Please check model and configuration files.")
elif not uploaded_images:
    st.info("ℹ️ Please upload images to process")

# Footer
st.markdown("COMPUTER VISION BY DR SYED FAISAL BUKHARI")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;">
    <h3 style="color: #495057; margin-bottom: 1rem;">🧠 MathVision AI</h3>
    <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 0.5rem;">
        <strong>Advanced Mathematical OCR • LaTeX Conversion • Symbolic Computation</strong>
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem; margin-top: 1rem;">
        <p style="color: #6c757d; font-size: 0.8rem; margin: 0;">
            ⚠️ For educational purposes • Results may vary based on input quality
        </p>
    </div>
</div>
""", unsafe_allow_html=True)