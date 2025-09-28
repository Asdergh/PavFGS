import streamlit as st
from streamlit_stl import stl_from_text
import time
import os 
import base64
from pathlib import Path

st.set_page_config(layout="wide", page_title="PavFGS")

st.html("""
    <div style='
        background: #1a1a1a;
        padding: 50px 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #333;
        margin: 20px 0;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(255,255,255,0.05) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(255,255,255,0.05) 0%, transparent 20%);
    '>
        <div style='
            font-size: 60px;
            margin-bottom: 10px;
            filter: drop-shadow(0 0 10px #00d4ff);
        '>🧬</div>
        
        <h1 style='
            color: #00d4ff;
            font-size: 56px;
            font-weight: 800;
            margin: 0;
            font-family: "Arial", sans-serif;
            text-transform: uppercase;
            letter-spacing: 4px;
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0,212,255,0.5);
        '>PAVFGS</h1>
        
        <p style='
            color: #888;
            font-size: 16px;
            margin: 15px 0 0 0;
            font-weight: 300;
            letter-spacing: 3px;
            text-transform: uppercase;
        '>Next Generation 3D Modeling Platform</p>
        
        <div style='
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 10px;
        '>
            <span style='color: #00d4ff; font-size: 14px;'>⚡ Real-time</span>
            <span style='color: #00d4ff; font-size: 14px;'>🎯 Precision</span>
            <span style='color: #00d4ff; font-size: 14px;'>🚀 Performance</span>
        </div>
    </div>
""")


GLB_PATH = r"C:\projects\python\PavFGS\src\interface\creepy_scarecrow__horror_stylized.glb"

if not os.path.exists(GLB_PATH):
    st.error(f"❌ Файл не найден: {GLB_PATH}")
    st.stop()

with open(GLB_PATH, "rb") as f:
    glb_data = f.read()
    glb_base64 = base64.b64encode(glb_data).decode("utf-8")

html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <script type="module" src="https://unpkg.com/@google/model-viewer@^2.0.0/dist/model-viewer.min.js"></script>
</head>
<body>
    <model-viewer 
        src="data:model/gltf-binary;base64,{glb_base64}"
        alt="3D Model"
        auto-rotate
        camera-controls
        style="width: 100%; height: 600px; background-color: #1e1e1e;"
        ar
        animation-name="All"
        autoplay
    >
        <div class="progress-bar" slot="progress-bar">
            <div class="update-bar"></div>
        </div>
    </model-viewer>
    
    <script>
        console.log('Model viewer loaded');
    </script>
</body>
</html>
"""
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.components.v1.html(html_code, height=620)
with col2:
    st.components.v1.html(html_code, height=620)
with col3:
    st.components.v1.html(html_code, height=620)
with col4:
    st.components.v1.html(html_code, height=620)