import streamlit as st
from streamlit_stl import stl_from_text
import tempfile
import time
import os 
import base64
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plyfile import PlyData


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

def load_ply_file(ply_path):
    try:
        plydata = PlyData.read(ply_path)
        
        vertices = plydata['vertex']
        x = vertices['x']
        y = vertices['y'] 
        z = vertices['z']
        
        try:
            r = vertices['red'] / 255.0
            g = vertices['green'] / 255.0
            b = vertices['blue'] / 255.0
            colors = np.column_stack([r, g, b])
        except:
            colors = z
        
        return x, y, z, colors
    except Exception as e:
        st.error(f"Ошибка чтения PLY: {e}")
        return None, None, None, None

def create_viewer_html(ply_file_path=None):
    
    with open('C:\\projects\\python\\PavFGS\\src\\interface\\splat\\main.js', 'r', encoding='utf-8') as f:
        main_js = f.read()
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Gaussian Splatting</title>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #000;
            }}
            #canvas {{
                width: 100vw;
                height: 100vh;
                display: block;
            }}
            #ui {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: white;
                font-family: Arial, sans-serif;
                background: rgba(0,0,0,0.5);
                padding: 10px;
                border-radius: 5px;
            }}
            #spinner {{
                display: none;
                color: #fff;
            }}
            #progress {{
                width: 200px;
                height: 4px;
                background: #333;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        <div id="ui">
            <div id="fps">0 fps</div>
            <div id="camid"></div>
            <div id="progress"></div>
            <div id="spinner">Loading...</div>
            <div id="message"></div>
        </div>
        
        <script>
            {main_js}
        </script>
    </body>
    </html>
    """
    
    return html_template


def main():
    
    uploaded_file = st.file_uploader("Загрузите PLY файл", type=['ply'])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            ply_path = tmp_file.name
        
        st.success(f"Файл {uploaded_file.name} загружен!")
        
        html_content = create_viewer_html()
        
        col1, col2 = st.columns(2)
        with col1:
            st.components.v1.html(html_content, height=600)
        with col2:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            x, y, z, colors = load_ply_file(tmp_path)
            
            if x is not None:
                fig = go.Figure(data=[go.Scatter3d(
                    x=x,
                    y=y, 
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors,
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )])

                fig.update_layout(
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='data'
                    ),
                    width=800,
                    height=600,
                    title="3D Point Cloud"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Информация:** {len(x)} точек")
        
        with st.expander("ℹ️ Управление камерой"):
            st.markdown("""
            ### Управление:
            - **ЛКМ + перемещение**: Вращение камеры
            - **ПКМ + перемещение**: Перемещение камеры  
            - **Колесо мыши**: Приближение/отдаление
            - **WASD**: Перемещение камеры
            - **Цифры 0-9**: Переключение между предустановленными камерами
            - **+/-**: Следующая/предыдущая камера
            - **P**: Автоматический пролет (carousel)
            - **V**: Сохранить текущий вид в URL
            """)
        
        os.unlink(ply_path)

if __name__ == "__main__":
    main()

# GLB_PATH = r"C:\projects\python\PavFGS\src\interface\creepy_scarecrow__horror_stylized.glb"

# if not os.path.exists(GLB_PATH):
#     st.error(f"❌ Файл не найден: {GLB_PATH}")
#     st.stop()

# with open(GLB_PATH, "rb") as f:
#     glb_data = f.read()
#     glb_base64 = base64.b64encode(glb_data).decode("utf-8")

# html_code = f"""
# <!DOCTYPE html>
# <html>
# <head>
#     <script type="module" src="https://unpkg.com/@google/model-viewer@^2.0.0/dist/model-viewer.min.js"></script>
# </head>
# <body>
#     <model-viewer 
#         src="data:model/gltf-binary;base64,{glb_base64}"
#         alt="3D Model"
#         auto-rotate
#         camera-controls
#         style="width: 100%; height: 600px; background-color: #1e1e1e;"
#         ar
#         animation-name="All"
#         autoplay
#     >
#         <div class="progress-bar" slot="progress-bar">
#             <div class="update-bar"></div>
#         </div>
#     </model-viewer>
    
#     <script>
#         console.log('Model viewer loaded');
#     </script>
# </body>
# </html>
# """
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.components.v1.html(html_code, height=620)
# with col2:
#     st.components.v1.html(html_code, height=620)
# with col3:
#     st.components.v1.html(html_code, height=620)
# with col4:
#     st.components.v1.html(html_code, height=620)