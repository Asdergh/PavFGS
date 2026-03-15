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

# def load_ply_file(ply_path):
#     try:
#         plydata = PlyData.read(ply_path)

#         vertices = plydata['vertex']
#         x = vertices['x']
#         y = vertices['y']
#         z = vertices['z']

#         try:
#             r = vertices['red'] / 255.0
#             g = vertices['green'] / 255.0
#             b = vertices['blue'] / 255.0
#             colors = np.column_stack([r, g, b])
#         except:
#             colors = z

#         return x, y, z, colors
#     except Exception as e:
#         st.error(f"Ошибка чтения PLY: {e}")
#         return None, None, None, None

# def create_viewer_html(ply_file_path=None):

#     with open('C:\\projects\\python\\PavFGS\\src\\interface\\splat\\main.js', 'r', encoding='utf-8') as f:
#         main_js = f.read()

#     html_template = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>3D Gaussian Splatting</title>
#         <style>
#             body, html {{
#                 margin: 0;
#                 padding: 0;
#                 overflow: hidden;
#                 background: #000;
#             }}
#             #canvas {{
#                 width: 100vw;
#                 height: 100vh;
#                 display: block;
#             }}
#             #ui {{
#                 position: absolute;
#                 top: 10px;
#                 left: 10px;
#                 color: white;
#                 font-family: Arial, sans-serif;
#                 background: rgba(0,0,0,0.5);
#                 padding: 10px;
#                 border-radius: 5px;
#             }}
#             #spinner {{
#                 display: none;
#                 color: #fff;
#             }}
#             #progress {{
#                 width: 200px;
#                 height: 4px;
#                 background: #333;
#                 margin: 5px 0;
#             }}
#         </style>
#     </head>
#     <body>
#         <canvas id="canvas"></canvas>
#         <div id="ui">
#             <div id="fps">0 fps</div>
#             <div id="camid"></div>
#             <div id="progress"></div>
#             <div id="spinner">Loading...</div>
#             <div id="message"></div>
#         </div>

#         <script>
#             {main_js}
#         </script>
#     </body>
#     </html>
#     """

#     return html_template


# def main():

#     uploaded_file = st.file_uploader("Загрузите PLY файл", type=['ply'])

#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             ply_path = tmp_file.name

#         st.success(f"Файл {uploaded_file.name} загружен!")

#         html_content = create_viewer_html()

#         col1, col2 = st.columns(2)
#         with col1:
#             st.components.v1.html(html_content, height=600)
#         with col2:
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_path = tmp_file.name

#             x, y, z, colors = load_ply_file(tmp_path)

#             if x is not None:
#                 fig = go.Figure(data=[go.Scatter3d(
#                     x=x,
#                     y=y,
#                     z=z,
#                     mode='markers',
#                     marker=dict(
#                         size=2,
#                         color=colors,
#                         colorscale='Viridis',
#                         opacity=0.8
#                     )
#                 )])

#                 fig.update_layout(
#                     scene=dict(
#                         xaxis_title='X',
#                         yaxis_title='Y',
#                         zaxis_title='Z',
#                         aspectmode='data'
#                     ),
#                     width=800,
#                     height=600,
#                     title="3D Point Cloud"
#                 )

#                 st.plotly_chart(fig, use_container_width=True)

#                 st.info(f"**Информация:** {len(x)} точек")

#         with st.expander("ℹ️ Управление камерой"):
#             st.markdown("""
#             ### Управление:
#             - **ЛКМ + перемещение**: Вращение камеры
#             - **ПКМ + перемещение**: Перемещение камеры
#             - **Колесо мыши**: Приближение/отдаление
#             - **WASD**: Перемещение камеры
#             - **Цифры 0-9**: Переключение между предустановленными камерами
#             - **+/-**: Следующая/предыдущая камера
#             - **P**: Автоматический пролет (carousel)
#             - **V**: Сохранить текущий вид в URL
#             """)

#         os.unlink(ply_path)

# if __name__ == "__main__":
#     main()


# SHOW FOR RAMZAN
#
#
import streamlit as st
import base64


def create_ply_viewer(ply_data=None):
    if ply_data:
        ply_base64 = base64.b64encode(ply_data).decode("utf-8")
    else:
        ply_base64 = ""

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PLY Viewer</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #1e1e1e;
                font-family: Arial, sans-serif;
            }}
            #container {{
                position: relative;
                width: 100vw;
                height: 100vh;
            }}
            #canvas {{
                display: block;
                width: 100%;
                height: 100%;
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 18px;
                background: rgba(0,0,0,0.7);
                padding: 20px;
                border-radius: 10px;
                z-index: 100;
            }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: white;
                background: rgba(0,0,0,0.7);
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }}
            #controls {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: white;
                background: rgba(0,0,0,0.7);
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <canvas id="canvas"></canvas>
            <div id="loading">🔄 Loading PLY model...</div>
            <div id="info">PLY Viewer</div>
            <div id="controls">
                🖱️ Left: Rotate | 🖱️ Right: Pan | 🖱️ Wheel: Zoom
            </div>
        </div>

        <script>
            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1e1e1e);
            
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ 
                canvas: document.getElementById('canvas'),
                antialias: true,
                alpha: true 
            }});
            
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(1, 1, 1).normalize();
            scene.add(directionalLight);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight2.position.set(-1, -1, -1).normalize();
            scene.add(directionalLight2);
            
            // Camera controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.minDistance = 0.1;
            controls.maxDistance = 1000;
            
            // Set initial camera position
            camera.position.set(2, 2, 2);
            controls.update();
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
            
            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}
            
            // Load PLY function
            function loadPLYFromBase64(base64Data) {{
                const loadingElement = document.getElementById('loading');
                const infoElement = document.getElementById('info');
                
                try {{
                    console.log('Decoding PLY data...');
                    
                    // Decode base64
                    const binaryString = atob(base64Data);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}
                    
                    // Create blob and URL
                    const blob = new Blob([bytes], {{ type: 'application/octet-stream' }});
                    const url = URL.createObjectURL(blob);
                    
                    console.log('Loading PLY from URL:', url);
                    
                    // Load PLY
                    const loader = new THREE.PLYLoader();
                    loader.load(
                        url,
                        function (geometry) {{
                            console.log('PLY geometry loaded:', geometry);
                            
                            // Compute vertex colors if not present
                            if (!geometry.getAttribute('color')) {{
                                const colors = [];
                                const position = geometry.getAttribute('position');
                                
                                for (let i = 0; i < position.count; i++) {{
                                    // Create gradient color based on position
                                    const x = position.getX(i);
                                    const y = position.getY(i); 
                                    const z = position.getZ(i);
                                    
                                    const r = (x + 1) / 2;
                                    const g = (y + 1) / 2;
                                    const b = (z + 1) / 2;
                                    
                                    colors.push(r, g, b);
                                }}
                                
                                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                            }}
                            
                            // Create material
                            const material = new THREE.PointsMaterial({{
                                size: 0.02,
                                vertexColors: true,
                                sizeAttenuation: true,
                                transparent: true,
                                opacity: 0.8
                            }});
                            
                            // Create points
                            const points = new THREE.Points(geometry, material);
                            scene.add(points);
                            
                            // Compute bounding box and center camera
                            geometry.computeBoundingSphere();
                            const sphere = geometry.boundingSphere;
                            
                            if (sphere) {{
                                const center = sphere.center;
                                const radius = sphere.radius;
                                
                                // Position camera to view entire model
                                camera.position.copy(center);
                                camera.position.x += radius * 1.5;
                                camera.position.y += radius * 1.5;
                                camera.position.z += radius * 1.5;
                                camera.lookAt(center);
                                
                                controls.target.copy(center);
                                controls.update();
                                
                                infoElement.innerHTML = `PLY Viewer | Points: ${{geometry.attributes.position.count}} | Radius: ${{radius.toFixed(2)}}`;
                            }}
                            
                            loadingElement.style.display = 'none';
                            console.log('PLY successfully loaded and displayed');
                            
                            // Clean up URL
                            URL.revokeObjectURL(url);
                        }},
                        function (xhr) {{
                            // Progress
                            const percent = (xhr.loaded / xhr.total * 100) || 0;
                            loadingElement.innerHTML = `🔄 Loading PLY model... ${{percent.toFixed(1)}}%`;
                            console.log('Loading progress:', percent + '%');
                        }},
                        function (error) {{
                            console.error('Error loading PLY:', error);
                            loadingElement.innerHTML = '❌ Error loading model: ' + error.message;
                        }}
                    );
                    
                }} catch (error) {{
                    console.error('Error processing PLY data:', error);
                    loadingElement.innerHTML = '❌ Error: ' + error.message;
                }}
            }}
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // Load PLY data if available
            if ("{ply_base64}" && "{ply_base64}".length > 0) {{
                console.log('PLY data found, length:', "{ply_base64}".length);
                loadPLYFromBase64("{ply_base64}");
            }} else {{
                document.getElementById('loading').innerHTML = '❌ No PLY data provided';
            }}
        </script>
    </body>
    </html>
    """

    return html_template


import streamlit as st
import base64
import tempfile
import os
import requests
from pathlib import Path


def create_supersplat_viewer(ply_data=None, file_name="model.ply"):
    """
    Создает HTML для встраивания SuperSplat-like вьюера
    """
    if ply_data:
        ply_base64 = base64.b64encode(ply_data).decode("utf-8")
    else:
        ply_base64 = ""

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SuperSplat Viewer</title>
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
            }}
        }}
        </script>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            #container {{
                position: relative;
                width: 100vw;
                height: 100vh;
            }}
            #canvas {{
                display: block;
                width: 100%;
                height: 100%;
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 18px;
                background: rgba(0,0,0,0.9);
                padding: 20px 30px;
                border-radius: 12px;
                z-index: 1000;
                text-align: center;
                border: 1px solid #333;
            }}
            #loading::before {{
                content: "⏳";
                font-size: 24px;
                display: block;
                margin-bottom: 10px;
            }}
            #toolbar {{
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 10px;
                color: white;
                font-size: 14px;
                border: 1px solid #333;
                backdrop-filter: blur(10px);
            }}
            #controls {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 10px;
                color: white;
                font-size: 12px;
                line-height: 1.4;
                border: 1px solid #333;
                backdrop-filter: blur(10px);
            }}
            #stats {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(0,0,0,0.8);
                padding: 12px;
                border-radius: 8px;
                color: #00ff88;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #333;
            }}
            .button {{
                background: #00ff88;
                color: #000;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 2px;
                cursor: pointer;
                font-size: 11px;
                font-weight: bold;
            }}
            .button:hover {{
                background: #00cc66;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <canvas id="canvas"></canvas>
            <div id="loading">
                <div>Loading SuperSplat Viewer</div>
                <div style="font-size: 12px; margin-top: 8px; opacity: 0.8;">Processing Gaussian Splatting data...</div>
            </div>
            <div id="toolbar">
                <strong>🎯 SuperSplat Viewer</strong>
                <div style="margin-top: 8px; font-size: 11px; opacity: 0.8;">{file_name}</div>
            </div>
            <div id="controls">
                <div><strong>🖱️ Controls:</strong></div>
                <div>• Left Drag: Rotate</div>
                <div>• Right Drag: Pan</div>
                <div>• Wheel: Zoom</div>
                <div>• W/A/S/D: Move</div>
                <div>• Q/E: Roll</div>
                <div>• R: Reset View</div>
            </div>
            <div id="stats">
                <div>FPS: <span id="fps">0</span></div>
                <div>Splats: <span id="splatCount">0</span></div>
                <div>Quality: <span id="quality">High</span></div>
            </div>
        </div>

        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
            import {{
                GaussianSplats3D,
                VisualizationType
            }} from 'https://cdn.jsdelivr.net/npm/gaussian-splats-3d@1.0.0/dist/gaussian-splats-3d.module.js';

            let scene, camera, renderer, controls;
            let gaussianSplats;
            let animationId;
            let lastTime = 0;
            let frameCount = 0;
            let fps = 0;

            async function init() {{
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);

                // Camera
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 0, 5);

                // Renderer
                const canvas = document.getElementById('canvas');
                renderer = new THREE.WebGLRenderer({{
                    canvas,
                    antialias: true,
                    powerPreference: "high-performance"
                }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

                // Controls
                controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.screenSpacePanning = false;
                controls.minDistance = 0.1;
                controls.maxDistance = 100;
                controls.maxPolarAngle = Math.PI;

                // Lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(1, 1, 1).normalize();
                scene.add(directionalLight);

                // Handle resize
                window.addEventListener('resize', onWindowResize);

                // Load Gaussian Splatting data
                await loadGaussianSplattingData();

                // Start animation
                animate();
            }}

            async function loadGaussianSplattingData() {{
                const loadingElement = document.getElementById('loading');
                
                try {{
                    loadingElement.innerHTML = `
                        <div>🎯 Loading Gaussian Splatting</div>
                        <div style="font-size: 12px; margin-top: 8px; opacity: 0.8;">Decoding PLY data...</div>
                    `;

                    if (!"{ply_base64}" || "{ply_base64}".length === 0) {{
                        throw new Error('No PLY data provided');
                    }}

                    // Decode base64 data
                    const binaryString = atob("{ply_base64}");
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}

                    loadingElement.innerHTML = `
                        <div>🎯 Processing Splats</div>
                        <div style="font-size: 12px; margin-top: 8px; opacity: 0.8;">Creating 3D visualization...</div>
                    `;

                    // Create blob for GaussianSplats3D
                    const blob = new Blob([bytes], {{ type: 'application/octet-stream' }});
                    const url = URL.createObjectURL(blob);

                    // Initialize Gaussian Splatting
                    gaussianSplats = new GaussianSplats3D({{
                        scene: scene,
                        camera: camera,
                        renderer: renderer,
                        fileURL: url,
                        useCache: false,
                        onProgress: (progress) => {{
                            const percent = Math.round(progress * 100);
                            loadingElement.innerHTML = `
                                <div>🎯 Loading {file_name}</div>
                                <div style="font-size: 12px; margin-top: 8px; opacity: 0.8;">${{percent}}% complete</div>
                            `;
                        }}
                    }});

                    await gaussianSplats.initialize();

                    // Clean up URL
                    URL.revokeObjectURL(url);

                    // Update stats
                    document.getElementById('splatCount').textContent = gaussianSplats.splatCount?.toLocaleString() || 'N/A';

                    loadingElement.style.display = 'none';
                    
                    console.log('✅ Gaussian Splatting loaded successfully');

                }} catch (error) {{
                    console.error('❌ Error loading Gaussian Splatting:', error);
                    loadingElement.innerHTML = `
                        <div>❌ Loading Failed</div>
                        <div style="font-size: 12px; margin-top: 8px; opacity: 0.8;">${{error.message}}</div>
                    `;
                }}
            }}

            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}

            function animate(currentTime) {{
                animationId = requestAnimationFrame(animate);

                // Calculate FPS
                frameCount++;
                if (currentTime - lastTime >= 1000) {{
                    fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                    document.getElementById('fps').textContent = fps;
                    frameCount = 0;
                    lastTime = currentTime;
                }}

                // Update controls
                controls.update();

                // Render
                renderer.render(scene, camera);
            }}

            // Keyboard controls
            window.addEventListener('keydown', (event) => {{
                if (!camera) return;

                const moveSpeed = 0.2;
                const rotateSpeed = 0.03;

                switch(event.key.toLowerCase()) {{
                    case 'w':
                        camera.translateZ(-moveSpeed);
                        break;
                    case 's':
                        camera.translateZ(moveSpeed);
                        break;
                    case 'a':
                        camera.translateX(-moveSpeed);
                        break;
                    case 'd':
                        camera.translateX(moveSpeed);
                        break;
                    case 'q':
                        camera.rotateZ(rotateSpeed);
                        break;
                    case 'e':
                        camera.rotateZ(-rotateSpeed);
                        break;
                    case 'r':
                        // Reset camera
                        camera.position.set(0, 0, 5);
                        camera.rotation.set(0, 0, 0);
                        controls.reset();
                        break;
                }}
            }});

            // Initialize when page loads
            window.addEventListener('load', init);

            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {{
                if (animationId) {{
                    cancelAnimationFrame(animationId);
                }}
                if (gaussianSplats) {{
                    gaussianSplats.dispose();
                }}
            }});
        </script>
    </body>
    </html>
    """

    return html_template


def main():
    # st.set_page_config(
    #     page_title="SuperSplat Viewer",
    #     layout="wide",
    #     page_icon="🎯"
    # )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main .block-container {
        padding-top: 1rem;
    }
    .stButton button {
        background: linear-gradient(45deg, #00ff88, #00ccff);
        color: black;
        font-weight: bold;
        border: none;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    # col1, col2 = st.columns([1, 3])
    # with col1:
    #     st.markdown("# 🎯")
    # with col2:
    #     st.title("SuperSplat Viewer")
    #     st.markdown("**Real-time Gaussian Splatting 3D Visualization**")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Gaussian Splatting PLY File",
        type=["ply"],
        help="Upload a .ply file containing Gaussian Splatting data",
    )

    if uploaded_file:
        ply_data = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # File analysis
        st.success(f"✅ **{file_name}** loaded successfully!")

        # File info cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("File Size", f"{len(ply_data) / 1024 / 1024:.2f} MB")

        with col2:
            # Check if it's Gaussian Splatting format
            header_text = ply_data[:2000].decode("utf-8", errors="ignore")
            if all(prop in header_text for prop in ["f_dc_0", "scale_0", "rot_0"]):
                st.metric("Format", "🎯 Gaussian Splatting")
            else:
                st.metric("Format", "⚠️ Check Format")

        with col3:
            # Count vertices
            vertex_match = header_text.find("element vertex")
            if vertex_match != -1:
                vertex_line = header_text[vertex_match : vertex_match + 100].split(
                    "\\n"
                )[0]
                try:
                    vertex_count = int(vertex_line.split()[2])
                    st.metric("Splats", f"{vertex_count:,}")
                except:
                    st.metric("Splats", "N/A")

        with col4:
            st.metric("Status", "🟢 Ready")

        # Create and display viewer
        with st.spinner("🚀 Initializing SuperSplat Viewer..."):
            html_content = create_supersplat_viewer(ply_data, file_name)

        # Viewer
        st.markdown("### 🎮 3D Viewer")
        st.components.v1.html(html_content, height=800, scrolling=False)

        # Controls and info
        with st.expander("🎯 About SuperSplat Viewer"):
            st.markdown("""
            **SuperSplat Technology:**
            - 🎯 **Gaussian Splatting**: Advanced 3D representation using millions of Gaussian distributions
            - ⚡ **Real-time Rendering**: Hardware-accelerated visualization
            - 🎨 **Photorealistic Quality**: Per-splat colors and materials
            - 🔧 **Interactive Controls**: Full 6DOF camera control
            
            **Optimized Features:**
            - Automatic level-of-detail (LOD)
            - Frustum culling for performance
            - Efficient memory usage
            - Smooth camera movements
            
            **Best for:**
            - 3D scanning data
            - Neural radiance fields (NeRF)
            - Photogrammetry reconstructions
            - Real-time 3D applications
            """)

        with st.expander("🎮 Controls Guide"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Mouse Controls:**
                - **Left Click + Drag**: Rotate view
                - **Right Click + Drag**: Pan camera
                - **Mouse Wheel**: Zoom in/out
                - **Double Click**: Focus on point
                """)
            with col2:
                st.markdown("""
                **Keyboard Controls:**
                - **W/S**: Move forward/backward
                - **A/D**: Move left/right
                - **Q/E**: Roll camera
                - **R**: Reset view
                - **Space**: Toggle auto-rotate
                """)

    else:
        # Welcome screen
        st.info("👆 **Upload a Gaussian Splatting PLY file to begin**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🚀 Quick Start
            
            1. **Upload** a `.ply` file in Gaussian Splatting format
            2. **Wait** for automatic processing
            3. **Interact** with the 3D viewer
            4. **Explore** your model in real-time
            
            ### 📁 Supported Format
            
            Files should contain:
            - Position data (x, y, z)
            - Spherical harmonics (f_dc_0, f_dc_1, f_dc_2)
            - Scale parameters (scale_0, scale_1, scale_2)
            - Rotation quaternions (rot_0, rot_1, rot_2, rot_3)
            - Opacity values
            """)

        with col2:
            st.markdown("""
            ### 🎯 What is Gaussian Splatting?
            
            Gaussian Splatting is a revolutionary 3D technique that:
            
            - ✅ **Represents scenes** as millions of Gaussian distributions
            - ✅ **Real-time rendering** at high quality
            - ✅ **Efficient compression** of 3D data
            - ✅ **Photorealistic results** from images/video
            
            ### 🔧 Technical Features
            
            - **62 properties** per splat
            - **Hardware acceleration** via WebGL
            - **Automatic optimization** for performance
            - **Interactive editing** capabilities
            """)


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
