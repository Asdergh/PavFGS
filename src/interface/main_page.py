import base64
import json

# from streamlit_stl import stl_from_text
import os

import streamlit as st

# from plyfile import PlyData


st.set_page_config(
    layout="wide",
    page_title="PavFGS — 4D / sEEG",
    page_icon="🧠",
)

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


BRUSH_VIEWER_URL = os.environ.get(
    "PAVFGS_BRUSH_URL", "https://arthurbrussee.github.io/brush-demo"
)


def _render_brush_window():
    st.markdown("### 🖌️ Brush — 4D Gaussian Splatting Viewer")
    brush_url = st.text_input(
        "Brush URL (оставь пустым для демо)",
        value=BRUSH_VIEWER_URL,
        key="brush_url",
        help="Демо: https://arthurbrussee.github.io/brush-demo | Локально: http://localhost:3000 после npm run dev в brush/brush_nextjs",
    )
    url = brush_url.strip() or BRUSH_VIEWER_URL
    if not url.strip().lower().startswith("http://localhost"):
        st.info(
            "**Ползунок времени** и загрузка файлов по URL работают только в **локальной** сборке Brush. "
            "Сейчас открыт демо — ползунка там нет. Чтобы увидеть ползунок: в терминале выполни "
            "`cd brush/brush_nextjs && npm run build:wasm-dev && npm run dev`, затем укажи URL **http://localhost:3000**."
        )
    iframe_html = f"""
    <iframe
        src="{url}"
        style="width:100%; height:800px; border:1px solid #333; border-radius:8px; background:#000;"
        allow="webgpu"
        allowfullscreen
        title="Brush Viewer"
   ></iframe>
    """
    st.components.v1.html(iframe_html, height=820, scrolling=False)
    # with st.expander("ℹ️ О Brush"):
    #     st.markdown("""
    #     **Brush** — движок 3D-реконструкции на базе Gaussian Splatting из папки `brush/`.
    #     - Поддерживает .ply, .compressed.ply и zip с анимацией (несколько кадров).
    #     - В браузере только Chrome/Edge (WebGPU).
    #     - **Ползунок времени:** при загрузке анимации (zip с кадрами или 4D) внизу сцены появляется ползунок **⏱ … / N с** — перетаскивай для просмотра по секунде/кадру. Сверху справа — кнопка **▶/⏸** (play/pause).
    #     Локально: `cd brush/brush_nextjs && npm run dev`, затем URL `http://localhost:3000`.
    #     """)


def _render_math_showcase(*, is_en: bool) -> None:
    """
    Пример оформления формул (KaTeX в iframe) — шаблон для роста объёма ВКР.
    """
    if is_en:
        title = "Sample LaTeX block (Poisson-type PDE)"
        sub = (
            "Same class of elliptic problems as in the thesis when discussing FVM on the volume domain "
            "(forward / inverse EEG context)."
        )
        sep = "Divergence form (conductivity σ):"
        foot = (
            "As the paper grows: add <code>align</code>, numbered equations, theorem/corollary "
            "environments, and bibliography—same wrapper style."
        )
    else:
        title = "Пример оформления формул (уравнение Пуассона / эллиптическая постановка)"
        sub = (
            "Тот же класс задач, что и при дискретизации <strong>МКО</strong> на объёмной сцене в обсуждении "
            "прямой/обратной ЭЭГ в тексте ВКР."
        )
        sep = "Дивергентная форма (проводимость σ):"
        foot = (
            "Когда объём ВКР вырастет: сюда же — нумерация, <code>align</code>, ссылки на формулы и "
            "библиография; визуальный стиль можно сохранить."
        )

    tex_main = (
        r"\begin{gathered}"
        r"\nabla^2 V(\mathbf{r}) = f(\mathbf{r}),\qquad \mathbf{r} \in \Omega \subset \mathbb{R}^3 \\[0.55em]"
        r"V\big|_{\partial\Omega} = g"
        r"\end{gathered}"
    )
    tex_div = (
        r"\nabla \cdot \bigl( \sigma(\mathbf{r}) \nabla V(\mathbf{r}) \bigr) = S(\mathbf{r})"
    )

    safe_main = json.dumps(tex_main)
    safe_div = json.dumps(tex_div)

    html = f"""
    <!DOCTYPE html>
    <html lang="{'en' if is_en else 'ru'}">
    <head>
      <meta charset="utf-8"/>
      <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
        crossorigin="anonymous"/>
      <style>
        :root {{
          --bg0: rgba(8, 14, 34, 0.97);
          --edge: rgba(125, 152, 255, 0.35);
          --t1: #dce8ff;
          --t2: #9fb0d4;
          --accent: #6ab0ff;
        }}
        body {{
          margin: 0;
          font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
          background: transparent;
          color: var(--t2);
        }}
        .math-card {{
          margin-top: 4px;
          border-radius: 18px;
          padding: 22px 22px 18px 22px;
          background:
            radial-gradient(ellipse 90% 70% at 100% 0%, rgba(80, 140, 255, 0.18), transparent 55%),
            radial-gradient(ellipse 60% 50% at 0% 100%, rgba(0, 200, 255, 0.1), transparent 50%),
            linear-gradient(165deg, var(--bg0) 0%, rgba(12, 22, 52, 0.95) 100%);
          border: 1px solid var(--edge);
          box-shadow:
            0 16px 36px rgba(0, 0, 0, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }}
        .math-card h3 {{
          margin: 0 0 8px 0;
          font-size: 1.05rem;
          font-weight: 700;
          color: var(--t1);
          letter-spacing: 0.02em;
        }}
        .math-card .sub {{
          margin: 0 0 16px 0;
          font-size: 0.86rem;
          line-height: 1.55;
          color: var(--t2);
        }}
        .k-wrap {{
          text-align: center;
          padding: 12px 8px;
          border-radius: 12px;
          background: rgba(5, 10, 28, 0.55);
          border: 1px solid rgba(100, 130, 220, 0.2);
        }}
        .sep {{
          margin: 14px 0 6px 0;
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--accent);
          text-align: center;
          opacity: 0.95;
        }}
        .foot {{
          margin-top: 14px;
          padding-top: 12px;
          border-top: 1px solid rgba(125, 152, 255, 0.2);
          font-size: 0.8rem;
          line-height: 1.45;
          color: #8a9ec4;
        }}
        .katex {{ font-size: 1.08em !important; }}
        .k-wrap.second .katex {{ font-size: 0.98em !important; }}
      </style>
    </head>
    <body>
      <div class="math-card">
        <h3>{title}</h3>
        <p class="sub">{sub}</p>
        <div class="k-wrap" id="k1"></div>
        <div class="sep">{sep}</div>
        <div class="k-wrap second" id="k2"></div>
        <div class="foot">{foot}</div>
      </div>
      <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        crossorigin="anonymous"></script>
      <script>
        function go() {{
          if (typeof katex === "undefined") {{
            setTimeout(go, 30);
            return;
          }}
          katex.render({safe_main}, document.getElementById("k1"), {{
            displayMode: true,
            throwOnError: false
          }});
          katex.render({safe_div}, document.getElementById("k2"), {{
            displayMode: true,
            throwOnError: false
          }});
        }}
        go();
      </script>
    </body>
    </html>
    """

    st.components.v1.html(html, height=430, scrolling=False)


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
    :root {
        --page-bg: #040813;
        --text-primary: #e5ecff;
        --text-secondary: #9fb0d4;
        --card-bg: rgba(10, 16, 36, 0.78);
        --border-color: rgba(125, 152, 255, 0.26);
        --accent: #5b8cff;
        --accent-soft: rgba(69, 111, 255, 0.16);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 18%, rgba(84, 131, 255, 0.22), transparent 24%),
            radial-gradient(circle at 80% 10%, rgba(125, 82, 255, 0.18), transparent 25%),
            radial-gradient(circle at 80% 68%, rgba(35, 181, 255, 0.16), transparent 30%),
            linear-gradient(180deg, #030712 0%, #050a18 45%, #02050e 100%);
        color: var(--text-primary);
    }
    [data-testid="stHeader"] {
        background: rgba(3, 7, 18, 0.45);
        backdrop-filter: blur(8px);
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .paper-hero {
        background:
            radial-gradient(circle at 75% -15%, rgba(110, 142, 255, 0.34), transparent 45%),
            radial-gradient(circle at 10% 110%, rgba(0, 221, 255, 0.14), transparent 40%),
            linear-gradient(130deg, rgba(8, 14, 34, 0.98) 0%, rgba(13, 28, 66, 0.96) 55%, rgba(28, 57, 143, 0.9) 100%);
        border-radius: 22px;
        padding: 36px 34px;
        border: 1px solid rgba(145, 168, 255, 0.28);
        color: #eef4ff;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.07);
        margin-bottom: 1.25rem;
    }
    .paper-title {
        margin: 0 0 10px 0;
        font-size: 2.1rem;
        line-height: 1.22;
        letter-spacing: 0.2px;
        font-weight: 800;
    }
    .paper-subtitle {
        margin: 0;
        opacity: 0.9;
        font-size: 1.03rem;
        line-height: 1.65;
        max-width: 930px;
    }
    .author-line {
        margin-top: 12px;
        font-size: 0.95rem;
        color: #b8c8ec;
        letter-spacing: 0.3px;
    }
    .key-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }
    .key-tag {
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(100, 140, 255, 0.12);
        border: 1px solid rgba(140, 170, 255, 0.25);
        color: #b5c7f5;
    }
    .ol-clinical {
        margin: 0.4rem 0 0 0;
        padding-left: 1.1rem;
        color: var(--text-secondary);
        font-size: 0.94rem;
        line-height: 1.5;
    }
    .ol-clinical li { margin-bottom: 6px; }
    .stack-line {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.93rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .stack-line strong { color: #d4e0ff; min-width: 7.2rem; font-weight: 600; }
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }
    .meta-pill {
        background: rgba(191, 211, 255, 0.11);
        border: 1px solid rgba(177, 200, 255, 0.26);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.83rem;
        letter-spacing: 0.2px;
        color: #d9e4ff;
    }
    .section-title {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 1.12rem;
        margin-top: 0.3rem;
        margin-bottom: 0.55rem;
    }
    .paper-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(170, 191, 255, 0.08);
        backdrop-filter: blur(9px);
        height: 100%;
    }
    .paper-card h4 {
        margin: 0 0 9px 0;
        color: #dce8ff;
        font-size: 1.01rem;
    }
    .paper-card p, .paper-card li {
        color: var(--text-secondary);
        line-height: 1.55;
        font-size: 0.95rem;
    }
    .paper-callout {
        border-left: 4px solid var(--accent);
        background: linear-gradient(90deg, var(--accent-soft) 0%, rgba(53, 74, 145, 0.2) 100%);
        border: 1px solid rgba(120, 152, 255, 0.3);
        border-radius: 12px;
        padding: 14px 14px 12px 14px;
        color: #c8d7ff;
        margin: 0.4rem 0 0.75rem 0;
    }
    .placeholder {
        border: 1px dashed rgba(160, 184, 255, 0.44);
        border-radius: 12px;
        padding: 13px 14px;
        margin-top: 10px;
        color: #8ea2cc;
        background: rgba(13, 20, 45, 0.72);
        font-size: 0.92rem;
    }
    .paper-prose p:last-child { margin-bottom: 0; }
    .stButton button {
        background: linear-gradient(135deg, #3765ff, #30b7ff);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    language = st.radio("Language / Язык", options=["RU", "EN"], horizontal=True, key="paper_language")
    is_en = language == "EN"

    if is_en:
        kicker = "Graduation project · epileptologist office + PavFGS research pipeline"
        hero_title = "PavFGS — 4D visualization of brain activity from stereo-EEG (sEEG)"
        hero_subtitle = (
            "Diploma topic: <em>Software implementation of an epileptologist’s office</em>. "
            "The PavFGS track builds a pipeline for representative <strong>4D (dynamic 3D)</strong> models of "
            "cortical activity from sEEG, combining <strong>TensoRF</strong>-style tensor factorization of "
            "voxel feature maps, <strong>recurrent</strong> modeling in the frequency domain, and "
            "<strong>3D Gaussian Splatting</strong> for interactive display — integrated below via the "
            "<strong>Brush</strong> WebGPU viewer when run locally."
        )
        author_line = "Ramzan Khatсiev · Stepan Ershov · Egor Skvortsov · April 2026"
        version_pill = "Diploma / VKR"
        language_pill = "EN"
        status_pill = "sEEG · neurology"
        format_pill = "Web + WebGPU (Brush)"
        section_context = "Abstract and research context"
        abstract_title = "Abstract"
        keywords_title = "Keywords"
        abstract_html = (
            "<p>Modeling and visualization of high-dimensional data is central to modern medicine; "
            "neurology and BCI research increasingly rely on joint geometric and activity maps. "
            "Stereoelectroencephalography (sEEG) combines a metric brain scene with time-resolved activity "
            "in electrode regions, but classical volumetric PDE solvers (e.g. finite volumes for inverse EEG) "
            "are too heavy to pair with end-to-end deep learning. This work develops a <strong>software "
            "environment for an epileptologist’s practice</strong> and a <strong>PavFGS</strong> pipeline: "
            "4D activity models from sEEG using TensoRF-style decomposition, recurrent processing of "
            "frequency-content representations, and 3D Gaussian Splatting so clinicians can "
            "<strong>rotate, pan, and slice</strong> an interactive 3D model and navigate arbitrary "
            "time windows with emphasis on <strong>temporal anomalies</strong> — going beyond tools limited "
            "to flat slices and slow long-sequence review.</p>"
        )
        keywords = (
            "stereo-EEG, sEEG, TensoRF, 3D Gaussian Splatting, 4D reconstruction, BCI, neurology, "
            "inverse EEG, WebGPU, Brush, epileptology"
        )
        research_problem_title = "Research problem"
        research_problem_html = (
            "<p>Spatial clinical viewers for this modality mostly offer <strong>isolated 2D slices</strong> "
            "and struggle with <strong>long recordings</strong>. Classical field-based simulators need full "
            "state access and mesh-scale solvers, which blocks lightweight coupling with modern neural "
            "networks. The project asks for a <strong>differentiable, compact scene primitive</strong> and a "
            "<strong>time model</strong> aligned with sEEG, enabling efficient training-time simulation and "
            "real-time-quality exploration in the browser.</p>"
        )
        section_mid = "From sEEG workflow to PavFGS stack"
        section_clinical = "Clinical pipeline (sEEG)"
        clinical_html = (
            "<p class='paper-prose' style='margin:0;'>Typical sEEG workflow (as in the thesis):</p>"
            "<ol class='ol-clinical'>"
            "<li>MRI to obtain a metric, cartographic head/brain map (e.g. perfusion- or contrast-based).</li>"
            "<li>Invasive placement of depth electrodes in regions of interest.</li>"
            "<li>Repeat imaging with electrodes in place to co-register leads with anatomy.</li>"
            "</ol>"
            "<p style='margin:10px 0 0 0;font-size:0.92rem;color:var(--text-secondary);'>"
            "That separates <strong>metric geometry</strong> from <strong>dynamic activity maps</strong> — "
            "the gap PavFGS targets at the activity / 4D side.</p>"
        )
        section_stack = "Technology stack (PavFGS)"
        stack_html = (
            "<div class='stack-line'><strong>TensoRF</strong><span>Tensor decomposition of voxel feature "
            "grids for compact 3D–temporal fields.</span></div>"
            "<div class='stack-line'><strong>Recurrent + freq.</strong><span>Sequence modeling with "
            "representations in a frequency basis over sEEG.</span></div>"
            "<div class='stack-line'><strong>3DGS</strong><span>Anisotropic Gaussians: continuity, "
            "densification, per-primitive motion for 4D.</span></div>"
            "<div class='stack-line'><strong>Brush</strong><span>WebGPU splat viewer; local dev unlocks a "
            "time slider and 4D assets.</span></div>"
        )
        section_structure = "Thesis map"
        intro_title = "1. Introduction"
        intro_body = (
            "<p>Framed by <strong>3D vision</strong> (voxels vs. point clouds), <strong>NeRF</strong>, and "
            "<strong>3DGS</strong> as a compromise between semantic smoothness and dynamic deformations. "
            "Connects to <strong>multimodal neuro-AI</strong>, sEEG clinical reality, and the need for a "
            "neural-friendly volumetric front-end (section 1 of the thesis).</p>"
        )
        method_title = "2. Methodology (outline)"
        method_body = (
            "<p><strong>Backbone / encoder</strong> narrative for spatial codes; <strong>3D convolutions</strong> "
            "on volumetric blocks <em>T</em> with learnable 3D window banks (thesis §2.1). The PavFGS path "
            "combines TensoRF factorization, recurrent frequency-domain sEEG coding, and 3DGS export for "
            "viewing. Full derivations follow the document’s theory chapter.</p>"
        )
        results_title = "3. Experiments & expected outcomes"
        results_body = (
            "<p>Targets: <strong>resource- and time-efficient</strong> 4D models of brain activity under "
            "sEEG monitoring, anomaly-friendly timelines, and comparison with <strong>slice-only</strong> "
            "baselines. Quantitative tables and ablations are added as datasets and baselines are finalized "
            "in the thesis.</p>"
        )
        callout_label = "Viewer:"
        callout_text = (
            "The embed below is <strong>Brush</strong> (4D Gaussian Splatting in the browser). The public "
            "demo is WebGPU-only; for the <strong>time scrubber</strong> and file loads, run the local app "
            "(see the note under the URL field)."
        )
        section_interactive = "Interactive preview (Brush)"
    else:
        kicker = "ВКР · кабинет эпилептолога + исследовательский конвейер PavFGS"
        hero_title = "PavFGS — 4D-визуализация активности мозга по данным сЭЭГ"
        hero_subtitle = (
            "Тема диплома: <em>Программная реализация кабинета врача-эпилептолога</em>. "
            "Направление PavFGS — пайплайн <strong>репрезентативных 4D (динамических 3D) моделей</strong> "
            "распределения активности мозга по сЭЭГ: <strong>тензорная декомпозиция</strong> воксельных карт "
            "признаков (стиль <strong>TensoRF</strong>), <strong>рекуррентная</strong> обработка сигналов в "
            "<strong>частотной области</strong> (разложение по базису) и <strong>3D Gaussian Splatting</strong> "
            "для интерактивного просмотра; ниже — вьювер <strong>Brush</strong> (WebGPU) при локальном запуске."
        )
        author_line = "Хациев Рамзан · Ершов Степан · Скворцов Егор · май 2026"
        version_pill = "Диплом / ВКР"
        language_pill = "RU"
        status_pill = "сЭЭГ · неврология"
        format_pill = "Web + WebGPU (Brush)"
        section_context = "Аннотация и научный контекст"
        abstract_title = "Аннотация"
        keywords_title = "Ключевые слова"
        abstract_html = (
            "<p>Моделирование и визуализация многомерных данных — базис современной медицины; неврология и "
            "направление <strong>BCI</strong> (интерфейс «мозг–компьютер») опираются на сочетание "
            "картографически точных сцен и карт <strong>физиологической активности</strong>. Для "
            "стерео-ЭЭГ (сЭЭГ) типичен конвейер: метрическая сцена головы, инвазивные электроды, повторная "
            "визуализация с привязкой контактов. Классические объёмные методы (МКО, уравнение "
            "Эйлера–Пуассона для обратной ЭЭГ-проекции) <strong>ресурсоёмки</strong> и плохо стыкуются с "
            "современным deep learning. Работа посвящена <strong>программной реализации кабинета врача-"
            "эпилептолога</strong> и конвейеру <strong>PavFGS</strong>: генерация 4D-моделей активности по "
            "сЭЭГ с опорой на TensoRF-подобную тензорную декомпозицию, рекуррентную обработку частотных "
            "репрезентаций и 3DGS, чтобы врач мог работать с <strong>интерактивной 3D-моделью</strong> "
            "(вращение, перенос, сечения) и исследовать <strong>любой интервал времени</strong> с акцентом на "
            "<strong>темпоральные аномалии</strong> — в отличие от инструментов, ограниченных плоскими слайсами "
            "и длинными «тормозящими» просмотрами.</p>"
        )
        keywords = (
            "стерео-ЭЭГ, сЭЭГ, TensoRF, 3D Gaussian Splatting, 4D-реконструкция, BCI, неврология, "
            "обратная проекция ЭЭГ, WebGPU, Brush, эпилептология"
        )
        research_problem_title = "Проблема исследования"
        research_problem_html = (
            "<p>Для объёмной активности мозга по сЭЭГ <strong>не хватает удобных пространственных "
            "визуализаторов</strong>: существующие аналоги в основном дают отдельные слайсы и слабо "
            "масштабируются на <strong>длинные ряды</strong>. Численные полевые методы требуют полного доступа к "
            "данным и тяжело сочетаются с нейросетями. Требуется <strong>компактный дифференцируемый</strong> "
            "каркас 3D-сцены и <strong>временная модель</strong>, согласованная с сигналом сЭЭГ, приемлемая по "
            "ресурсам и пригодная для обучения и клинического просмотра.</p>"
        )
        section_mid = "От клинического сЭЭГ-контейнера к стеку PavFGS"
        section_clinical = "Клинический контекст (сЭЭГ)"
        clinical_html = (
            "<p class='paper-prose' style='margin:0;'>В работе выделена типовая цепочка сЭЭГ:</p>"
            "<ol class='ol-clinical'>"
            "<li>МРТ: картографически точная сцена (в т.ч. васкулярный/оксигенационный контраст по ВКР).</li>"
            "<li>Инвазивная установка датчиков в зоны ЦНС.</li>"
            "<li>Повторное сканирование с электродами — наглядная привязка контактов к анатомии.</li>"
            "</ol>"
            "<p style='margin:10px 0 0 0;font-size:0.92rem;color:var(--text-secondary);'>"
            "Так отделяется <strong>метрическая геометрия</strong> от <strong>динамики активности</strong>; "
            "PavFGS фокусируется на 4D-представлении активности.</p>"
        )
        section_stack = "Технологический стек (PavFGS)"
        stack_html = (
            "<div class='stack-line'><strong>TensoRF</strong><span>Декомпозиция тензора воксельных карт "
            "признаков — компактное 3D+время поле.</span></div>"
            "<div class='stack-line'><strong>Рекуррент + частоты</strong><span>Обработка последовательностей "
            "сЭЭГ в частотных репрезентациях (разложение по базису).</span></div>"
            "<div class='stack-line'><strong>3DGS</strong><span>Анизотропные гауссианы: непрерывность сцены, "
            "денсификация, динамика по примитивам.</span></div>"
            "<div class='stack-line'><strong>Brush</strong><span>Веб-вьювер сплатов на WebGPU; локально — "
            "ползунок времени и загрузка 4D.</span></div>"
        )
        section_structure = "Структура работы (по ВКР)"
        intro_title = "1. Введение"
        intro_body = (
            "<p>Обоснование двух типов 3D-визуализации в медицине, роли <strong>3D Vision</strong> (воксели, "
            "облака точек), пути <strong>NeRF → 3DGS</strong> как компромисса между сглаженностью и динамикой. "
            "Связь с BCI, нейроинтерфейсами и клиникой сЭЭГ (раздел 1 в документе).</p>"
        )
        method_title = "2. Методология (по тексту ВКР)"
        method_body = (
            "<p>Формирование <strong>backbone</strong>-представлений и <strong>3D-свёртки</strong> на воксельных "
            "блоках <em>T</em> с банками обучаемых окон (п. 2.1). Конвейер PavFGS: TensoRF, рекуррентный "
            "частотный контур, 3DGS и интеграция с веб-вьюером. Подробные формулы — в PDF ВКР.</p>"
        )
        results_title = "3. Эксперименты и ожидаемые результаты"
        results_body = (
            "<p>Целевые эффекты: <strong>оптимальное по времени и ресурсам</strong> формирование моделей "
            "активности при мониторинге сЭЭГ, работа с <strong>темпоральными аномалиями</strong>, навигация по "
            "времени; сравнение с baseline «только слайсы». Итоговые метрики — по мере замыкания "
            "экспериментальной части в финальной ВКР.</p>"
        )
        callout_label = "Просмотр:"
        callout_text = (
            "Ниже встроен <strong>Brush</strong> (4D Gaussian Splatting в браузере). Публичное демо — только "
            "WebGPU; <strong>ползунок времени</strong> и загрузка файлов — в <strong>локальной</strong> "
            "сборке (см. подсказку под полем URL)."
        )
        section_interactive = "Интерактивное превью (Brush)"

    key_tags_html = "".join(
        f'<span class="key-tag">{k.strip()}</span>' for k in keywords.split(",") if k.strip()
    )

    st.markdown(
        f"""
        <div class="paper-hero">
            <div style="font-size: 0.88rem; opacity: 0.88; margin-bottom: 8px; letter-spacing: 0.04em;">
                {kicker}
            </div>
            <h1 class="paper-title">{hero_title}</h1>
            <p class="paper-subtitle">{hero_subtitle}</p>
            <div class="author-line">{author_line}</div>
            <div class="meta-row">
                <span class="meta-pill">{version_pill}</span>
                <span class="meta-pill">{language_pill}</span>
                <span class="meta-pill">{status_pill}</span>
                <span class="meta-pill">{format_pill}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="section-title">{section_context}</div>', unsafe_allow_html=True)
    left, right = st.columns([1.7, 1.3], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="paper-card paper-prose">
                <h4>{abstract_title}</h4>
                {abstract_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="paper-card">
                <h4>{keywords_title}</h4>
                <div class="key-tags">{key_tags_html}</div>
                <h4 style="margin-top:16px;">{research_problem_title}</h4>
                <div class="paper-prose">{research_problem_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-title">{section_mid}</div>', unsafe_allow_html=True)
    clin, tech = st.columns(2, gap="large")
    with clin:
        st.markdown(
            f"""
            <div class="paper-card paper-prose">
                <h4>{section_clinical}</h4>
                {clinical_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tech:
        st.markdown(
            f"""
            <div class="paper-card paper-prose">
                <h4>{section_stack}</h4>
                {stack_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-title">{section_structure}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="paper-card paper-prose">
                <h4>{intro_title}</h4>
                {intro_body}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="paper-card paper-prose">
                <h4>{method_title}</h4>
                {method_body}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="paper-card paper-prose">
                <h4>{results_title}</h4>
                {results_body}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="paper-callout">
            <strong>{callout_label}</strong> {callout_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="section-title">{section_interactive}</div>', unsafe_allow_html=True)
    _render_brush_window()
    _render_math_showcase(is_en=is_en)

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
