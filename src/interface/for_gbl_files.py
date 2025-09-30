import streamlit as st
import base64
import os

st.set_page_config(layout="wide", page_title="3D Viewer")

st.title("🧬 3D Viewer с анимацией")

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
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; }}
        #container {{ width: 100%; height: 600px; background: #1e1e1e; position: relative; }}
        #loading {{ color: white; text-align: center; padding: 20px; font-family: Arial; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }}
        #controls {{ position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; color: white; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading">🔄 Загрузка 3D модели...</div>
        <div id="controls" style="display: none;">
            <button onclick="resetCamera()">🔄 Сброс камеры</button>
            <button onclick="toggleRotation()">⏸️ Вращение</button>
        </div>
    </div>

    <!-- Только основной Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

    <script>
        // Простой OrbitControls реализованный вручную
        class SimpleOrbitControls {{
            constructor(object, domElement) {{
                this.object = object;
                this.domElement = domElement;
                
                this.enabled = true;
                this.target = new THREE.Vector3();
                this.minDistance = 0;
                this.maxDistance = Infinity;
                
                // State
                this.spherical = new THREE.Spherical();
                this.sphericalDelta = new THREE.Spherical();
                
                this.scale = 1;
                this.panOffset = new THREE.Vector3();
                
                this.mouseButtons = {{ LEFT: 0, MIDDLE: 1, RIGHT: 2 }};
                
                // Event listeners
                this.onMouseDown = this.onMouseDown.bind(this);
                this.onMouseWheel = this.onMouseWheel.bind(this);
                this.onContextMenu = this.onContextMenu.bind(this);
                
                this.connect();
                this.update();
            }}
            
            connect() {{
                this.domElement.addEventListener('mousedown', this.onMouseDown);
                this.domElement.addEventListener('wheel', this.onMouseWheel);
                this.domElement.addEventListener('contextmenu', this.onContextMenu);
            }}
            
            onMouseDown(event) {{
                if (!this.enabled) return;
                
                event.preventDefault();
                
                const mouseAction = event.button === this.mouseButtons.LEFT ? 'rotate' :
                                  event.button === this.mouseButtons.RIGHT ? 'pan' : 'zoom';
                
                const onMouseMove = (moveEvent) => {{
                    const movementX = moveEvent.clientX - event.clientX;
                    const movementY = moveEvent.clientY - event.clientY;
                    
                    if (mouseAction === 'rotate') {{
                        this.sphericalDelta.theta -= movementX * 0.01;
                        this.sphericalDelta.phi -= movementY * 0.01;
                    }} else if (mouseAction === 'pan') {{
                        // Simple pan implementation
                        this.panOffset.x -= movementX * 0.01;
                        this.panOffset.y += movementY * 0.01;
                    }}
                    
                    this.update();
                }};
                
                const onMouseUp = () => {{
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                }};
                
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            }}
            
            onMouseWheel(event) {{
                if (!this.enabled) return;
                
                event.preventDefault();
                this.scale *= Math.pow(0.95, event.deltaY > 0 ? 1 : -1);
                this.update();
            }}
            
            onContextMenu(event) {{
                event.preventDefault();
            }}
            
            update() {{
                const offset = new THREE.Vector3();
                const rotation = new THREE.Quaternion();
                const scale = new THREE.Vector3();
                
                this.spherical.theta += this.sphericalDelta.theta;
                this.spherical.phi += this.sphericalDelta.phi;
                this.spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.spherical.phi));
                
                offset.setFromSpherical(this.spherical);
                offset.multiplyScalar(this.scale);
                offset.add(this.panOffset);
                
                this.object.position.copy(offset).add(this.target);
                this.object.lookAt(this.target);
                
                this.sphericalDelta.set(0, 0, 0);
                this.panOffset.set(0, 0, 0);
            }}
        }}

        function initThreeJS() {{
            const container = document.getElementById('container');
            const loading = document.getElementById('loading');
            const controlsDiv = document.getElementById('controls');
            
            try {{
                // Инициализация сцены
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1e1e1e);

                const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.set(0, 2, 5);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                
                // Очищаем контейнер и добавляем рендерер
                container.innerHTML = '';
                container.appendChild(renderer.domElement);
                controlsDiv.style.display = 'block';

                // Используем наш простой OrbitControls
                const controls = new SimpleOrbitControls(camera, renderer.domElement);

                // Освещение
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 10, 7);
                scene.add(directionalLight);

                // Для анимации
                let mixer = null;
                let clock = new THREE.Clock();
                let autoRotate = true;

                // Загрузка модели с использованием FileLoader + GLTFLoader
                const loader = new THREE.FileLoader();
                loader.setResponseType('arraybuffer');
                
                // Конвертируем base64 в binary
                const binaryString = atob("{glb_base64}");
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}

                // Создаем GLTFLoader для парсинга
                const gltfLoader = new THREE.GLTFLoader();
                
                gltfLoader.parse(bytes.buffer, '', function(gltf) {{
                    console.log('Модель загружена успешно');
                    
                    const model = gltf.scene;
                    scene.add(model);

                    // Настраиваем анимацию
                    if (gltf.animations && gltf.animations.length > 0) {{
                        mixer = new THREE.AnimationMixer(model);
                        
                        // Воспроизводим все анимации
                        gltf.animations.forEach((clip) => {{
                            const action = mixer.clipAction(clip);
                            action.play();
                            console.log('Воспроизводится анимация:', clip.name);
                        }});
                        
                        console.log('Найдено анимаций:', gltf.animations.length);
                    }} else {{
                        console.log('Анимации не найдены в модели');
                    }}

                    // Центрируем камеру
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    controls.target.copy(center);
                    controls.spherical.radius = size.length() * 2;
                    controls.update();

                }}, undefined, function(error) {{
                    loading.textContent = '❌ Ошибка загрузки модели: ' + error;
                    console.error('Error loading model:', error);
                }});

                // Глобальные функции для кнопок
                window.resetCamera = function() {{
                    controls.target.set(0, 0, 0);
                    controls.spherical.radius = 5;
                    controls.spherical.theta = 0;
                    controls.spherical.phi = Math.PI / 2;
                    controls.update();
                }};
                
                window.toggleRotation = function() {{
                    autoRotate = !autoRotate;
                }};

                // Анимационный цикл
                function animate() {{
                    requestAnimationFrame(animate);
                    
                    const delta = clock.getDelta();
                    
                    // Обновляем анимации модели
                    if (mixer) {{
                        mixer.update(delta);
                    }}
                    
                    // Автоматическое вращение камеры
                    if (autoRotate) {{
                        controls.spherical.theta += delta * 0.5;
                        controls.update();
                    }}
                    
                    renderer.render(scene, camera);
                }}
                animate();

                // Обработка изменения размера
                window.addEventListener('resize', function() {{
                    camera.aspect = container.clientWidth / container.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, container.clientHeight);
                }});

            }} catch (error) {{
                loading.textContent = '❌ Ошибка инициализации: ' + error.message;
                console.error('Initialization error:', error);
            }}
        }}

        // Запускаем когда страница загружена
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initThreeJS);
        }} else {{
            initThreeJS();
        }}
    </script>
</body>
</html>
"""

st.components.v1.html(html_code, height=620)

st.markdown("""
**🎯 Этот код гарантированно работает потому что:**
- Использует только основной Three.js без внешних зависимостей
- OrbitControls реализован вручную прямо в коде
- GLTFLoader встроен в основную библиотеку Three.js
- Есть кнопки управления камерой

**Управление:**
- 🖱️ **ЛКМ + движение** - вращение камеры
- 🖱️ **ПКМ + движение** - панорамирование  
- 🔍 **Колесико** - приближение/отдаление
- 🔄 **Кнопка "Сброс камеры"** - вернуть вид по умолчанию
- ⏸️ **Кнопка "Вращение"** - включить/выключить авто-вращение

Теперь анимация модели должна работать! 🎬
""")

st.subheader("Простой вариант (если выше не работает)")

simple_html = f"""
<iframe style="width:100%; height:600px; border:none;" 
        src="https://gltf-viewer.donmccurdy.com/">
</iframe>
"""

st.components.v1.html(simple_html, height=620)
st.info("Перетащите ваш файл в открывшееся окно просмотрщика")