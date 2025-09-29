# import streamlit as st
# import pyvista as pv
# from stpyvista import stpyvista
# import tempfile
# import os

# # pv.start_xvfb()  # Для Linux
# pv.set_jupyter_backend('static')

# def view_ply_with_pyvista(ply_file):
#     try:
#         mesh = pv.read(ply_file)
        
#         plotter = pv.Plotter(window_size=[1080, 1920])
#         plotter.add_mesh(mesh, color='lightblue', show_edges=True)
#         plotter.background_color = 'white'
        
#         return plotter
#     except Exception as e:
#         st.error(f"Ошибка загрузки PLY: {e}")
#         return None

# def main():
#     st.title("🔮 3D PLY Viewer")
    
#     uploaded_file = st.file_uploader("Загрузите PLY файл", type=['ply'])
    
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_path = tmp_file.name

#         plotter = view_ply_with_pyvista(tmp_path)
#         if plotter:
#             stpyvista(plotter, key="pyvista")
        
#         os.unlink(tmp_path)

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import plotly.graph_objects as go
# import plotly.express as px
# import tempfile
# import numpy as np
# from plyfile import PlyData
# import os 

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

# def main():
#     st.title("📊 3D Point Cloud Viewer")
#     st.markdown("Загрузите PLY файл для просмотра 3D облака точек")
    
#     uploaded_file = st.file_uploader("Выберите PLY файл", type=['ply'])
    
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_path = tmp_file.name
        
#         x, y, z, colors = load_ply_file(tmp_path)
        
#         if x is not None:
#             fig = go.Figure(data=[go.Scatter3d(
#                 x=x,
#                 y=y, 
#                 z=z,
#                 mode='markers',
#                 marker=dict(
#                     size=2,
#                     color=colors,
#                     colorscale='Viridis',
#                     opacity=0.8
#                 )
#             )])

#             fig.update_layout(
#                 scene=dict(
#                     xaxis_title='X',
#                     yaxis_title='Y',
#                     zaxis_title='Z',
#                     aspectmode='data'
#                 ),
#                 width=800,
#                 height=600,
#                 title="3D Point Cloud"
#             )
            
#             st.plotly_chart(fig, use_container_width=True)
            
#             st.info(f"**Информация:** {len(x)} точек")
        
#         os.unlink(tmp_path)

# if __name__ == "__main__":
#     main()


import streamlit as st
import base64
import tempfile
import os

def create_viewer_html(ply_file_path=None):
    """Создает HTML с встроенным Gaussian Splatting viewer"""
    
    # Читаем ваш main.js файл
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
    st.set_page_config(page_title="3D Gaussian Splatting", layout="wide")
    
    st.title("🎯 3D Gaussian Splatting Viewer")
    st.markdown("Просмотр PLY файлов с технологией Gaussian Splatting")
    
    # Загрузка PLY файла
    uploaded_file = st.file_uploader("Загрузите PLY файл", type=['ply'])
    
    if uploaded_file:
        # Сохраняем файл временно
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            ply_path = tmp_file.name
        
        st.success(f"Файл {uploaded_file.name} загружен!")
        
        # Создаем HTML с viewer
        html_content = create_viewer_html()
        
        # Показываем в Streamlit
        st.components.v1.html(html_content, height=600)
        
        # Инструкция
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
        
        # Очистка
        os.unlink(ply_path)

if __name__ == "__main__":
    main()