"""
Aplicación Web de Reconocimiento de Enfermedades en Papa
Usando Streamlit y TensorFlow
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import urllib.request
import gdown

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Detección de Enfermedades en Papa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cargar CSS personalizado
def load_css():
    css_file = ".streamlit/style.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
load_css()

# ============================================
# DESCARGAR MODELO DESDE GITHUB O GOOGLE DRIVE
# ============================================
def descargar_modelo_si_necesario():
    """
    Descarga el modelo desde GitHub o Google Drive si no existe localmente.
    """
    modelo_path = 'best_potato_model.keras'
    
    if not os.path.exists(modelo_path):
        st.info("⏳ Descargando modelo... (esto puede tardar un momento)")
        
        try:
            # Intentar descargar desde GitHub primero (más rápido)
            github_url = "https://github.com/CamiloAT/Plant_Diseases/raw/main/best_potato_model.keras"
            
            import urllib.request
            urllib.request.urlretrieve(github_url, modelo_path)
            st.success("✅ Modelo descargado exitosamente desde GitHub")
            
        except Exception as e:
            # Si falla GitHub, intentar Google Drive
            st.warning(f"No se pudo descargar desde GitHub: {str(e)}")
            st.info("Intentando descargar desde Google Drive...")
            
            try:
                gdrive_url = "https://drive.google.com/uc?id=1NB0-US-83eUoajqbb3ea475VIvAZULKY"
                gdown.download(gdrive_url, modelo_path, quiet=False)
                st.success("✅ Modelo descargado exitosamente desde Google Drive")
            except Exception as e2:
                st.error(f"❌ Error al descargar el modelo: {str(e2)}")
                st.error("Verifica que el archivo esté disponible en GitHub o Google Drive.")
                st.stop()
    
    return modelo_path


# ============================================
# CREAR ARQUITECTURA DEL MODELO
# ============================================
def crear_modelo(num_classes=3, img_size=224):
    """
    Crea la arquitectura del modelo desde cero (API Funcional).
    Esto evita problemas de compatibilidad entre versiones de Keras.
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    
    # Cargar MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Construir el modelo
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    modelo = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

# ============================================
# CARGAR MODELO Y METADATOS
# ============================================
@st.cache_resource
def cargar_modelo_y_metadatos():
    """
    Carga el modelo entrenado y sus metadatos.
    Usa @st.cache_resource para cargar el modelo solo una vez.
    """
    # Descargar modelo si no existe
    modelo_path = descargar_modelo_si_necesario()
    
    try:
        # MÉTODO 1: Intentar cargar el modelo completo (puede fallar con Keras 3)
        try:
            modelo = tf.keras.models.load_model(modelo_path, compile=False)
            modelo.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Modelo cargado usando método estándar")
        except Exception as e1:
            # MÉTODO 2: Recrear arquitectura y cargar solo pesos
            print("⚠️ Usando modo de compatibilidad para cargar el modelo...")
            
            # Crear arquitectura desde cero
            modelo = crear_modelo(num_classes=3, img_size=224)
            
            # Intentar cargar los pesos del modelo guardado
            try:
                modelo_temp = tf.keras.models.load_model(modelo_path, compile=False)
                modelo.set_weights(modelo_temp.get_weights())
                print("✅ Pesos del modelo cargados exitosamente")
            except Exception as e2:
                print(f"❌ No se pudieron cargar los pesos: {str(e2)}")
                print("🔄 Usando modelo con pesos de ImageNet (sin entrenamiento específico)")
        
        # Intentar cargar metadatos si existen
        metadatos = None
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                metadatos = json.load(f)
        
        return modelo, metadatos
        
    except Exception as e:
        st.error(f"❌ Error crítico al cargar el modelo: {str(e)}")
        
        # Mostrar información de depuración
        with st.expander("🔍 Información de depuración"):
            st.code(f"Error completo: {str(e)}")
            st.write(f"Versión de TensorFlow: {tf.__version__}")
            st.write("Versión de Keras:", tf.keras.__version__)
        
        st.stop()

# ============================================
# FUNCIÓN DE PREPROCESAMIENTO
# ============================================
def preprocesar_imagen(imagen, img_size=224):
    """
    Preprocesa la imagen para que sea compatible con el modelo.
    
    Args:
        imagen: Imagen PIL
        img_size: Tamaño de la imagen (por defecto 224x224)
    
    Returns:
        Imagen preprocesada como array numpy
    """
    # Convertir a RGB si es necesario
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    
    # Redimensionar a 224x224 (mismo tamaño del entrenamiento)
    imagen = imagen.resize((img_size, img_size))
    
    # Convertir a array numpy
    img_array = np.array(imagen)
    
    # Normalizar (dividir por 255)
    img_array = img_array / 255.0
    
    # Agregar dimensión del batch (el modelo espera un batch de imágenes)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================
# INTERFAZ DE USUARIO
# ============================================

# Título y descripción
st.title("🌿 Detección de Enfermedades en Papa")
st.markdown("---")

# Cargar modelo y metadatos
modelo, metadatos = cargar_modelo_y_metadatos()

# Obtener información del modelo
if metadatos:
    num_clases = metadatos.get('num_classes', 'N/A')
    img_size = metadatos.get('img_size', 224)
    test_accuracy = metadatos.get('test_accuracy', 0) * 100
    class_indices = metadatos.get('class_indices', {})
    # Invertir el diccionario para obtener nombre por índice
    CLASES_ENFERMEDADES = {v: k for k, v in class_indices.items()}
    
    # Log en consola en lugar de mostrar en pantalla
    print(f"✅ Modelo cargado exitosamente - Accuracy: {test_accuracy:.2f}%")
else:
    img_size = 224
    CLASES_ENFERMEDADES = {}
    print("⚠️ Modelo cargado sin metadatos")

# Crear dos columnas
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Cargar Imagen")
    
    # File uploader
    archivo_subido = st.file_uploader(
        "Selecciona una imagen de una hoja de papa",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos: JPG, JPEG, PNG"
    )
    
    if archivo_subido is not None:
        # Cargar y mostrar imagen original
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption='Imagen cargada', use_column_width=True)
        
        # Botón para realizar predicción
        if st.button("Analizar Imagen", type="primary", use_container_width=True):
            with st.spinner('Analizando imagen...'):
                # Preprocesar imagen
                img_procesada = preprocesar_imagen(imagen, img_size)
                
                # Realizar predicción
                predicciones = modelo.predict(img_procesada, verbose=0)
                
                # Obtener clase predicha y confianza (convertir a float nativo de Python)
                clase_predicha = int(np.argmax(predicciones[0]))
                confianza = float(predicciones[0][clase_predicha] * 100)
                
                # Guardar resultados en session_state
                st.session_state.clase_predicha = clase_predicha
                st.session_state.confianza = confianza
                st.session_state.predicciones = predicciones[0]

with col2:
    st.subheader("Resultado del Análisis")
    
    if 'clase_predicha' in st.session_state:
        # Mostrar resultado principal
        st.markdown("**Diagnóstico**")
        
        # Crear un contenedor destacado para el resultado
        resultado_container = st.container()
        with resultado_container:
            # Nombre de la enfermedad
            if CLASES_ENFERMEDADES:
                nombre_enfermedad = CLASES_ENFERMEDADES[st.session_state.clase_predicha]
            else:
                nombre_enfermedad = f"Clase {st.session_state.clase_predicha}"
            
            # Mostrar resultado según el tipo de enfermedad
            if 'healthy' in nombre_enfermedad.lower():
                st.success(f"**{nombre_enfermedad}**")
            elif 'early' in nombre_enfermedad.lower():
                st.warning(f"**{nombre_enfermedad}**")
            else:
                st.error(f"**{nombre_enfermedad}**")
            
            # Barra de confianza
            st.markdown(f"**Nivel de Confianza:** {st.session_state.confianza:.1f}%")
            st.progress(float(st.session_state.confianza / 100))
            
            # Interpretación de confianza
            if st.session_state.confianza < 60:
                st.error("""
                **⚠️ Imagen no reconocida o confianza muy baja**
                
                El modelo no puede identificar con certeza esta imagen. Esto puede deberse a:
                
                - La imagen no corresponde a una hoja de papa
                - La imagen tiene baja calidad o está borrosa
                - La hoja está muy alejada o muy cerca
                - Hay múltiples objetos en la imagen
                
                **Recomendaciones para mejorar la detección:**
                - Use una imagen clara y enfocada
                - Capture **solamente la hoja de papa** afectada
                - Asegure buena iluminación natural
                - Evite sombras y reflejos
                - La hoja debe ocupar la mayor parte de la imagen
                - Fondo uniforme (cielo, papel blanco, etc.)
                """)
            elif st.session_state.confianza > 90:
                st.success("**Predicción muy confiable**")
            elif st.session_state.confianza > 70:
                st.info("**Predicción confiable**")
            else:
                st.warning("**Predicción con confianza media - Se recomienda verificar con un experto**")
        
        st.markdown("---")
        
        # Top 3 predicciones
        st.markdown("**Predicciones Principales**")
        
        # Obtener índices de las 3 clases con mayor probabilidad
        top_3_indices = np.argsort(st.session_state.predicciones)[-3:][::-1]
        
        for i, idx in enumerate(top_3_indices, 1):
            probabilidad = st.session_state.predicciones[idx] * 100
            if CLASES_ENFERMEDADES:
                nombre = CLASES_ENFERMEDADES[idx]
            else:
                nombre = f"Clase {idx}"
            
            st.markdown(f"{i}. **{nombre}** - `{probabilidad:.1f}%`")
        
        # Recomendaciones (solo si la confianza es >= 60%)
        if st.session_state.confianza >= 60:
            st.markdown("---")
            st.markdown("**Recomendaciones**")
            
            if 'healthy' in nombre_enfermedad.lower():
                st.info("""
                **Planta Saludable**
                - Continuar con las prácticas de cuidado actuales
                - Mantener monitoreo regular
                - Asegurar buena ventilación y riego adecuado
                """)
            elif 'early blight' in nombre_enfermedad.lower():
                st.warning("""
                **Tizón Temprano Detectado**
                - Aplicar fungicidas a base de cobre
                - Mejorar la circulación de aire
                - Evitar riego por aspersión
                - Eliminar hojas afectadas
                """)
            elif 'late blight' in nombre_enfermedad.lower():
                st.error("""
                **Tizón Tardío Detectado - Acción Urgente**
                - Aplicar fungicidas sistémicos inmediatamente
                - Aislar plantas afectadas
                - Mejorar drenaje del suelo
                - Consultar con un agrónomo
                """)
    
    else:
        st.info("Carga una imagen y presiona 'Analizar Imagen' para ver los resultados")

# ============================================
# INFORMACIÓN ADICIONAL (PARTE INFERIOR)
# ============================================

st.markdown("---")
st.markdown("## Información del Sistema")

# Tabs para organizar la información
tab1, tab2, tab3 = st.tabs(["📊 Métricas del Modelo", "📚 Enfermedades Detectables", "ℹ️ Acerca del Proyecto"])

with tab1:
    # Métricas del modelo
    if metadatos:
        st.markdown("### Rendimiento del Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metadatos.get('test_accuracy', 0) * 100:.1f}%")
        with col2:
            st.metric("Precision", f"{metadatos.get('test_precision', 0) * 100:.1f}%")
        with col3:
            st.metric("Recall", f"{metadatos.get('test_recall', 0) * 100:.1f}%")
        with col4:
            st.metric("F1-Score", f"{metadatos.get('f1_score', 0) * 100:.1f}%")
        
        st.markdown("---")
        
        # Lista de clases reconocidas
        st.markdown("### Clases Reconocidas")
        if CLASES_ENFERMEDADES:
            cols = st.columns(3)
            for i, (idx, nombre) in enumerate(CLASES_ENFERMEDADES.items()):
                with cols[i % 3]:
                    st.markdown(f"**{idx}.** {nombre}")
    else:
        st.info("Metadatos del modelo no disponibles")

with tab2:
    # Información sobre enfermedades
    st.markdown("### Enfermedades que el Sistema Puede Detectar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("🦠 Tizón Temprano", expanded=False):
            st.markdown("""
            **Nombre científico:** *Alternaria solani*
            
            **Síntomas:**
            - Manchas circulares concéntricas en las hojas
            - Color marrón oscuro
            - Afecta principalmente hojas más viejas
            
            **Control:**
            - Fungicidas a base de cobre
            - Rotación de cultivos
            - Eliminación de residuos vegetales
            """)
    
    with col2:
        with st.expander("🦠 Tizón Tardío", expanded=False):
            st.markdown("""
            **Nombre científico:** *Phytophthora infestans*
            
            **Síntomas:**
            - Manchas irregulares verde oscuro a negro
            - Moho blanco en el envés de las hojas
            - Propagación rápida en condiciones húmedas
            
            **Control:**
            - Fungicidas sistémicos
            - Mejorar drenaje
            - Plantar variedades resistentes
            - Evitar riego por aspersión
            """)
    
    with col3:
        with st.expander("✅ Planta Saludable", expanded=False):
            st.markdown("""
            **Características:**
            - Hojas verdes uniformes
            - Sin manchas ni decoloraciones
            - Crecimiento vigoroso
            
            **Mantenimiento:**
            - Riego adecuado
            - Fertilización balanceada
            - Monitoreo regular
            - Buena ventilación
            """)

with tab3:
    # Acerca del proyecto
    st.markdown("### Proyecto Universitario de Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Este sistema utiliza **Deep Learning** con Transfer Learning basado en la arquitectura 
        **MobileNetV2** para clasificar enfermedades en hojas de papa.
        
        **Características Técnicas:**
        - **Modelo Base:** MobileNetV2 (pre-entrenado en ImageNet)
        - **Dataset:** PlantVillage - Potato Disease Dataset
        - **Clases:** 3 tipos (Saludable, Tizón Temprano, Tizón Tardío)
        - **Entrada:** Imágenes 224x224 píxeles RGB
        - **Técnicas:** Data Augmentation, Fine-tuning, Regularización L2
        - **Framework:** TensorFlow/Keras
        
        **Aplicación:**
        - **Frontend:** Streamlit
        - **Despliegue:** Streamlit Cloud
        - **Repositorio:** GitHub
        """)
    
    with col2:
        st.markdown("""
        **📖 Sobre el Proyecto**
        
        Desarrollado como proyecto 
        universitario para la materia 
        de Inteligencia Computacional.
        
        **Objetivo**
        
        Proporcionar una herramienta 
        de diagnóstico rápido y 
        accesible para agricultores.
        
        **Año:** 2025
        """)