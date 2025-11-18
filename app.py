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
import urllib.request  # Para descargar desde GitHub
import gdown  # Para descargar desde Google Drive

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Reconocimiento de Enfermedades en Papa",
    page_icon="🥔",
    layout="wide"
)

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
            st.success("✅ Modelo cargado usando método estándar")
        except Exception as e1:
            # MÉTODO 2: Recrear arquitectura y cargar solo pesos
            st.warning("⚠️ Usando modo de compatibilidad para cargar el modelo...")
            
            # Crear arquitectura desde cero
            modelo = crear_modelo(num_classes=3, img_size=224)
            
            # Intentar cargar los pesos del modelo guardado
            try:
                modelo_temp = tf.keras.models.load_model(modelo_path, compile=False)
                modelo.set_weights(modelo_temp.get_weights())
                st.success("✅ Pesos del modelo cargados exitosamente")
            except Exception as e2:
                st.error(f"❌ No se pudieron cargar los pesos: {str(e2)}")
                st.info("🔄 Usando modelo con pesos de ImageNet (sin entrenamiento específico)")
                # El modelo ya tiene pesos de ImageNet en la base
        
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
st.title("🥔 Reconocimiento de Enfermedades en Papa")
st.markdown("### Sistema de Clasificación Automática usando Deep Learning")
st.markdown("---")

# Información del proyecto
with st.expander("ℹ️ Acerca de este proyecto"):
    st.write("""
    **Proyecto Universitario de Machine Learning**
    
    Este sistema utiliza una Red Neuronal Convolucional (CNN) con Transfer Learning (MobileNetV2) 
    entrenada con el dataset PlantVillage para detectar enfermedades en hojas de papa.
    
    **Características:**
    - 🧠 Modelo: CNN con Transfer Learning (MobileNetV2)
    - 📊 Dataset: PlantVillage - Potato Disease Dataset
    - 🎯 Clases: Enfermedades comunes en plantas de papa
    - 🖼️ Entrada: Imágenes de 224x224 píxeles
    - 📈 Técnicas: Data Augmentation, Fine-tuning, Class Weighting
    """)

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
    
    st.success(f"✅ Modelo cargado exitosamente - Accuracy: {test_accuracy:.2f}%")
    
    if 'class_distribution' in metadatos:
        with st.expander("📊 Información del Dataset"):
            st.write(f"**Total de clases:** {num_clases}")
            st.write(f"**Imágenes de entrenamiento:** {metadatos.get('total_train_samples', 'N/A')}")
            st.write(f"**Imágenes de prueba:** {metadatos.get('total_test_samples', 'N/A')}")
            st.write(f"**Precisión del modelo:** {metadatos.get('test_precision', 0) * 100:.2f}%")
            st.write(f"**Recall del modelo:** {metadatos.get('test_recall', 0) * 100:.2f}%")
            st.write(f"**F1-Score:** {metadatos.get('f1_score', 0) * 100:.2f}%")
else:
    img_size = 224
    CLASES_ENFERMEDADES = {}
    st.warning("⚠️ Modelo cargado sin metadatos. Algunas funciones pueden estar limitadas.")

# Crear dos columnas
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar Imagen")
    
    # File uploader
    archivo_subido = st.file_uploader(
        "Selecciona una imagen de una hoja de papa",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
    
    if archivo_subido is not None:
        # Cargar y mostrar imagen original
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption='Imagen cargada', width='stretch')
        
        # Botón para realizar predicción
        if st.button("🔍 Analizar Hoja de Papa", type="primary", use_container_width=True):
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
    st.subheader("🎯 Resultado del Análisis")
    
    if 'clase_predicha' in st.session_state:
        # Mostrar resultado principal
        st.markdown("### Diagnóstico:")
        
        # Crear un contenedor destacado para el resultado
        resultado_container = st.container()
        with resultado_container:
            # Nombre de la enfermedad
            if CLASES_ENFERMEDADES:
                nombre_enfermedad = CLASES_ENFERMEDADES[st.session_state.clase_predicha]
            else:
                nombre_enfermedad = f"Clase {st.session_state.clase_predicha}"
            
            # Emoji según el tipo de enfermedad
            if 'healthy' in nombre_enfermedad.lower():
                emoji = "✅"
                st.success(f"## {emoji} **{nombre_enfermedad}**")
            elif 'early' in nombre_enfermedad.lower():
                emoji = "⚠️"
                st.warning(f"## {emoji} **{nombre_enfermedad}**")
            else:
                emoji = "🦠"
                st.error(f"## {emoji} **{nombre_enfermedad}**")
            
            # Barra de confianza
            st.markdown(f"**Confianza:** {st.session_state.confianza:.2f}%")
            st.progress(float(st.session_state.confianza / 100))
            
            # Interpretación de confianza
            if st.session_state.confianza > 90:
                st.success("✅ Predicción muy confiable")
            elif st.session_state.confianza > 70:
                st.info("ℹ️ Predicción confiable")
            else:
                st.warning("⚠️ Predicción con baja confianza - Se recomienda verificar con un experto")
        
        st.markdown("---")
        
        # Top 3 predicciones
        st.markdown("### 📊 Top 3 Predicciones:")
        
        # Obtener índices de las 3 clases con mayor probabilidad
        top_3_indices = np.argsort(st.session_state.predicciones)[-3:][::-1]
        
        for i, idx in enumerate(top_3_indices, 1):
            probabilidad = st.session_state.predicciones[idx] * 100
            if CLASES_ENFERMEDADES:
                nombre = CLASES_ENFERMEDADES[idx]
            else:
                nombre = f"Clase {idx}"
            
            col_num, col_nombre, col_prob = st.columns([0.5, 3, 1])
            with col_num:
                st.markdown(f"**{i}.**")
            with col_nombre:
                st.markdown(f"{nombre}")
            with col_prob:
                st.markdown(f"`{probabilidad:.1f}%`")
        
        # Recomendaciones
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones:")
        
        if 'healthy' in nombre_enfermedad.lower():
            st.info("""
            ✅ **Planta saludable detectada**
            - Continúa con las prácticas de cuidado actuales
            - Mantén un monitoreo regular
            - Asegura buena ventilación y riego adecuado
            """)
        elif 'early blight' in nombre_enfermedad.lower():
            st.warning("""
            ⚠️ **Tizón Temprano (Early Blight) detectado**
            - Aplicar fungicidas a base de cobre
            - Mejorar la circulación de aire
            - Evitar riego por aspersión
            - Eliminar hojas afectadas
            """)
        elif 'late blight' in nombre_enfermedad.lower():
            st.error("""
            🦠 **Tizón Tardío (Late Blight) detectado**
            - ⚠️ ACCIÓN URGENTE REQUERIDA
            - Aplicar fungicidas sistémicos inmediatamente
            - Aislar plantas afectadas
            - Mejorar drenaje del suelo
            - Consultar con un agrónomo
            """)
    
    else:
        st.info("👆 Carga una imagen y presiona 'Analizar' para ver los resultados")

# ============================================
# SECCIÓN ADICIONAL: LISTA DE ENFERMEDADES
# ============================================
st.markdown("---")
st.subheader("📋 Enfermedades Reconocidas por el Sistema")

if CLASES_ENFERMEDADES:
    with st.expander(f"Ver todas las clases ({len(CLASES_ENFERMEDADES)})"):
        # Mostrar en 2 columnas
        cols = st.columns(2)
        
        for idx, nombre in CLASES_ENFERMEDADES.items():
            col_idx = idx % 2
            with cols[col_idx]:
                if 'healthy' in nombre.lower():
                    st.markdown(f"✅ **{idx}.** {nombre}")
                else:
                    st.markdown(f"🦠 **{idx}.** {nombre}")
else:
    st.info("ℹ️ Información de clases no disponible. Carga el archivo 'model_metadata.json' para ver las clases.")

# ============================================
# INFORMACIÓN ADICIONAL
# ============================================
st.markdown("---")
st.subheader("📚 Información sobre Enfermedades Comunes en Papa")

with st.expander("🦠 Tizón Temprano (Early Blight)"):
    st.write("""
    **Causado por:** Alternaria solani
    
    **Síntomas:**
    - Manchas circulares concéntricas en las hojas
    - Color marrón oscuro
    - Afecta principalmente hojas más viejas
    
    **Control:**
    - Fungicidas a base de cobre
    - Rotación de cultivos
    - Eliminación de residuos vegetales
    """)

with st.expander("🦠 Tizón Tardío (Late Blight)"):
    st.write("""
    **Causado por:** Phytophthora infestans
    
    **Síntomas:**
    - Manchas irregulares de color verde oscuro a negro
    - Moho blanco en el envés de las hojas
    - Propagación rápida en condiciones húmedas
    
    **Control:**
    - Fungicidas sistémicos
    - Mejorar drenaje
    - Plantar variedades resistentes
    - Evitar riego por aspersión
    """)

with st.expander("✅ Planta Saludable (Healthy)"):
    st.write("""
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

# ============================================
# PIE DE PÁGINA
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🥔 Desarrollado con ❤️ usando TensorFlow, MobileNetV2 y Streamlit</p>
    <p>Proyecto Universitario - Inteligencia Computacional - 2025</p>
    <p>Dataset: PlantVillage - Potato Disease Classification</p>
</div>
""", unsafe_allow_html=True)