import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Wound Analyzer App")
st.write("Sube una imagen de una herida para analizar el área y el perímetro.")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir la imagen a un array de numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Mostrar la imagen subida
    st.image(img, channels="BGR", caption="Imagen subida", use_column_width=True)
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Procesamiento para detectar contornos de la herida
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcular área y perímetro de la herida
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Conversión a cm (suponiendo 1 píxel = 0.026 cm, ajustar según referencia)
        pixel_to_cm = 0.026
        area_cm2 = area * (pixel_to_cm ** 2)
        perimeter_cm = perimeter * pixel_to_cm
        
        # Dibujar contornos sobre la imagen original
        cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
        
        # Mostrar la imagen con los contornos
        st.image(img, channels="BGR", caption="Imagen procesada", use_column_width=True)
        
        # Mostrar resultados
        st.write(f"Área: {area_cm2:.2f} cm²")
        st.write(f"Perímetro: {perimeter_cm:.2f} cm")
