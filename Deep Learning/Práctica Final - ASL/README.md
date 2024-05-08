# Modelo de Redes Neuronales Convolucionales para ASL (American Sign Language)

**Autores:**
* Luis Bueno Archaga
* Yago Tobio Souto

**Enlace a Google Drive con todos los archivos utilizados en el proyecto:** https://drive.google.com/drive/folders/1GqzC3_wMGsMVbtiheRbnX0BFpszrV4b_?usp=sharing

## Descripción del Proyecto
Este proyecto constituye la práctica final de la asignatura de Business Intelligence. Está dividido en dos partes principales:

1. **Modelo de CNN para reconocimiento del abecedario en ASL:**
   - Desarrollo de un modelo de Redes Neuronales Convolucionales (CNN) para el reconocimiento de las letras del alfabeto en Lengua de Signos Americana (ASL).
   - Análisis de la precisión (accuracy) y evaluación de diferentes características del modelo.

2. **Aplicación práctica del modelo con OpenCV:**
   - Implementación de una aplicación que utiliza OpenCV para capturar imágenes mediante una webcam.
   - El modelo procesa las imágenes capturadas para predecir las letras del ASL indicadas por el usuario.

## Fuentes de Datos
Los datasets utilizados para entrenar y validar nuestro modelo han sido obtenidos de Kaggle:

- **ASL Alphabet:** [Acceder al Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
  - Contiene 3.000 imágenes por letra, proporcionando una amplia variedad de datos para un entrenamiento efectivo.
  
- **American Sign Language Dataset:** [Acceder al Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
  - Compuesto por 700 imágenes por letra, este dataset complementa al anterior permitiendo un entrenamiento más robusto y diversificado.

## Implementación y Tecnologías
El proyecto ha sido implementado utilizando Python, con el apoyo de librerías especializadas como TensorFlow para el desarrollo del modelo de CNN y OpenCV para la captura y procesamiento de imágenes en tiempo real.