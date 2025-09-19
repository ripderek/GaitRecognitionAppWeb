#ARCHIVO DE CONFIGURACION DONDE SE GUARDAN TODAS LAS VARIBALES CON SUS VALORES PARA IMPORTARLOS EN TODOS LOS ARCHIVOS
#Y FUNCIONES

from collections import defaultdict, deque
import math
import os
import cv2
from controllers.EvaluacionesController import InsertarResultadosVideos
from joblib import load
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
#import tensorflow as tf

# Índices de los puntos del rostro en MediaPipe Pose
#hasta el punto 9 son del rostro, desde el 17 son puntos de las manos que de momento son innecesarios
# el punto 0 es el centro de la cabeza para tomar como referencia el medio
# 18 y 19 son puntos  mano
#29 y 30 talones
puntos_rostro = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  17,18, 19, 20, 22, 21 ,29, 30}

#resolucion de video
ANCHO = 960
ALTO = 540

#resolucion para normalizar los videos sin importar la resulicion original de los videos
#NORMALIZADO_ANCHO = 256
#NORMALIZADO_ALTO = 256

NORMALIZADO_ANCHO = 200  #600
NORMALIZADO_ALTO = 400

# Historial para suavizado
history = defaultdict(lambda: deque(maxlen=10))  # Últimos 5 frames

padding = 0.3 #margen del cuadro con respecto a la persona identificada

#DELAY PARA VER LOS VIDEOS
delay =1

tamano_texto_identificacion =4.5

precision_identificacion = 60  #incialmente en 70 ----> si es neceario colocarlo en 0



#suavizar el fecto de distorcion
def suavizar_landmark(idx, x, y):
    history[idx].append((x, y))
    # Promedio
    x_avg = sum(p[0] for p in history[idx]) / len(history[idx])
    y_avg = sum(p[1] for p in history[idx]) / len(history[idx])
    return x_avg, y_avg

def calcular_distancias(puntoA_X,puntoA_Y, puntoB_X, puntoB_Y):
    return  math.sqrt((puntoB_X - puntoA_X)**2 + (puntoB_Y - puntoA_Y)**2)

def distancia_euclidiana(x_A, x_B, y_A, y_B):
    return math.sqrt((x_A - x_B)**2 + (y_A - y_B)**2)


def obtener_escala_x(min_x,escala_x,punto):
    return int((punto - min_x) * escala_x)

def obtener_escala_y(min_y,escala_y,punto):
    return int((punto - min_y) * escala_y)

def obtener_promedio(vector_distancia):
    return sum(vector_distancia) / len(vector_distancia)

def obtener_desviacion(vector_distancia):
    promedio = obtener_promedio(vector_distancia)
    varianza = sum((x - promedio) ** 2 for x in vector_distancia) / len(vector_distancia)
    desviacion = math.sqrt(varianza)
    return desviacion

def AbrirArchivoResultados(texto):
    ruta_archivo = os.path.join("Resultados.txt")
    with open(ruta_archivo, "a") as f:
        f.write(f"{texto}\n")


#mejorar el frame pero con rescalador
def mejorar_frame(frame):
    # 1. Reducir tamaño para acelerar filtrado y detección
    scale_factor = 0.8  # ajustar (0.5 = mitad tamaño)
    small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # 2. Reducción de ruido rápida
    small = cv2.GaussianBlur(small, (3, 3), 0)

    # 3. Mejorar contraste
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.6, tileGridSize=(1, 1))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    small = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Nitidez ligera
    #kernel_sharpening = np.array([
       # [0, -0.25, 0],
      #  [-0.25, 2, -0.25],
     #   [0, -0.25, 0]
    #])
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    small = cv2.filter2D(small, -1, kernel_sharpening)

    # 5. Devuelve el frame reducido y mejorado (NO vuelvas a tamaño original)
    return small, scale_factor

#funcion para mejorar los frames para ayudar a la deteccion de la persona
def mejorar_frame_solo_filtro(frame):
    # 1. Reducir tamaño para acelerar filtrado
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # 2. Reducción de ruido más rápida que bilateral
    #small = cv2.GaussianBlur(small, (3, 3), 0)

    # 3. Mejorar contraste con CLAHE ligero
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    small = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Nitidez ligera
    kernel_sharpening = np.array([
        [0, -0.25, 0],
        [-0.25, 2, -0.25],
        [0, -0.25, 0]
    ])
    #small = cv2.filter2D(small, -1, kernel_sharpening)

    # 5. Volver al tamaño original
    frame = cv2.resize(small, (frame.shape[1], frame.shape[0]))
    return frame

#primero guardar en la base de datos y luego guardar en un archivo txt para visualizacion
def GuardarResultados(participante, escenario, numero, orientacion, evaluacionID, evaluacionID2,
                      VP_Contador, FP_Contador, PI_Contador):
    print(f"\n ---------------------Resultados de la predicción-----------------------------------------")
    delimitador = "-----------------------------------"
    
    encabezado = f"Resultados de la predicción para el participante {participante}, escenario {escenario}, video {numero}, orientación {orientacion}"
    print(encabezado)

    print(f"(VP)= {VP_Contador}")
    print(f"(FP)= {FP_Contador}")
    print(f"(PI)= {PI_Contador}")

    suma = PI_Contador + VP_Contador
    print(f"Resultados de Precisión de Identificación (PI) + (VP)= {suma}")

    try:
        # cálculos que pueden fallar por división entre 0
        pc = VP_Contador / (VP_Contador + FP_Contador) * 100
        PC_I = suma / (suma + FP_Contador) * 100

        print(f"PC-> {pc:.2f}%")
        print(f"PC_I-> {PC_I:.2f}%")

        # guardar el registro en la base de datos solo si todo salió bien
        InsertarResultadosVideos(
            numero, orientacion, escenario,
            VP_Contador, FP_Contador, PI_Contador,
            suma, pc, PC_I,
            evaluacionID, evaluacionID2
        )
    
    except ZeroDivisionError:
        print("⚠️ Error de cálculo: división por cero. No se guardará el resultado en la BD.")
    except Exception as e:
        print(f"⚠️ Ocurrió un error inesperado durante el cálculo: {e}. No se guardará el resultado en la BD.")





def predecir_persona_desde_vectores(vectores_distancia: dict, orientacion):
    """
    vectores_distancia: diccionario con claves tipo "32_31" y valores tipo lista con las 25 distancias
    """

    # Cargar modelo
    #modelo = load(modelo_entrenado)
    modelo = load(f"Modelos/RF/modelo_{orientacion}.joblib")

    # Construir el vector de entrada ordenadamente
    vector = []
    for clave in sorted(vectores_distancia.keys()):
        distancias = vectores_distancia[clave]
        promedio = obtener_promedio(distancias)
        desviacion = obtener_desviacion(distancias)
        #print(f"{clave} => {promedio}, {desviacion}")
        vector.extend([promedio, desviacion])

    # Realizar la predicción
    prediccion = modelo.predict([vector])[0]
    probabilidades = modelo.predict_proba([vector])[0]

    # Mostrar resultados
    #print(f"\n Persona predicha: {prediccion}")
    #print(" Probabilidades:")
    #for persona, prob in zip(modelo.classes_, probabilidades):
        #print(f"- {persona}: {prob * 100:.2f}%")

    # Obtener la probabilidad correspondiente a la persona predicha
    indice = list(modelo.classes_).index(prediccion)
    probabilidad_predicha = round(probabilidades[indice] * 100, 2)

    #return prediccion, probabilidad_predicha, dict(zip(modelo.classes_, [round(p * 100, 2) for p in probabilidades]))
    
    #devolver en 2 decimales
    probabilidades_redondeadas = dict(
    zip(modelo.classes_, [round(p * 100, 2) for p in probabilidades])
    )
    probabilidades_formato = {k: f"{v:.2f}" for k, v in probabilidades_redondeadas.items()}
    return prediccion, probabilidad_predicha,probabilidades_formato

"""
def predecir_persona_desde_vectores_tf(vectores_distancia: dict, orientacion):
  
    #Realiza una predicción de persona usando el modelo entrenado en TensorFlow.
    #vectores_distancia: diccionario con claves tipo "32_31" y valores tipo lista con las 25 distancias


    # Rutas de los modelos
    model_path =   f"MODELOS/MLP/mlp_model_{orientacion}.h5"
    encoder_path = f"MODELOS/MLP/label_encoder_{orientacion}.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("No se encontró el modelo o el encoder. Entrena el modelo primero.")

    # Cargar modelo y encoder
    modelo = tf.keras.models.load_model(model_path)
    label_encoder = load(encoder_path)

    # Construir el vector de entrada ordenadamente
    vector = []
    for clave in sorted(vectores_distancia.keys()):
        distancias = vectores_distancia[clave]
        promedio = obtener_promedio(distancias)
        desviacion = obtener_desviacion(distancias)
        vector.extend([promedio, desviacion])

    # Convertir a forma que TensorFlow entienda
    vector = np.array(vector, dtype="float32").reshape(1, -1)

    # Realizar la predicción
    probabilidades = modelo.predict(vector)[0]

    # Índice con mayor probabilidad
    indice_predicho = np.argmax(probabilidades)
    prediccion = label_encoder.inverse_transform([indice_predicho])[0]

    # Probabilidad asociada a la predicción
    probabilidad_predicha = round(probabilidades[indice_predicho] * 100, 2)

    # Diccionario con todas las probabilidades
    #probabilidades_dict = dict(zip(label_encoder.classes_, [round(p * 100, 2) for p in probabilidades]))
    probabilidades_dict = {
    persona: float(f"{prob*100:.2f}") 
        for persona, prob in zip(label_encoder.classes_, probabilidades)
    }
    # Mostrar resultados
    #for persona, prob in probabilidades_dict.items():
        #print(f"- {persona}: {prob:.2f}%")

    return prediccion, probabilidad_predicha, probabilidades_dict
    

#el modelo MLP pero todo en uno ALL
def predecir_persona_desde_vectores_tf_all(vectores_distancia: dict, orientacion):
   
    #Realiza una predicción de persona usando el modelo entrenado en TensorFlow.
    #vectores_distancia: diccionario con claves tipo "32_31" y valores tipo lista con las 25 distancias
  

    # Rutas de los modelos
    model_path =   f"MODELOS/MLP/mlp_model_ALL.h5"
    encoder_path = f"MODELOS/MLP/label_encoder_All.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("No se encontró el modelo o el encoder. Entrena el modelo primero.")

    # Cargar modelo y encoder
    modelo = tf.keras.models.load_model(model_path)
    label_encoder = load(encoder_path)

    # Construir el vector de entrada ordenadamente
    vector = []
    for clave in sorted(vectores_distancia.keys()):
        distancias = vectores_distancia[clave]
        promedio = obtener_promedio(distancias)
        desviacion = obtener_desviacion(distancias)
        vector.extend([promedio, desviacion])

    # Convertir a forma que TensorFlow entienda
    vector = np.array(vector, dtype="float32").reshape(1, -1)

    # Realizar la predicción
    probabilidades = modelo.predict(vector)[0]

    # Índice con mayor probabilidad
    indice_predicho = np.argmax(probabilidades)
    prediccion = label_encoder.inverse_transform([indice_predicho])[0]

    # Probabilidad asociada a la predicción
    probabilidad_predicha = round(probabilidades[indice_predicho] * 100, 2)

    # Diccionario con todas las probabilidades
    #probabilidades_dict = dict(zip(label_encoder.classes_, [round(p * 100, 2) for p in probabilidades]))
    probabilidades_dict = {
    persona: float(f"{prob*100:.2f}") 
        for persona, prob in zip(label_encoder.classes_, probabilidades)
    }
    # Mostrar resultados
    #for persona, prob in probabilidades_dict.items():
        #print(f"- {persona}: {prob:.2f}%")

    return prediccion, probabilidad_predicha, probabilidades_dict
"""
    


def RealizarPrediccion(vectores,orientacion,opcion):
    if opcion == "RF":
        #llamar a la funcion que tiene el modelo RF
        prediccion,probabilidad_predicha, probabilidades = predecir_persona_desde_vectores(vectores,orientacion)
        return  prediccion,probabilidad_predicha, probabilidades
    #elif opcion == "MLP_ALL":
         #llamar a la funcion que tiene el modelo MLP_ALL
        #prediccion,probabilidad_predicha, probabilidades = predecir_persona_desde_vectores_tf_all(vectores,orientacion)
        #return  prediccion,probabilidad_predicha, probabilidades
    #else:
        #llamar a la funcion que tiene el modelo MLP
        #prediccion,probabilidad_predicha, probabilidades = predecir_persona_desde_vectores_tf(vectores,orientacion)
        #return  prediccion,probabilidad_predicha, probabilidades

#funcion para recorrer la lista de participantes de las carpetas
def obtener_participantes(ruta_base="Participantes"):
    # Listar el contenido de la carpeta
    participantes = [
        nombre for nombre in os.listdir(ruta_base)
        if os.path.isdir(os.path.join(ruta_base, nombre))
    ]
    return participantes