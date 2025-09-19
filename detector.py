# detector.py
from ultralytics import YOLO
import cv2
import config
import sys
import os
import mediapipe as mp
#import controllers.MuestrasController as sv
import controllers.EvaluacionesController as sv
import numpy as np
import pandas as pd

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Cargar YOLO versión liviana (n = nano)
model = YOLO("yolov8n.pt")


reproducir_flag = {"estado": True}

#estado para contar los VP -> Verdaderos Positivos
VP = {"contador": 0}
#estado para contar los FP -> Falso
# Positivos
FP = {"contador": 0}
#estado para contrar los PI -> precision_identificacion
PI = {"contador": 0}


selected_id = None  # ID de la persona seleccionada
last_detections = []  # [(id, (x1,y1,x2,y2)), ...]
CONF_THRES = 0.35
contador = 0
label = ""
#vectores de distancias
vector_distancia_32_31 = []
vector_distancia_28_27 =[]
vector_distancia_26_25 =[]
vector_distancia_31_23 =[]
vector_distancia_32_24 =[]
#nuevos vectores 
vector_distancia_16_12 =[]
vector_distancia_15_11 =[]
vector_distancia_32_16 =[]
vector_distancia_31_15 =[]
orientacionString= ''

def RecorteNormalizacion(frame_mejorado, min_y, max_y, min_x, max_x):
    global contador
    # --- Paso 1: padding sobre el bbox ---
    ancho = max_x - min_x
    alto = max_y - min_y

    padding_x = int(ancho * config.padding)
    padding_y = int(alto * config.padding)

    x1 = max(min_x - padding_x, 0)
    y1 = max(min_y - padding_y, 0)
    x2 = min(max_x + padding_x, frame_mejorado.shape[1])
    y2 = min(max_y + padding_y, frame_mejorado.shape[0])

    # --- Paso 2: recorte de la persona ---
    persona_recortada = frame_mejorado[y1:y2, x1:x2]
    if persona_recortada.size == 0:
        return frame_mejorado, [], [], 1, 1, 0, 0, 0, 0

    # --- Paso 3: normalización ---
    persona_normalizada = cv2.resize(persona_recortada, (config.NORMALIZADO_ANCHO, config.NORMALIZADO_ALTO))

    # --- Paso 4: MediaPipe Pose sobre la imagen normalizada ---
    image_rgb = cv2.cvtColor(persona_normalizada, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)

    gris_claro = 250
    fondo_negro = np.full_like(persona_normalizada, fill_value=gris_claro)

    xs, ys = [], []
    escala_x = escala_y = 1
    min_x_, max_x_, min_y_, max_y_ = 0, 0, 0, 0

    if results_pose.pose_landmarks:
        h_norm, w_norm, _ = persona_normalizada.shape
        landmarks = results_pose.pose_landmarks.landmark

        
        for idx, lm in enumerate(landmarks):
            # coordenadas ya están en el espacio normalizado
            x_norm = lm.x * w_norm
            y_norm = lm.y * h_norm
            xs.append(x_norm)
            ys.append(y_norm)

        # Bounding box de landmarks (en el espacio normalizado)
        min_x_, max_x_ = int(min(xs)), int(max(xs))
        min_y_, max_y_ = int(min(ys)), int(max(ys))

        # Escalas relativas al bbox de la persona normalizada
        escala_x = config.NORMALIZADO_ANCHO / (max_x_ - min_x_) if (max_x_ - min_x_) > 0 else 1
        escala_y = config.NORMALIZADO_ALTO / (max_y_ - min_y_) if (max_y_ - min_y_) > 0 else 1

        # Dibujar puntos y líneas (opcional)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            if idx not in config.puntos_rostro:
                cv2.circle(fondo_negro, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(fondo_negro, f'p{idx} ({int(x)},{int(y)})', (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx not in config.puntos_rostro and end_idx not in config.puntos_rostro:
                x1, y1 = int(xs[start_idx]), int(ys[start_idx])
                x2, y2 = int(xs[end_idx]), int(ys[end_idx])
                cv2.line(fondo_negro, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else: 
        contador = 0
        
    
    return fondo_negro, xs, ys, escala_x, escala_y, min_x_, max_x_, min_y_, max_y_


def distancia_puntos(min_x,escala_x,min_y,escala_y,xs1,xs2,ys1,ys2):
    x1_1 = config.obtener_escala_x(min_x,escala_x,xs1)
    y1_1 = config.obtener_escala_y(min_y,escala_y,ys1) 

    x1_2 = config.obtener_escala_x(min_x,escala_x,xs2)
    y1_2 = config.obtener_escala_y(min_y,escala_y,ys2) 
    
    return config.distancia_euclidiana(x1_2,x1_1,y1_2,y1_1)




def set_selected_person(tid):
    """Se llama desde Flask cuando el usuario hace click en una persona"""
    global selected_id,contador,label,vector_distancia_32_31,vector_distancia_28_27,vector_distancia_26_25,vector_distancia_31_23,vector_distancia_32_24,vector_distancia_16_12,vector_distancia_15_11,vector_distancia_32_16,vector_distancia_31_15
    selected_id = tid
    print(f"[INFO] Persona seleccionada desde frontend con ID {tid}")
    contador = 0
    label = f""
    vector_distancia_32_31.clear()
    vector_distancia_28_27.clear()
    vector_distancia_26_25.clear()
    vector_distancia_31_23.clear()
    vector_distancia_32_24.clear()
    #nuevos vectores
    vector_distancia_16_12.clear()
    vector_distancia_15_11.clear()
    vector_distancia_32_16.clear()
    vector_distancia_31_15.clear()


#JosselynV
def generate_frames(video_path,participante="JosselynV"):
    global label,last_detections, selected_id,contador,vector_distancia_32_31,vector_distancia_28_27,vector_distancia_26_25,vector_distancia_31_23,vector_distancia_32_24,vector_distancia_16_12,vector_distancia_15_11,vector_distancia_32_16,vector_distancia_31_15,orientacionString
    selected_id= None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el video.")
        return

    frame_count = 0  # Contador de frames

    #variables globales de los videos
    
    #vectores de distancias
    #vectores de distancias
    vector_distancia_32_31.clear()
    vector_distancia_28_27.clear()
    vector_distancia_26_25.clear()
    vector_distancia_31_23.clear()
    vector_distancia_32_24.clear()
    #nuevos vectores
    vector_distancia_16_12.clear()
    vector_distancia_15_11.clear()
    vector_distancia_32_16.clear()
    vector_distancia_31_15.clear()


    persona_identificada = "No identificada"
    persona_identificada2 = "No identificada"
    
    #cruce de rodillas 
    cruce_rodillas_indicador = False
    orientacion=1 #por defecto se inicia frontal
    #orientacion 1= frontal, 2= espalda, 3= lateral


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_r = cv2.resize(frame, (config.ANCHO, config.ALTO))
        frame_mejorado = frame_r

        results = model.track(frame_r, persist=True, classes=[0], conf=CONF_THRES, verbose=False)
        detections = []

        if results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box[:4])
                tid = int(tid.item())
                detections.append((tid, (x1, y1, x2, y2)))

                color = (0, 255, 0)
                if tid == selected_id:
                    color = (0, 0, 255)  # rojo si está seleccionado
                    contador += 1

                    normalizacion, xs, ys, escala_x, escala_y,min_x_,max_x_,min_y_,max_y_ = RecorteNormalizacion(frame_mejorado, y1, y2, x1, x2)
                    #RecorteNormalizacion(frame_mejorado, min_y, max_y, min_x, max_x):

                    if xs is not None and ys is not None and len(xs) > 0 and len(ys) > 0:

                        x_23, x_24 = xs[23] * escala_x, xs[24] * escala_x
                        x_11, x_12 = xs[11] * escala_x, xs[12] * escala_x
                        x_25, x_26 = xs[25] * escala_x, xs[26] * escala_x
                        centro_horizontal = escala_x // 2

                        cruce_caderas = (x_23 < centro_horizontal < x_24) or (x_24 < centro_horizontal < x_23)
                        cruce_hombros = (x_11 < centro_horizontal < x_12) or (x_12 < centro_horizontal < x_11)
                        cruce_rodillas = (x_25 < centro_horizontal < x_26) or (x_26 < centro_horizontal < x_25)
                        cruce_rodillas_2 = abs(x_25 - x_26) < 5

                        diff_caderas = abs(x_23 - x_24)
                        diff_hombros = abs(x_11 - x_12)

                        # tomar las medidas correspondientes skere modo diablo
                        #32_31
                        r_32_31=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[31],xs[32],ys[31],ys[32])
                        vector_distancia_32_31.append(r_32_31)
                        cv2.putText(frame_mejorado, f'r_32_31: {r_32_31}', (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #28_27
                        r_28_27=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[27],xs[28],ys[27],ys[28])
                        vector_distancia_28_27.append(r_28_27)
                        cv2.putText(frame_mejorado, f'r_28_27: {r_28_27}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #26_25
                        r_26_25=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[25],xs[26],ys[25],ys[26])
                        vector_distancia_26_25.append(r_26_25)
                        cv2.putText(frame_mejorado, f'r_26_25: {r_26_25}', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #31_23
                        r_31_23=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[23],xs[31],ys[23],ys[31])
                        vector_distancia_31_23.append(r_31_23)
                        cv2.putText(frame_mejorado, f'r_31_23: {r_31_23}', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #32_24
                        r_32_24=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[24],xs[32],ys[24],ys[32])
                        vector_distancia_32_24.append(r_32_24)
                        cv2.putText(frame_mejorado, f'r_32_24: {r_32_24}', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #16_12
                        r_16_12=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[12],xs[16],ys[12],ys[16])
                        vector_distancia_16_12.append(r_16_12)
                        cv2.putText(frame_mejorado, f'r_16_12: {r_16_12}', (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #15_11
                        r_15_11=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[11],xs[15],ys[11],ys[15])
                        vector_distancia_15_11.append(r_15_11)
                        cv2.putText(frame_mejorado, f'r_15_11: {r_15_11}', (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #32_16
                        r_32_16=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[16],xs[32],ys[16],ys[32])
                        vector_distancia_32_16.append(r_32_16)
                        cv2.putText(frame_mejorado, f'r_32_16: {r_32_16}', (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        #31_15
                        r_31_15=distancia_puntos(min_x_,escala_x,min_y_,escala_y,xs[15],xs[31],ys[15],ys[31])
                        vector_distancia_31_15.append(r_31_15)
                        cv2.putText(frame_mejorado, f'r_31_15: {r_31_15}', (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        #deteccion de la orientacion de la persona Frontal, Espalda, Lateral
                        mano_izquierda= int((xs[16]))
                        mano_derecha = int((xs[15]))

                        cv2.putText(frame_mejorado, f'mano_der: {mano_derecha} mano_izq: {mano_izquierda} ', (30, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        #si la mano izquierda es menor a 60 entonces esta de frente

                        #en los primeros 10 fotogramas se evalua la orientacion si esque el cruce de rodillas es false
                        if contador<=10 and cruce_rodillas_indicador == False: #and mano_derecha > 40
                            orientacion = 1 if (mano_izquierda < 70 ) else 2
                        # en caso de que no se ha detectado cruce de rodillas entonces evaluarlo
                        if cruce_rodillas_indicador == False:
                            #si esta en false es porque no se ha evaluado la orientacion en lateral
                            if (cruce_caderas or cruce_hombros or cruce_rodillas or cruce_rodillas_2 or (diff_caderas < 20 and diff_hombros < 20)):
                                #orientacion = "Lateral"
                                orientacion = 3
                                cruce_rodillas_indicador= True
                            
                        #si la los puntos p32 o p31 se pasan de cierta coordenada entonces resetear el contador a 0 y dejar de tomar datos
                        pie_xd = int((ys[32]))
                        pie_xd_2 = int((ys[31]))

                        cv2.putText(frame_mejorado, f'pie_der: {pie_xd} pie_izq: {pie_xd_2}', (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        #si los pies estan fuera de rango se omite el analisis
                        if pie_xd >= 380 or pie_xd_2>=380:
                            #cv2.putText(frame_mejorado, f'Marcha Fuera de Rango', (30, 520),
                            #cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

                            contador=0
                            vector_distancia_26_25.clear()
                            vector_distancia_28_27.clear()
                            vector_distancia_31_23.clear()
                            vector_distancia_32_24.clear()
                            vector_distancia_32_31.clear()
                            vector_distancia_16_12.clear()
                            vector_distancia_15_11.clear()
                            vector_distancia_32_16.clear()
                            vector_distancia_31_15.clear()
                        
                        else:
                            tamano_texto = 0.5
                            #cv2.putText(frame_mejorado, f'{persona_identificada}', (int(xs[0]+110), int(ys[0]+60)),
                            #cv2.FONT_HERSHEY_SIMPLEX, tamano_texto, (255, 255, 255), 2)
                            #cv2.putText(frame_mejorado, label, (x1, y1 - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


                            # Texto y posición
                            texto = label
                            org = (x1, y1 - 10)  # posición inicial (arriba de la caja)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            escala = 0.6
                            grosor = 1

                            # Calcular tamaño del texto
                            (t_w, t_h), _ = cv2.getTextSize(texto, font, escala, grosor)

                            # Ajustar coordenadas para el fondo
                            top_left = (org[0] - 5, org[1] - t_h - 5)
                            bottom_right = (org[0] + t_w + 5, org[1] + 5)

                            # Dibujar fondo negro
                            cv2.rectangle(
                                frame_mejorado,
                                top_left,
                                bottom_right,
                                (0, 0, 0),  # negro
                                -1          # relleno
                            )

                            # Dibujar texto encima (color que ya usas para cada persona)
                            cv2.putText(
                                frame_mejorado,
                                texto,
                                org,
                                font,
                                escala,
                                (0, 255, 255),  # amarillo en BGR, 
                                grosor,
                                cv2.LINE_AA
                            )



                            if (contador>=35):
                                vectores = {
                                "26_25": vector_distancia_26_25,
                                "28_27": vector_distancia_28_27,
                                "31_23": vector_distancia_31_23,
                                "32_24": vector_distancia_32_24,
                                "32_31": vector_distancia_32_31,
                                "16_12": vector_distancia_16_12,
                                "15_11": vector_distancia_15_11,
                                "32_16": vector_distancia_32_16,
                                "31_15": vector_distancia_31_15
                                    }

                                prediccion,probabilidad_predicha, probabilidades = config.RealizarPrediccion(vectores,orientacion,"RF")#modelo_seleccionado
                                #diccionario_stream.write(f"{probabilidades}")
                                print(probabilidades)
                                #df = pd.DataFrame(list(probabilidades.items()), columns=["Nombre", "Valor"])
                                #diccionario_stream.table(df)
                                #print(f"prediccion=>{prediccion}")
                                #si la probabilidad es igual o mas alta que la precision de probabilidad entonces alli si afirmar a la persona identificada

                                if probabilidad_predicha >= 0: #config.precision_identificacion: #0: #config.precision_identificacion: 
                                    identificacion = f"{prediccion} -> {probabilidad_predicha:.2f} %"
                                    print(identificacion)
                                    #persona_identificada = f"{identificacion}"
                                    label = f"{identificacion}"
                                    #identificacion_stream.write(f"Identificación -> {persona_identificada}   {probabilidad_predicha:.2f} %")
                                    #identificacion_stream.markdown(
                                    #estilos.subtitulo_centrado(f"Identificación -> {persona_identificada}   {probabilidad_predicha:.2f} %"),
                                    #unsafe_allow_html=True
                                    #)
                                    persona_identificada2 = prediccion
                                    #print(f"Persona identificada > {config.precision_identificacion} >: {prediccion}")
                                    #Calcular el PI    
                                    #if prediccion == participante:
                                    #PI["contador"] += 1
                                    #print(f"(PI)=+1")
                                else:
                                    identificacion = f"No identificado"
                                    label = f"{identificacion}"

                                #Calcular el PI    
                                if persona_identificada2 == participante:
                                    PI["contador"] += 1
                                    print(f"Se mantiene el (PI)=+1 {PI['contador']}")

                                #Calcular los VP, FP    
                                if prediccion == participante:
                                    VP["contador"] += 1
                                    #PI["contador"] += 1
                                    print(f"(VP)=+1 {VP['contador']}")
                                    #print(f"(PI)=+1")
                                else:
                                    FP["contador"] += 1
                                    print(f"(FP)=+1 {FP['contador']}")
                    
                                #guardar las muestras en la BD
                                #sv.registrar_puntos_muestra(videoid,muestraid,promedio_32_31,desviacion_32_31,promedio_28_27,desviacion_28_27,promedio_26_25,desviacion_26_25,promedio_31_23,desviacion_31_23,promedio_32_24,desviacion_32_24,promedio_16_12,desviacion_16_12,promedio_15_11,desviacion_15_11,promedio_32_16,desviacion_32_16,promedio_31_15,desviacion_31_15,orientacion)

                                vector_distancia_26_25.clear()
                                vector_distancia_28_27.clear()
                                vector_distancia_31_23.clear()
                                vector_distancia_32_24.clear()
                                vector_distancia_32_31.clear()
                                vector_distancia_16_12.clear()
                                vector_distancia_15_11.clear()
                                vector_distancia_32_16.clear()
                                vector_distancia_31_15.clear()

                                cruce_rodillas_indicador= False
                                contador=0
                                
                        
                        #cv2.putText(frame_mejorado, f'{label}', (40, 200),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        #cont2 = f"(VP) {VP['contador']}"
                        #cv2.putText(frame_mejorado, f'{cont2}', (30, 400),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        #cont1 = f"(FP) {FP['contador']}"
                        #cv2.putText(frame_mejorado, f'{cont1}', (120, 400),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        
                        #cv2.putText(frame_mejorado, f'{contador}', (30, 500),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Texto y posición
                        texto = f'{contador}'
                        org = (30, 500)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        escala = 0.7
                        grosor = 2

                        # Calcular el tamaño del texto
                        (t_w, t_h), _ = cv2.getTextSize(texto, font, escala, grosor)

                        # Dibujar rectángulo de fondo (contorno cuadrado)
                        cv2.rectangle(
                            frame_mejorado,
                            (org[0] - 5, org[1] - t_h - 5),  # esquina superior izquierda
                            (org[0] + t_w + 5, org[1] + 5),  # esquina inferior derecha
                            (0, 0, 0),  # color negro (fondo)
                            -1          # -1 = relleno
                        )

                        # Dibujar texto encima (color amarillo)
                        cv2.putText(
                            frame_mejorado,
                            texto,
                            org,
                            font,
                            escala,
                            (0, 255, 255),  # amarillo en BGR
                            grosor,
                            cv2.LINE_AA
                        )

                        cv2.putText(frame_mejorado, f'{video_path}', (30, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        #cv2.putText(normalizacion, f'Contador: {contador}', (30, 20),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                        orientacionString = "Frontal" if orientacion == 1 else "Espalda" if orientacion == 2 else "Lateral"
                        #cv2.putText(frame_mejorado, f': {orientacionString}', (30, 300),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                        #nueva_w, nueva_h = config.NORMALIZADO_ANCHO, config.NORMALIZADO_ALTO
                        #cv2.line(normalizacion, (nueva_w // 2, 0), (nueva_w // 2, nueva_h), (255, 255, 0), 2)
                        #cv2.line(normalizacion, (0, nueva_h // 2), (nueva_w, nueva_h // 2), (255, 0, 0), 2)
                        #y_linea = int(nueva_h * 0.96)
                        #cv2.line(normalizacion, (0, y_linea), (nueva_w, y_linea), (255, 0, 0), 2)





                #cv2.rectangle(frame_r, (x1, y1), (x2, y2), color, 2)
                #cv2.putText(frame_r, f"ID {tid}", (x1, y1 - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        last_detections = detections

        # Codificar frame_r en JPEG para streaming
        ret, buffer = cv2.imencode('.jpg', frame_r)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


