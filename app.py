# app.py
from flask import Flask, render_template, Response, request, jsonify
import detector  # archivo con YOLO
import muestas
import entrenamiento as et
import threading

#from utils.videos import listar_videos
from videos import listar_videos

app = Flask(__name__)

VIDEO_PATH = None
Persona_detection = None

# Variable global para estado del entrenamiento
estado_entrenamiento = {
    "activo": False,
    "mensaje": ""
}


# --- Ruta principal ---
#ruta para listar los videos
@app.route("/")
def list_videos():
    arbol = listar_videos()
    return render_template("treeview.html", arbol=arbol)




# --- IDENTIFICACION ------------------------------------------------------------------>
@app.route('/ver_identificacion_video')
def index():
    global VIDEO_PATH
    video = request.args.get("video")
    persona = request.args.get("persona")
    if video:  # si viene desde el treeview
        VIDEO_PATH = video
    return render_template('identificacion.html', video=VIDEO_PATH, persona=persona)


# --- Ruta para actualizar la selección ---
@app.route('/select_person', methods=['POST'])
def select_person():
    data = request.json
    tid = int(data['tid'])  # recibir el ID desde el frontend
    detector.set_selected_person(tid)  # actualizar en detector.py
    return jsonify({"status": "ok"})


# --- Ruta del feed de video ---
@app.route('/video_feed')
def video_feed():
    if not VIDEO_PATH:
        return "No hay video seleccionado", 400
    return Response(detector.generate_frames(VIDEO_PATH),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detections')
def get_detections():
    # detector.last_detections = [(tid, (x1,y1,x2,y2)), ...]
    data = [{"tid": tid, "box": box} for tid, box in detector.last_detections]
    return jsonify(data)

#ruta para obtener mas informacion de la identificacion de la persona para no mostrarlo en la ventanda del openCV
@app.route('/get_info')
def get_info():
    data = {
        "selected_id": detector.selected_id,
        "contador": detector.contador,
        "label": detector.label,
        "CONF_THRES": detector.CONF_THRES,
        "vector_distancia_32_31": detector.vector_distancia_32_31,
        "vector_distancia_28_27": detector.vector_distancia_28_27,
        "vector_distancia_26_25": detector.vector_distancia_26_25,
        "vector_distancia_31_23": detector.vector_distancia_31_23,
        "vector_distancia_32_24": detector.vector_distancia_32_24,
        "vector_distancia_16_12": detector.vector_distancia_16_12,
        "vector_distancia_15_11": detector.vector_distancia_15_11,
        "vector_distancia_32_16": detector.vector_distancia_32_16,
        "vector_distancia_31_15": detector.vector_distancia_31_15,
        "VP":detector.VP['contador'],
        "FP": detector.FP['contador'],
        'orientacionString': detector.orientacionString
    }
    return jsonify(data)




# MUESTRAS ---------------------------------------------------------------------->
#vista para obtener las muestras
@app.route("/VerMuestra")
def ver_muestra():
    global VIDEO_PATH,Persona_detection
    video = request.args.get("video")
    persona = request.args.get("persona")
    if video:  # si viene desde el treeview
        VIDEO_PATH = video
    if persona:
        Persona_detection = persona
    return render_template('muestra.html', video=VIDEO_PATH,persona =persona)


# ESTAS FUNCIONES CAMBIARLAS PARA LA VISTA DE MUESTRAS DE LOS VIDEOS 
# --- Ruta para actualizar la selección ---
@app.route('/select_person_m', methods=['POST'])
def select_person_m():
    data = request.json
    tid = int(data['tid'])  # recibir el ID desde el frontend
    muestas.set_selected_person(tid)  # actualizar en muestras.py
    return jsonify({"status": "ok"})


# --- Ruta del feed de video ---
@app.route('/video_feed_m')
def video_feed_m():
    if not VIDEO_PATH:
        return "No hay video seleccionado", 400
    return Response(muestas.generate_frames(VIDEO_PATH,Persona_detection),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_detections_m')
def get_detections_m():
    # detector.last_detections = [(tid, (x1,y1,x2,y2)), ...]
    data = [{"tid": tid, "box": box} for tid, box in muestas.last_detections]
    return jsonify(data)

#ruta para obtener mas informacion de la identificacion de la persona para no mostrarlo en la ventanda del openCV
@app.route('/get_info_m')
def get_info_m():
    data = {
        "selected_id": muestas.selected_id,
        "contador": muestas.contador,
        "label": muestas.label,
        "CONF_THRES": muestas.CONF_THRES,
        "vector_distancia_32_31": muestas.vector_distancia_32_31,
        "vector_distancia_28_27": muestas.vector_distancia_28_27,
        "vector_distancia_26_25": muestas.vector_distancia_26_25,
        "vector_distancia_31_23": muestas.vector_distancia_31_23,
        "vector_distancia_32_24": muestas.vector_distancia_32_24,
        "vector_distancia_16_12": muestas.vector_distancia_16_12,
        "vector_distancia_15_11": muestas.vector_distancia_15_11,
        "vector_distancia_32_16": muestas.vector_distancia_32_16,
        "vector_distancia_31_15": muestas.vector_distancia_31_15,
        'orientacionString': muestas.orientacionString,
        "muestras_estado":"Guardar muestras activo" if muestas.save_samples else "Guardar muestras desactivado",
        "muestras_tipo":"Para entrenamiento" if muestas.train_sampele else "Para prueba",
    }
    return jsonify(data)

# PARA CAMBIAR EL ESTADO DEL STREAMING PAUSADO O REPRODUCIENDO
@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    data = request.json
    action = data.get("action")
    if action == "pause":
        muestas.set_streaming(False)
    elif action == "play":
        muestas.set_streaming(True)

    return jsonify({"status": "ok", "streaming": muestas.is_streaming()})

@app.route('/restart_stream', methods=['POST'])
def restart_stream():
    muestas.restart_video(VIDEO_PATH)
    return jsonify({"status": "ok", "message": "Video reiniciado"})

# para indicar si se tienen que guardar o no las muestras en tiempo real
@app.route('/toggle_save_samples', methods=['POST'])
def toggle_save_samples():
    data = request.json
    save = data.get("save", False)  # por defecto False si no viene
    muestas.set_save_samples(save)  # función  en muestras.py
    return jsonify({"status": "ok", "save": save})

# para indicar si las muestras que se estan tomando son para entrenamiento o prueba
@app.route('/toggle_train_samples', methods=['POST'])
def toggle_train_samples():
    data = request.json
    save = data.get("save", False)  # por defecto False si no viene
    muestas.set_train_samples(save)  # función  en muestras.py
    print("Muestras para entrenamiento" if save else "Muestras para prueba")
    return jsonify({"status": "ok", "save": save})











# ENTRENAMIENTO ------------------------------------------------------------------>
def entrenar_modelo(numero):
    global estado_entrenamiento
    estado_entrenamiento["activo"] = True
    estado_entrenamiento["mensaje"] = f"Iniciando entrenamiento Modelo_{numero}..."
    try:
        et.EntrenamientoRF(numero)
        estado_entrenamiento["mensaje"] = f"Modelo_{numero} entrenado correctamente."
    except Exception as e:
        estado_entrenamiento["mensaje"] = f"Error: {str(e)}"
    estado_entrenamiento["activo"] = False

@app.route("/entrenamiento")
def entrenamiento():
    return render_template("entrenamiento.html")

@app.route("/iniciar_entrenamiento/<int:numero>", methods=["POST"])
def iniciar_entrenamiento(numero):
    if not estado_entrenamiento["activo"]:
        thread = threading.Thread(target=entrenar_modelo, args=(numero,))
        thread.start()
        return jsonify({"status": "ok", "mensaje": f"Entrenamiento Modelo_{numero} iniciado."})
    else:
        return jsonify({"status": "busy", "mensaje": estado_entrenamiento["mensaje"]})

@app.route("/estado_entrenamiento")
def estado():
    return jsonify(estado_entrenamiento)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
    #app.run(port=3000)
    #app.run(debug=True)
