import os
import urllib.parse

BASE_PATH = "Dataset"

def listar_videos(ruta=BASE_PATH):
    """
    Retorna un árbol de participantes con sus videos.
    Estructura:
    [
        {"nombre": "Juan", "carpeta": True, "hijos": [...]},
        {"nombre": "video1.mp4", "carpeta": False, "urls": {"identificacion": "...", "muestra": "..."}},
    ]
    """
    elementos = []
    for elemento in sorted(os.listdir(ruta)):
        full_path = os.path.join(ruta, elemento)
        if os.path.isdir(full_path):
            elementos.append({
                "nombre": elemento,
                "carpeta": True,
                "hijos": listar_videos(full_path)  # ya no paso persona aquí
            })
        else:
            if elemento.lower().endswith((".mp4", ".mov")):
                video_param = urllib.parse.quote(full_path)

                # calcular la persona: primer carpeta después de BASE_PATH
                relative_path = os.path.relpath(full_path, BASE_PATH)
                persona = relative_path.split(os.sep)[0]  # toma solo la primera carpeta

                persona_param = urllib.parse.quote(persona)

                elementos.append({
                    "nombre": elemento,
                    "carpeta": False,
                    "urls": {
                        "identificacion": f"/ver_identificacion_video?video={video_param}&persona={persona_param}",
                        "muestra": f"/VerMuestra?video={video_param}&persona={persona_param}"
                    }
                })
            else:
                elementos.append({
                    "nombre": elemento,
                    "carpeta": False,
                    "urls": {}
                })
    return elementos
