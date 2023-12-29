from ultralytics import YOLO
import os
import cv2
import numpy as np
from pyproj import CRS, Transformer
import pandas as pd
import json
from itertools import combinations
import math
import tkinter as tk
from tkinter import filedialog
from glob import glob

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def calcular_area_poligono(puntos):
    n = len(puntos)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += puntos[i][0] * puntos[j][1]
        area -= puntos[j][0] * puntos[i][1]
    area = abs(area) / 2.0
    return area

def centroide(puntos):
    x = sum(punto[0] for punto in puntos) / len(puntos)
    y = sum(punto[1] for punto in puntos) / len(puntos)
    return x, y

def angulo_con_respecto_al_centro(punto, centro):
    return math.atan2(punto[1] - centro[1], punto[0] - centro[0])

def ordenar_puntos_dis(puntos):
    centro = centroide(puntos)
    puntos = sorted(puntos, key=lambda punto: angulo_con_respecto_al_centro(punto, centro))

    return [puntos[0], puntos[2], puntos[1], puntos[3]]

def ordenar_puntos(puntos):
    # Ordenar los puntos basándose en su coordenada x
    puntos = sorted(puntos, key=lambda punto: punto[0])

    # Separar los puntos en dos grupos basados en su posición x
    izquierda = puntos[:2]
    derecha = puntos[2:]

    # Dentro de cada grupo, ordenarlos por su coordenada y
    izquierda = sorted(izquierda, key=lambda punto: punto[1])
    derecha = sorted(derecha, key=lambda punto: punto[1], reverse=True)

    # El orden final es: superior izquierdo, inferior izquierdo, inferior derecho, superior derecho
    return [izquierda[0], izquierda[1], derecha[0], derecha[1]]


def findClosest(x1,y1,df):
    x_utm, y_utm = geoImg[y1][x1][0], geoImg[y1][x1][1]
                
    # Dibujar el polígono en la imagen original
    lat, lon = transformer.transform(x_utm, y_utm)
    min_distance = float('inf')
    closest_point = None
    closest_name = ""
    polyPname = ""

    # Iterar sobre todas las filas del DataFrame
    for index, row in df.iterrows():
        for col in ['polyP1', 'polyP2', 'polyP3', 'polyP4']:
            latMap, lonMap = row[col]
            distance = haversine_distance(lat, lon, latMap, lonMap)
            if distance < min_distance:
                min_distance = distance
                closest_point = (latMap, lonMap)
                closest_name = row['name']  # Almacenar el nombre asociado al polígono más cercano
                polyPname = col
    return closest_name, min_distance, polyPname

def closest_values_sorted(lst, n=3):
    if len(lst) < n:
        return lst  # Retorna la lista completa si es más corta que n

    lst.sort()  # Ordena la lista

    min_diff = float('inf')
    closest_subset = []

    # Itera a través de la lista, considerando secuencias de n valores consecutivos
    for i in range(len(lst) - n + 1):
        current_subset = lst[i:i + n]
        diff = max(current_subset) - min(current_subset)

        if diff < min_diff:
            min_diff = diff
            closest_subset = current_subset

    return closest_subset

def anguloNorte(lat1, lon1, lat2, lon2):
    # Convierte latitud y longitud de grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calcula el cambio en las coordenadas
    dlon = lon2 - lon1

    # Calcula el angulo
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)

    # Convierte el angulo de radianes a grados
    initial_bearing = np.degrees(initial_bearing)

    # Normaliza el angulo
    bearing = (initial_bearing + 360) % 360

    return bearing

# Función para seleccionar múltiples directorios
def select_directories():
    path_root = filedialog.askdirectory(title='Seleccione el directorio raíz')
    while path_root:
        list_folders.append(path_root)
        path_root = filedialog.askdirectory(title='Seleccione otro directorio o cancele para continuar')
    if not list_folders:
        raise Exception("No se seleccionó ningún directorio")


list_folders = []
list_images = []
model_path = 'best.pt'
csv_file_path = 'kmlTable_FIT.csv'

# Iniciar Tkinter
root = tk.Tk()
root.withdraw()

# Llamar a la función para seleccionar directorios
select_directories()

for path_root in list_folders:
    print(f"Procesando Carpeta:{path_root}")
    # Construir rutas a los subdirectorios
    folder_path = os.path.join(path_root, 'original_img')  # Para las imágenes originales
    imgsFolder = os.path.join(path_root, 'cvat')
    geonp_path = os.path.join(path_root, 'georef_numpy')  # Para archivos numpy georeferenciados
    metadata_path = os.path.join(path_root, 'metadata')  # Para archivos JSON de metadatos
    metadatanew_path = os.path.join(path_root, 'metadata')  # Para archivos JSON con offset_yaw modificado


    # folder_path = 'test1/TC13PP/original_img' # Carpeta que contiene las imágenes originales
    # geonp_path = 'test1/TC13PP/georef_numpy_old' # Carpeta que contiene los archivos numpy georeferenciados
    # metadata_path = 'test1/TC13PP/metadata' # Carpeta que contiene los archivos JSON de metadatos
    # metadatanew_path = 'test1/TC13PP/metadata' # Carpeta que contiene los archivos JSON de metadatos con el offset_yaw modificado

    # folder_path = 'images/testImg' # Carpeta que contiene las imágenes originales
    # geonp_path = 'images/testNP' # Carpeta que contiene los archivos numpy georeferenciados
    # metadata_path = 'images/testMD' # Carpeta que contiene los archivos JSON de metadatos
    # metadatanew_path = 'images/testMD' # Carpeta que contiene los archivos JSON de metadatos con el offset_yaw modificado

    img_names = os.listdir(imgsFolder)
    img_names.sort()


    # Tus coordenadas UTM
    zone_number = 19
    zone_letter = 'S'

    # Define la proyección UTM (incluyendo la zona y el hemisferio)
    utm_crs = CRS(f"+proj=utm +zone={zone_number} +{'+south' if zone_letter > 'N' else ''} +ellps=WGS84")

    # Define la proyección de latitud/longitud
    latlon_crs = CRS("EPSG:4326")

    # Crear un objeto Transformer para la transformación de coordenadas
    transformer = Transformer.from_crs(utm_crs, latlon_crs, always_xy=True)



    if not os.path.exists(metadatanew_path):
            os.mkdir(metadatanew_path)
    # Preprocesar coordenadas en el DataFrame
    print("Cargando datos de KML...")

    df = pd.read_csv(csv_file_path)
    for col in ['polyP1', 'polyP2', 'polyP3', 'polyP4']:
        df[col] = df[col].apply(lambda x: tuple(map(float, x.split(','))))

    yawKML = df['yaw'].mean()
    # yawKML = 180
    print("El angulo del KML es: ", yawKML)
    print("Datos cargados")

    print("Cargando modelo YOLO..")
    model = YOLO(model_path)
    print("Modelo cargado")
    masking = 10
    print("Iniciando análisis de imágenes...")
    for image_path in img_names:
        keypoint = []

        img = cv2.imread(folder_path + "/" + image_path)

        H, W, _ = img.shape
        img_resized = cv2.resize(img, (640, 640))
        results = model(img_resized)
        yawList = []
        yawArea = []
        for result in results:
            if result.masks is not None:
                for j, mask in enumerate(result.masks.data):
                    mask = mask.cpu().numpy() * 255
                    mask = cv2.resize(mask, (W, H))
                    img = cv2.resize(img, (W, H))
                    # Convertir la máscara a una imagen binaria
                    _, thresholded = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

                    # Encontrar contornos
                    contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if image_path in list_images:
                        cv2.imwrite(f'masks/{image_path[:-4]}_{j}.png', mask)
                    if contours:
                        # Encuentra el contorno más grande
                        largest_contour = max(contours, key=cv2.contourArea)

                        # Aproximación del polígono
                        epsilon = 0.015* cv2.arcLength(largest_contour, True)
                        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
                        approx_polygon = sorted(approx_polygon, key=lambda x: x[0][0])
                        approx_polygon = np.array(approx_polygon, dtype=int)

                        # print(f"approx_polygon: {approx_polygon}")
                        if len(approx_polygon) > 3:
                            # print(f"Procesando Imagen: {image_path}")

                            x1 = approx_polygon[0][0][0]
                            y1 = approx_polygon[0][0][1]
                            x2 = approx_polygon[1][0][0]
                            y2 = approx_polygon[1][0][1]
                            x3 = approx_polygon[2][0][0]
                            y3 = approx_polygon[2][0][1]
                            x4 = approx_polygon[3][0][0]
                            y4 = approx_polygon[3][0][1]

                            puntos = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                            puntos_ordenados = ordenar_puntos(puntos)
                            x1, y1 = puntos_ordenados[0]
                            x2, y2 = puntos_ordenados[1]
                            x3, y3 = puntos_ordenados[2]
                            x4, y4 = puntos_ordenados[3]

                            area = calcular_area_poligono(puntos_ordenados)
                            if image_path in list_images:
                                print(f"area: {area}")
                            if area > 15000:
                                # Convertir a formato numpy
                                puntos_np = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], np.int32)
                                puntos_np = puntos_np.reshape((-1, 1, 2))
                                cv2.polylines(img, [puntos_np], isClosed=True, color=(0, 255, 0), thickness=3)

                                cv2.circle(img, (x1, y1), 5, (0, 0, 255), -1)
                                cv2.circle(img, (x4, y4), 5, (255, 0, 255), -1)
                                cv2.circle(img, (x2, y2), 5, (255, 0, 0), -1)
                                cv2.circle(img, (x3, y3), 5, (255, 255, 0), -1)



                                geoImg = np.load(f"{geonp_path}/{image_path[:-4]}.npy")

                                x1_utm, y1_utm = geoImg[y1][x1][0], geoImg[y1][x1][1]
                                x2_utm, y2_utm = geoImg[y2][x2][0], geoImg[y2][x2][1]
                                x3_utm, y3_utm = geoImg[y3][x3][0], geoImg[y3][x3][1]
                                x4_utm, y4_utm = geoImg[y4][x4][0], geoImg[y4][x4][1]
                                # print(f"coordenadas del poligono: {x1_utm, y1_utm}, {x2_utm, y2_utm}, {x3_utm, y3_utm}, {x4_utm, y4_utm}")

                                # Dibujar el polígono en la imagen original
                                lon1, lat1 = transformer.transform(x1_utm, y1_utm)
                                lon2, lat2 = transformer.transform(x2_utm, y2_utm)
                                lon3, lat3 = transformer.transform(x3_utm, y3_utm)
                                lon4, lat4 = transformer.transform(x4_utm, y4_utm)

                                # print(f"coordenadas del poligono: {lat1, lon1}, {lat2, lon2}, {lat3, lon3}, {lat4, lon4}")
                                # yaw1 = anguloNorte(float(lat1), float(lon1), float(lat4), float(lon4))
                                # yaw2 = anguloNorte(float(lat2), float(lon2), float(lat3), float(lon3))
                                
                                yaw1 = anguloNorte(float(lat4), float(lon4), float(lat1), float(lon1))
                                yaw2 = anguloNorte(float(lat3), float(lon3), float(lat2), float(lon2))
                                


                                offset_yaw1 = yawKML - yaw1
                                offset_yaw2 = yawKML - yaw2
                                # print(f"offset_yaw: {offset_yaw}")
                                yawList.append(offset_yaw1)
                                yawList.append(offset_yaw2)
        if image_path in list_images:

            cv2.imwrite("results/"+ image_path, img)
        masking += 1

        if len(yawList) == 0:
            offset_yaw = 0
        else:
            offsetList = closest_values_sorted(yawList, n=5)
            # promdeio de los valores de yawList
            offset_yaw = np.mean(offsetList)

        print(f"El offset_yaw de {image_path}: {offset_yaw}")
        # Abre el archivo JSON en modo lectura
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
            data = json.load(archivo)

        # Modifica el valor de "offset_yaw" con el número deseado
        data['offset_yaw'] = offset_yaw
        # print(f"El offset_yaw de {image_path}: {offset_yaw}")
        # Abre el archivo JSON en modo escritura
        with open(f'{metadatanew_path}/{image_path[:-4]}.txt', 'w') as archivo:
            # Escribe el diccionario modificado de nuevo en el archivo JSON
            json.dump(data, archivo, indent=4)


        print("El valor de 'offset_yaw' se ha modificado con éxito.")
    print(f"Carpeta {path_root} OK")

print("Todas la carpetas OK")