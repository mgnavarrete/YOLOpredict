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

def calcular_centro_poligono(poly1, poly2, poly3, poly4):
        lat_centro = (poly1[1] + poly2[1] + poly3[1] + poly4[1]) / 4
        lon_centro = (poly1[0] + poly2[0] + poly3[0] + poly4[0]) / 4
        return lat_centro, lon_centro
    
def calcular_centroide(puntos):
    suma_x = sum(p[0] for p in puntos)
    suma_y = sum(p[1] for p in puntos)
    count = len(puntos)
    return (int(round(suma_x / count)), int(round(suma_y / count)))


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

# def angulo_con_respecto_al_centro(punto, centro):
#     return math.atan2(punto[1] - centro[1], punto[0] - centro[0])

# def ordenar_puntos(puntos):
#     centro = centroide(puntos)
#     puntos = sorted(puntos, key=lambda punto: angulo_con_respecto_al_centro(punto, centro))

#     return [puntos[0], puntos[2], puntos[1], puntos[3]]

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

# Función para dividir la columna 'poly' en dos columnas 'lat' y 'lon'
def split_poly_into_lat_lon(df, poly_column):
    df[['lon', 'lat']] = df[poly_column].str.split(',', expand=True).astype(float)
    return df

# Tu función findClosest modificada
def findClosest(lat, lon, df, poly):
    df = split_poly_into_lat_lon(df,poly)
    # Calcula la distancia usando una función vectorizada
    df['distance'] = df.apply(lambda row: haversine_distance(lat, lon, row['lat'], row['lon']), axis=1)

    # Encuentra el punto más cercano
    closest_row = df.loc[df['distance'].idxmin()]
    closest_name = closest_row['name']
    min_distance = closest_row['distance']

    return closest_name, min_distance, poly

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
list_images = ["FIT_C41_V13_DJI_20231216020920_0256_T_20231216020920.JPG","FIT_C41_V13_DJI_20231216020920_0255_T_20231216020920.JPG","FIT_C41_V13_DJI_20231216020357_0062_T_20231216020357.JPG","FIT_C41_V10_DJI_20231216014044_0154_T_20231216014044.JPG","FIT_C41_V10_DJI_20231216014045_0155_T_20231216014045.JPG", "FIT_C41_V08_DJI_20231216013559_0021_T_20231216013559.JPG"]
model_path = 'best.pt'


# Iniciar Tkinter
root = tk.Tk()
root.withdraw()

print("Seleccione la tabla KML...")
csv_file_path = filedialog.askopenfile(title='Seleccione Tabla KML')
if not csv_file_path:
        raise Exception("No se seleccionó ningúna Tabla KML")
print("Tabla KML seleccionada")

# Llamar a la función para seleccionar directorios
print("Seleccione el directorio raíz...")
select_directories()
print("Directorio raíz seleccionado")

for path_root in list_folders:
    print(f"Procesando Carpeta:{path_root}")

    # Construir rutas a los subdirectorios
    folder_path = os.path.join(path_root, 'original_img')  # Para las imágenes originales
    imgsFolder = os.path.join(path_root, 'cvat')
    geonp_path = os.path.join(path_root, 'georef_numpy')  # Para archivos numpy georeferenciados
    metadata_path = os.path.join(path_root, 'metadata')  # Para archivos JSON de metadatos
    metadatanew_path = os.path.join(path_root, 'metadata')  # Para archivos JSON con offset_yaw modificado

    img_names = os.listdir(imgsFolder)
    img_names.sort()


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
    print("Datos cargados")

    print("Cargando modelo YOLO..")
    model = YOLO(model_path)
    print("Modelo cargado")
    ancho = df['ancho'].mean()
    print("Iniciando análisis de imágenes...")
    # Crear un diccionario para mapear nombres a coordenadas de polyname



    coordenadas_dict = df.set_index('name').to_dict(orient='index')
    for image_path in img_names:

        keypoint = []

        img = cv2.imread(folder_path + "/" + image_path)

        H, W, _ = img.shape
        img_resized = cv2.resize(img, (640, 640))
        results = model(img_resized)
        oeList = []
        alturaList = []
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
                            # Obtén las dimensiones de la imagen
                            H, W, _ = img.shape

                            # Asegúrate de que las coordenadas estén dentro de los límites de la imagen
                            x, y, w, h = cv2.boundingRect(largest_contour)

                            # Ajusta las coordenadas para asegurarte de que no excedan las dimensiones de la imagen
                            x = max(0, min(x, W - 1))
                            y = max(0, min(y, H - 1))
                            w = max(0, min(w, W - x))
                            h = max(0, min(h, H - y))

                            # Calcula los puntos del rectángulo
                            approx_polygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)

                            # Extrae las coordenadas individuales, asegurándote de que estén dentro de los límites de la imagen
                            x1, y1 = max(0, min(approx_polygon[0][0], W - 1)), max(0, min(approx_polygon[0][1], H - 1))
                            x2, y2 = max(0, min(approx_polygon[1][0], W - 1)), max(0, min(approx_polygon[1][1], H - 1))
                            x3, y3 = max(0, min(approx_polygon[2][0], W - 1)), max(0, min(approx_polygon[2][1], H - 1))
                            x4, y4 = max(0, min(approx_polygon[3][0], W - 1)), max(0, min(approx_polygon[3][1], H - 1))


                            puntos = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                            puntos_ordenados = ordenar_puntos(puntos)
                            x1, y1 = puntos_ordenados[0]
                            x2, y2 = puntos_ordenados[1]
                            x3, y3 = puntos_ordenados[2]
                            x4, y4 = puntos_ordenados[3]
                            
                            geoImg = np.load(f"{geonp_path}/{image_path[:-4]}.npy")

                            x1_utm, y1_utm = geoImg[y1][x1][0], geoImg[y1][x1][1]
                            x2_utm, y2_utm = geoImg[y2][x2][0], geoImg[y2][x2][1]
                            x3_utm, y3_utm = geoImg[y3][x3][0], geoImg[y3][x3][1]
                            x4_utm, y4_utm = geoImg[y4][x4][0], geoImg[y4][x4][1]

                            lon1, lat1 = transformer.transform(x1_utm, y1_utm)
                            lon2, lat2 = transformer.transform(x2_utm, y2_utm)
                            lon3, lat3 = transformer.transform(x3_utm, y3_utm)
                            lon4, lat4 = transformer.transform(x4_utm, y4_utm)

                            # Calcular el centro
                            centro_lat, centro_lon = calcular_centro_poligono([lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4])
                        

                            namep1, minp1, polynamep1 = findClosest(centro_lat, centro_lon,df,'point')
                            lonKML, latKML= coordenadas_dict[namep1]['point'].split(",")
                            lonKML, latKML = float(lonKML), float(latKML)
                            diff_lon = lonKML - centro_lon
                            
                            # Earth's circumference along the equator in kilometers
                            earth_circumference_km = 40075.0

                            # Convert offset from degrees to kilometers (1 degree = Earth's circumference / 360)
                            offset_km = diff_lon * (earth_circumference_km / 360)
                            
                            # Convert kilometers to meters
                            offset_poly = offset_km * 1000
                            
                            oeList.append(offset_poly)
                            # cv2.imwrite(f"results/{image_path[:-4]}_E.png", img)
                            


        if len(oeList) > 0:

            offset_list = closest_values_sorted(oeList, n=2)
            
            offset_oe = sum(offset_list) / len(offset_list)

        else:
            offset_oe = 0

        if image_path in list_images:
            cv2.imwrite(f"results/{image_path[:-4]}_E.png", img)
        
        print(f"El offset_E de {image_path}: {offset_oe}")
        # Abre el archivo JSON en modo lectura
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
            data = json.load(archivo)

        # Modifica el valor de "offset_oe" con el número deseado
        data['offset_E'] = offset_oe
        # print(f"El offset_yaw de {image_path}: {offset_yaw}")
        # Abre el archivo JSON en modo escritura
        with open(f'{metadatanew_path}/{image_path[:-4]}.txt', 'w') as archivo:
            # Escribe el diccionario modificado de nuevo en el archivo JSON
            json.dump(data, archivo, indent=4)

    print(f"Carpeta {path_root} OK")

print("Todas la carpetas OK")