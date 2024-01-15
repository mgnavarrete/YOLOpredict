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
from tqdm import tqdm



def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance 

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

def angulo_con_respecto_al_centro(punto, centro):
    return math.atan2(punto[1] - centro[1], punto[0] - centro[0])

def ordenar_puntos(puntos):
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

# Función para dividir la columna 'poly' en dos columnas 'lat' y 'lon'
def split_poly_into_lat_lon(df, poly_column):
    df[['lon', 'lat']] = df[poly_column].str.split(',', expand=True).astype(float)
    return df

# Tu función findClosest modificada
def findClosest(x1, y1, df, poly, geoImg,transformer):
    df = split_poly_into_lat_lon(df,poly)
    x_utm, y_utm = geoImg[y1][x1][0], geoImg[y1][x1][1]
    lon, lat = transformer.transform(x_utm, y_utm)

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

def save_metadata(metadata_path, image_path, offsetValue, metadatanew_path, offsetkey):
    
        # Abre el archivo JSON en modo lectura
    with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
        data = json.load(archivo)


    data[offsetkey] = offsetValue

    # Abre el archivo JSON en modo escritura
    with open(f'{metadatanew_path}/{image_path[:-4]}.txt', 'w') as archivo:
        # Escribe el diccionario modificado de nuevo en el archivo JSON
        json.dump(data, archivo, indent=4)
    # print(f"El {offsetkey} de {image_path}: {offsetValue}")
# Función para seleccionar múltiples directorios
def select_directories():
    
    path_root = filedialog.askdirectory(title='Seleccione el directorio raíz')
    while path_root:
        list_folders.append(path_root)
        path_root = filedialog.askdirectory(title='Seleccione otro directorio o cancele para continuar')
    if not list_folders:
        raise Exception("No se seleccionó ningún directorio")

def correctNLLK(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, transformer):
    for image_path in tqdm(img_names, desc='Restableciendo Norte'):
        
        # abrir metadata y obtener lon1, lat1, lon2, lat2
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
            data = json.load(archivo)
        lon1 = data['lon1']
        lat1 = data['lat1']
        lon2 = data['lon2']
        lat2 = data['lat2']
        
        img = cv2.imread(folder_path + "/" + image_path)
        geoImg = np.load(f"{geonp_path}/{image_path[:-4]}.npy")
        H, W, _ = img.shape   
        x1corner = W-1
        y1corner = 0
        x2corner = W-1 
        y2corner = H-1
        x1c_utm, y1c_utm = geoImg[y1corner][x1corner][0], geoImg[y1corner][x1corner][1]
        x2c_utm, y2c_utm = geoImg[y2corner][x2corner][0], geoImg[y2corner][x2corner][1]
        lon1IMG, lat1IMG = transformer.transform(x1c_utm, y1c_utm)
        lon2IMG, lat2IMG = transformer.transform(x2c_utm, y2c_utm)
       
        # Calcular la diferencia de longitud y la distancia este-oeste
        diff_lat1 = lat1 - lat1IMG
        diff_lat2 = lat2 - lat2IMG
    
        # Earth's circumference along the equator in kilometers
        earth_circumference_km = 40075.0

        # Convert offset from degrees to kilometers (1 degree = Earth's circumference / 360)
        offset_N1 = diff_lat1 * (earth_circumference_km / 360)
        offset_N2 = diff_lat2 * (earth_circumference_km / 360)

        offset_np = (offset_N1 + offset_N2) / 2
        # Convert kilometers to meters
        offset_N = offset_np * 1000

            
        save_metadata(metadata_path, image_path, offset_N, metadatanew_path, 'offset_N')
        save_metadata(metadata_path, image_path, offset_N, metadatanew_path, 'offset_N_tot')
            # print("El valor de 'offset_altura' se ha modificado con éxito.")

    
    
if __name__ == '__main__':

    list_folders = []
    list_images = []
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
        path_root = path_root + "PP"
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

        print("Iniciando análisis de imágenes...")


        

    print("Todas la carpetas OK")
