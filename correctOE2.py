from ultralytics import YOLO
import os
import cv2
import numpy as np
from pyproj import CRS, Transformer
import pandas as pd
import json 
import math


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
def findClosest(x1, y1, df, poly):
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



model_path = 'best.pt'
csv_file_path = 'kmlTable.csv'

folder_path = 'test1/TC13PP/original_img' # Carpeta que contiene las imágenes originales
geonp_path = 'test1/TC13PP/georef_numpy' # Carpeta que contiene los archivos numpy georeferenciados
metadata_path = 'test1/TC13PP/metadata' # Carpeta que contiene los archivos JSON de metadatos
metadatanew_path = 'test1/TC13PP/metadata' # Carpeta que contiene los archivos JSON de metadatos con el offset_yaw modificado

# folder_path = 'images/testImg' # Carpeta que contiene las imágenes originales
# geonp_path = 'images/testNPnew' # Carpeta que contiene los archivos numpy georeferenciados
# metadata_path = 'images/testMD' # Carpeta que contiene los archivos JSON de metadatos
# metadatanew_path = 'images/testMD' # Carpeta que contiene los archivos JSON de metadatos con el offset_yaw modificado

img_names = os.listdir(folder_path)
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
# Crear un diccionario para mapear nombres a coordenadas de polyname
coordenadas_dict = df.set_index('name').to_dict(orient='index')
for image_path in img_names:

    keypoint = []
    
    img = cv2.imread(folder_path + "/" + image_path)

    H, W, _ = img.shape
    img_resized = cv2.resize(img, (640, 640))
    results = model(img_resized)
    oeList = []
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
    
                # cv2.imwrite(f'masks/{image_path[:-4]}_{j}.png', mask)
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
                        

                        # Convertir a formato numpy
                        puntos_np = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], np.int32)
                        puntos_np = puntos_np.reshape((-1, 1, 2))

                        geoImg = np.load(f"{geonp_path}/{image_path[:-4]}.npy")
                        
                        namep1, minp1, polynamep1 = findClosest(x1,y1,df,'polyP1')
                        namep2, minp2, polynamep2 = findClosest(x2,y2,df, 'polyP2')
                        namep3, minp3, polynamep3 = findClosest(x3,y3,df, 'polyP3')
                        namep4, minp4, polynamep4 = findClosest(x4,y4,df, 'polyP4')
                        
                        # Almacenar los resultados en una lista de tuplas
                        resultados = [
                            (namep1, minp1, polynamep1, x1,y1),
                            (namep2, minp2, polynamep2, x2,y2),
                            (namep3, minp3, polynamep3, x3,y3),
                            (namep4, minp4, polynamep4, x4,y4)
                        ]

                        # Ordenar los resultados por el valor de minpx (segundo elemento de cada tupla)
                        resultados_ordenados = sorted(resultados, key=lambda x: x[1])
                        # Extraer los nombres
                        # print(f"resultados_ordenados: {resultados_ordenados}")

                        

                        # Calcular constantes fuera del bucle
                        conversion_factor = 111319.9
                        oeListprev = []
                        for name, minp, polyname, x, y in resultados_ordenados:
                            
                            # Obtener lon y lat directamente del diccionario
                            lon, lat = coordenadas_dict[name][polyname].split(",")
                            lon, lat = float(lon), float(lat)
                            lat_rad = math.radians(lat)

                            x_utm, y_utm = geoImg[y][x][0], geoImg[y][x][1]
                            lonImg, latImg = transformer.transform(x_utm, y_utm)

                            # Calcular la diferencia de longitud y la distancia este-oeste
                            diff_lon = lon - lonImg
                            # Earth's circumference along the equator in kilometers
                            earth_circumference_km = 40075.0

                            # Convert offset from degrees to kilometers (1 degree = Earth's circumference / 360)
                            offset_km = diff_lon * (earth_circumference_km / 360)

                            # Convert kilometers to meters
                            offset_poly = offset_km * 1000
                                                            
                            oeListprev.append(offset_poly)
                        new_oelist = closest_values_sorted(oeListprev, n=2)
                        oeList.append(np.mean(new_oelist))
                        oeList.append(new_oelist[0])
                        oeList.append(new_oelist[1])
                        
                                
    # Guardar la imagen con el polígono           
    # cv2.imwrite(f'results/{image_path[:-4]}.png', img)
    
    if len(oeList) == 0:
        offset_oe = 0
    else:
        print("Buscando Valores más cercanos...")
        offsetList = closest_values_sorted(oeList, n=5)
        # promdeio de los valores de yawList
        offset_oe = np.mean(offsetList)
        
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
    
        
    print("El valor de 'offset_E' se ha modificado con éxito.")  
