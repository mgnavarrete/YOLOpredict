from ultralytics import YOLO
import os
import cv2
import numpy as np
from pyproj import Proj, transform
import pandas as pd
import json

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def findClosest(x1,y1,df):
    x_utm, y_utm = geoImg[y1][x1][0], geoImg[y1][x1][1]
                
    # Dibujar el polígono en la imagen original
    lat, lon = transform(utm_proj, latlon_proj, x_utm, y_utm)
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



model_path = 'last.pt'
csv_file_path = 'test1/kmlTable.csv'

folder_path = 'test1/TC13PP/original_img' # Carpeta que contiene las imágenes originales
geonp_path = 'test1/TC13PP/georef_numpy' # Carpeta que contiene los archivos numpy georeferenciados
metadata_path = 'test1/TC13PP/metadata' # Carpeta que contiene los archivos JSON de metadatos
metadatanew_path = 'test1/TC13PP/metadata_yaw' # Carpeta que contiene los archivos JSON de metadatos con el offset_yaw modificado

# folder_path = 'images/testImg' # Carpeta que contiene las imágenes originales
# geonp_path = 'images/testNP' # Carpeta que contiene los archivos numpy georeferenciados
# metadata_path = 'images/testMD' # Carpeta que contiene los archivos JSON de metadatos
# metadatanew_path = 'images/testMD' # Carpeta que contiene los archivos JSON de metadatos con el offset_yaw modificado

img_names = os.listdir(folder_path)
img_names.sort()


# Tus coordenadas UTM
zone_number = 19
zone_letter = 'S'

# Define la proyección UTM (incluyendo la zona y el hemisferio)
utm_proj = Proj(proj='utm', zone=zone_number, south=zone_letter > 'N', ellps='WGS84')

# Define la proyección de latitud/longitud
latlon_proj = Proj(proj='latlong', ellps='WGS84')



if not os.path.exists(metadatanew_path):
        os.mkdir(metadatanew_path)
# Preprocesar coordenadas en el DataFrame
print("Cargando datos de KML...")

df = pd.read_csv(csv_file_path)
for col in ['polyP1', 'polyP2', 'polyP3', 'polyP4']:
    df[col] = df[col].apply(lambda x: tuple(map(float, x.split(','))))

yawKML = df['yaw'].mean()
print("El angulo del KML es: ", yawKML)
print("Datos cargados")

print("Cargando modelo YOLO..")
model = YOLO(model_path)
print("Modelo cargado")

print("Iniciando análisis de imágenes...")  
for image_path in img_names:
    keypoint = []
    
    img = cv2.imread(folder_path + "/" + image_path)

    H, W, _ = img.shape
    img_resized = cv2.resize(img, (640, 640))
    results = model(img_resized)
    yawList = []
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
                    epsilon = 0.015 * cv2.arcLength(largest_contour, True)
                    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
                    approx_polygon = sorted(approx_polygon, key=lambda x: x[0][0])
                    approx_polygon = np.array(approx_polygon, dtype=int)
                    # print(f"approx_polygon: {approx_polygon}")
                    
                    # Encuentra el rectángulo rotado que mejor se ajusta al contorno
                    rect = cv2.minAreaRect(largest_contour)
                    rect_points = cv2.boxPoints(rect).astype(int)
                    rect_points = sorted(rect_points, key=lambda x: x[0])
                    rect_points = np.array(rect_points, dtype=int)
                    
                    #print(f"rect_points: {rect_points}")
                    # img = cv2.drawContours(img, [rect_points], -1, (255, 0, 0), 2)
        
                    
                    x1 = approx_polygon[0][0][0]
                    y1 = approx_polygon[0][0][1]
                    x2 = approx_polygon[1][0][0]
                    y2 = approx_polygon[1][0][1]
                    x3 = approx_polygon[2][0][0]
                    y3 = approx_polygon[2][0][1]
                    x4 = approx_polygon[3][0][0]
                    y4 = approx_polygon[3][0][1]
                
                    cv2.circle(img, (x1, y1), 3, (0, 255, 0), -1)
                    cv2.circle(img, (x3, y3), 3, (0, 255, 0), -1)
                    geoImg = np.load(f"{geonp_path}/{image_path[:-4]}.npy")
                    
                    x1_utm, y1_utm = geoImg[y1][x1][0], geoImg[y1][x1][1]
                    x2_utm, y2_utm = geoImg[y2][x2][0], geoImg[y2][x2][1]
                    x3_utm, y3_utm = geoImg[y3][x3][0], geoImg[y3][x3][1]
                    x4_utm, y4_utm = geoImg[y4][x4][0], geoImg[y4][x4][1]
                    
                    # Dibujar el polígono en la imagen original
                    lat1, lon1 = transform(utm_proj, latlon_proj, x1_utm, y1_utm)
                    lat2, lon2 = transform(utm_proj, latlon_proj, x2_utm, y2_utm)
                    lat3, lon3 = transform(utm_proj, latlon_proj, x3_utm, y3_utm)
                    lat4, lon4 = transform(utm_proj, latlon_proj, x4_utm, y4_utm)

                    
                    yaw1 = anguloNorte(float(lon1), float(lat1), float(lon3), float(lat3))            
                    yaw2 = anguloNorte(float(lon2), float(lat2), float(lon4), float(lat4))
                    # print(f"angulo del poligono: {yaw1} y {yaw2}")
                    
                    yaw = (yaw1 + yaw2)/2
                    offset_yaw = yawKML - yaw
                    yawList.append(offset_yaw)
                
    # cv2.imwrite("results/"+ image_path, img)               
                
    if len(yawList) == 0:
        offset_yaw = 0
    else:
        offset_yaw = np.mean(yawList)  
          
    # Abre el archivo JSON en modo lectura
    with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
        data = json.load(archivo)

    # Modifica el valor de "offset_yaw" con el número deseado
    data['offset_yaw'] = offset_yaw
    print(f"El offset_yaw de {image_path}: {offset_yaw}")
    # Abre el archivo JSON en modo escritura
    with open(f'{metadatanew_path}/{image_path[:-4]}.txt', 'w') as archivo:
        # Escribe el diccionario modificado de nuevo en el archivo JSON
        json.dump(data, archivo, indent=4)
    
        
    print("El valor de 'offset_yaw' se ha modificado con éxito.")  
