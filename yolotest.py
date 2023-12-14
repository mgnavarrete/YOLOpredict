from ultralytics import YOLO
import os
import cv2
import numpy as np
from pyproj import Proj, transform
import pandas as pd

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



model_path = 'last.pt'

folder_path = 'images/testImg' # Folder containing images to be segmented example: images/1.JPEG

img_names = os.listdir(folder_path)
img_names.sort()


# Tus coordenadas UTM
zone_number = 19
zone_letter = 'S'

# Define la proyección UTM (incluyendo la zona y el hemisferio)
utm_proj = Proj(proj='utm', zone=zone_number, south=zone_letter > 'N', ellps='WGS84')

# Define la proyección de latitud/longitud
latlon_proj = Proj(proj='latlong', ellps='WGS84')

# Specify the file path of the CSV file
csv_file_path = 'kmlTable.csv'

# Preprocesar coordenadas en el DataFrame
df = pd.read_csv(csv_file_path)
for col in ['polyP1', 'polyP2', 'polyP3', 'polyP4']:
    df[col] = df[col].apply(lambda x: tuple(map(float, x.split(','))))


for image_path in img_names:
    print(f"Analizando imagen: {image_path}")
    keypoint = []
    img = cv2.imread(folder_path + "/" + image_path)

    H, W, _ = img.shape
    img_resized = cv2.resize(img, (640, 640))

    model = YOLO(model_path)

    results = model(img_resized)
    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.cpu().numpy() * 255
            mask = cv2.resize(mask, (W, H))
            img = cv2.resize(img, (W, H))
            # Convertir la máscara a una imagen binaria
            _, thresholded = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

            # Encontrar contornos
            contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imwrite(f'masks/{image_path[:-4]}_{j}.png', mask)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # Aproximación del polígono
                epsilon = 0.015 * cv2.arcLength(largest_contour, True)
                approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                x1 = approx_polygon[0][0][0]
                y1 = approx_polygon[0][0][1]
                x2 = approx_polygon[1][0][0]
                y2 = approx_polygon[1][0][1]
                x3 = approx_polygon[2][0][0]
                y3 = approx_polygon[2][0][1]
                x4 = approx_polygon[3][0][0]
                y4 = approx_polygon[3][0][1]
            
                geoImg = np.load(f"images/testNP/{image_path[:-4]}.npy")
                namep1, minp1, polynamep1 = findClosest(x1,y1,df)
                namep2, minp2, polynamep2 = findClosest(x2,y2,df)
                namep3, minp3, polynamep3 = findClosest(x3,y3,df)
                namep4, minp4, polynamep4 = findClosest(x4,y4,df)
                
                # Almacenar los resultados en una lista de tuplas
                resultados = [
                    (namep1, minp1, polynamep1, x1,y1),
                    (namep2, minp2, polynamep2, x2,y2),
                    (namep3, minp3, polynamep3, x3,y3),
                    (namep4, minp4, polynamep4, x4,y4)
                ]

                # Ordenar los resultados por el valor de minpx (segundo elemento de cada tupla)
                resultados_ordenados = sorted(resultados, key=lambda x: x[1])

                # Seleccionar los dos primeros elementos de la lista ordenada
                dos_menores = resultados_ordenados[:2]

                # Guardar los nombres y polynam de los dos menores
                nombre_menor1, minp_menor1, polynam_menor1, xa, ya = dos_menores[0]
                nombre_menor2, minp_menor2, polynam_menor2, xb, yb = dos_menores[1]
            

                if nombre_menor1 != nombre_menor2:
                    print("El polígono es distinto")
                    # Buscar el con menor distancia
                    if minp_menor1 < minp_menor2:
                        nombre_menor2 = nombre_menor1
                        print(f"El polígono menor es {nombre_menor1}")
                        
                    else:
                        nombre_menor1 = nombre_menor2
                        print("El polígono menor es {nombre_menor2}")
                    
                
                print(f"Nombre: {nombre_menor1}, Distancia: {minp_menor1}, Polynam: {polynam_menor1}")
                print(f"Nombre: {nombre_menor2}, Distancia: {minp_menor2}, Polynam: {polynam_menor2}")
                cv2.circle(img, (xa, ya), 3, (0, 255, 0), -1)
                cv2.circle(img, (xb, yb), 3, (0, 255, 0), -1)
                keypoint.append([xa, ya])
                keypoint.append([xb, yb])
                       
        # Guardar la imagen con el polígono           
        cv2.imwrite(f'results/{image_path[:-4]}.png', img)
        
        # 
