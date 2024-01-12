from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np

def calcular_centro_poligono(poly1, poly2, poly3, poly4):
        lat_centro = (poly1[1] + poly2[1] + poly3[1] + poly4[1]) / 4
        lon_centro = (poly1[0] + poly2[0] + poly3[0] + poly4[0]) / 4
        return lat_centro, lon_centro

def procesar_archivo_kml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

# Function to parse coordinates string into latitude, longitude, and height
def parse_coordinates(coord_str):
    if coord_str:
        lon, lat, height = coord_str.split(',')
        return float(lat), float(lon), float(height)
    return None, None, None

# Function to parse polygon coordinates
def parse_polygon_coordinates(polygon):
    if polygon:
        outer_boundary = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
        if outer_boundary is not None and outer_boundary.text:
            return ' '.join([coords.strip() for coords in outer_boundary.text.split()])
    return None


def anguloNorte(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the change in coordinates
    dlon = lon2 - lon1

    # Calculate the bearing
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)

    # Convert bearing from radians to degrees
    initial_bearing = np.degrees(initial_bearing)

    # Normalize the bearing
    bearing = (initial_bearing + 360) % 360

    return bearing


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance 
list_file_path = ['kmlTables/LLKCorrection.kml']

new_data = []
id = 1
for file_path in list_file_path:
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespaces to parse the KML file
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Updated data extraction to include polygons
    updated_data = []
    for placemark in root.findall('.//kml:Placemark', namespaces):
        name = placemark.find('.//kml:name', namespaces).text if placemark.find('.//kml:name', namespaces) is not None else "Unknown"
        
        # Extracting polygon coordinates
        polygon = placemark.find('.//kml:Polygon', namespaces)
        polygon_coords = parse_polygon_coordinates(polygon)
        polygon_coords = polygon_coords.split(" ")
        
        polygon_coords = [x.split(",") for x in polygon_coords]
        polysCords = []
        for x in polygon_coords:
            polysCords.append([float(x[0]), float(x[1])])
        
        
        
        # Asignar coordenadas a las variables según los puntos cardinales
        poly1 = polysCords[0]  # Superior Izquierda
        poly2 = polysCords[1]  # inferior izquierda
        poly3 = polysCords[2]  # inferior derecha
        poly4 = polysCords[3]  # superior derecha
   
        # Extracting point coordinates
        point_coordinates = placemark.find('.//kml:Point/kml:coordinates', namespaces)
        if point_coordinates is not None:
            point = parse_coordinates(point_coordinates.text)
        else:
            # Calcular el punto central si no hay un punto y sí un polígono
            centro_lat, centro_lon = calcular_centro_poligono(poly2, poly1, poly4, poly3)
            point = (centro_lat, centro_lon)

        # Calcular el ángulo para cada polígono
        yaw1 = anguloNorte(float(poly1[1]), float(poly1[0]), float(poly3[1]), float(poly3[0]))
        yaw2 = anguloNorte(float(poly4[1]), float(poly4[0]), float(poly2[1]), float(poly2[0]))
        yawprom = (yaw1 + yaw2) / 2
        
        distancia1 = haversine_distance(poly1[1], poly1[0], poly4[1], poly4[0])
        distancia2 = haversine_distance(poly3[1], poly3[0], poly2[1], poly2[0])
        distancia_promedio = (distancia1 + distancia2) / 2
        # si yawprom sxe aleja de 180    
        if yawprom > 180 * 1.05 or yawprom < 180 * 0.95:
    
             # Calcular el ángulo para cada polígono
            yaw1 = anguloNorte(float(poly3[1]), float(poly3[0]), float(poly1[1]), float(poly1[0]))
            yaw2 = anguloNorte(float(poly2[1]), float(poly2[0]), float(poly4[1]), float(poly4[0]))

            yawprom = (yaw1 + yaw2) / 2
            
            distancia1 = haversine_distance(poly1[1], poly1[0], poly4[1], poly4[0])
            distancia2 = haversine_distance(poly3[1], poly3[0], poly2[1], poly2[0])
            distancia_promedio = (distancia1 + distancia2) / 2

        


        updated_data.append({
            'name': id, 
            'point':  f"{point[1]},{point[0]}", 
            'polyP1': f"{poly1[0]},{poly1[1]}", 
            'polyP2': f"{poly2[0]},{poly2[1]}", 
            'polyP3': f"{poly3[0]},{poly3[1]}", 
            'polyP4': f"{poly4[0]},{poly4[1]}",
            'yaw': yawprom,
            'ancho': distancia_promedio
        })
        id = id + 1
    print("Tabla de coordenadas creadas")
    # Creating an updated DataFrame
    updated_df = pd.DataFrame(updated_data)
    updated_df.head()  # Display the first few rows of the updated DataFrame to verify the data extraction
#PROMEDIO YAW
yawKML = updated_df['yaw'].mean()
print("Yaw promedio KML: ", yawKML)
# Specify the file path for the CSV file
csv_file_path = file_path.replace(".kml", ".csv")

# Save the DataFrame to a CSV file
updated_df.to_csv(csv_file_path, index=False)
print("Archivo CSV creado")