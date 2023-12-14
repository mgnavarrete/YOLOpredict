from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np

# Load and parse the KML file
file_path = 'PSN_Corregido.kml'
tree = ET.parse(file_path)
root = tree.getroot()

# Define namespaces to parse the KML file
namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

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


# Updated data extraction to include polygons
updated_data = []
for placemark in root.findall('.//kml:Placemark', namespaces):
    name = placemark.find('.//kml:name', namespaces).text if placemark.find('.//kml:name', namespaces) is not None else "Unknown"
    
    # Extracting point coordinates
    point_coordinates = placemark.find('.//kml:Point/kml:coordinates', namespaces)
    point = parse_coordinates(point_coordinates.text) if point_coordinates is not None else None

    # Extracting polygon coordinates
    polygon = placemark.find('.//kml:Polygon', namespaces)
    polygon_coords = parse_polygon_coordinates(polygon)
    polygon_coords = polygon_coords.split(" ")
    polygon_coords = [x.split(",") for x in polygon_coords]
    polygon_coords = [[float(x[0]), float(x[1])] for x in polygon_coords]
    poly1 = polygon_coords[0]
    poly2 = polygon_coords[1]
    poly3 = polygon_coords[2]
    poly4 = polygon_coords[3]


    # Calcular el ángulo para cada polígono
    yaw1 = anguloNorte(float(poly1[0]), float(poly1[1]), float(poly3[0]), float(poly3[1]))
    yaw2 = anguloNorte(float(poly2[0]), float(poly2[1]), float(poly4[0]), float(poly4[1]))

    yawprom = (yaw1 + yaw2) / 2

    updated_data.append({
        'name': name, 
        'point': point, 
        'polyP1': f"{poly1[0]},{poly1[1]}", 
        'polyP2': f"{poly2[0]},{poly2[1]}", 
        'polyP3': f"{poly3[0]},{poly3[1]}", 
        'polyP4': f"{poly4[0]},{poly4[1]}",
        'yaw': yawprom
    })
print("Tabla de coordenadas creadas")
# Creating an updated DataFrame
updated_df = pd.DataFrame(updated_data)
updated_df.head()  # Display the first few rows of the updated DataFrame to verify the data extraction

# Specify the file path for the CSV file
csv_file_path = 'kmlTable.csv'

# Save the DataFrame to a CSV file
updated_df.to_csv(csv_file_path, index=False)
print("Archivo CSV creado")