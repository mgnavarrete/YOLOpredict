from correctScripts.correctH import correctH
from correctScripts.correctYaw import correctYaw
from correctScripts.saveGeoMatriz import saveGeoM
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

# Función para seleccionar múltiples directorios
def select_directories():
    
    path_root = filedialog.askdirectory(title='Seleccione el directorio raíz')
    while path_root:
        list_folders.append(path_root)
        path_root = filedialog.askdirectory(title='Seleccione otro directorio o cancele para continuar')
    if not list_folders:
        raise Exception("No se seleccionó ningún directorio")


if __name__ == '__main__':

    list_folders = []
    list_images = []
    model_path = 'correctScripts/best.pt'


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

    # Preprocesar coordenadas en el DataFrame
    print("Cargando datos de KML...")

    df = pd.read_csv(csv_file_path)
    print("Datos cargados")
    
    for col in ['polyP1', 'polyP2', 'polyP3', 'polyP4']:
        df[col] = df[col].apply(lambda x: tuple(map(float, x.split(','))))

    yawKML = df['yaw'].mean()
    
    print("Cargando modelo YOLO..")
    model = YOLO(model_path)
    print("Modelo cargado")

    zone_number = 19
    zone_letter = 'S'

    # Define la proyección UTM (incluyendo la zona y el hemisferio)
    utm_crs = CRS(f"+proj=utm +zone={zone_number} +{'+south' if zone_letter > 'N' else ''} +ellps=WGS84")

    # Define la proyección de latitud/longitud
    latlon_crs = CRS("EPSG:4326")

    # Crear un objeto Transformer para la transformación de coordenadas
    transformer = Transformer.from_crs(utm_crs, latlon_crs, always_xy=True)

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

        if not os.path.exists(metadatanew_path):
                os.mkdir(metadatanew_path)
        

       

        print("Iniciando análisis de imágenes...")

        correctH(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
        saveGeoM(img_names, metadata_path, geonp_path, path_root)   
        correctYaw(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, list_images)
        saveGeoM(img_names, metadata_path, geonp_path, path_root)   
        

    print("Todas la carpetas OK")