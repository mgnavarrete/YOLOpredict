from correctScripts.correctH import correctH, correctHCDS, correctHLLK
from correctScripts.correctE import correctE, correctECDS, correctELLK2
from correctScripts.correctYaw import correctYaw, correctYawCDS, correctYawLLK
from correctScripts.correctN import correctNLLK
from correctScripts.saveGeoMatriz import saveGeoM, saveKML
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


def deleteGeoNp(geonp_path):
        for filename in tqdm(os.listdir(geonp_path), desc="Deleting GeoNumpy:"):
            file_path = os.path.join(geonp_path, filename)
            try:
                # Si es un archivo, lo elimina
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # También puedes agregar aquí una condición para eliminar directorios
                # elif os.path.isdir(file_path):
                #     shutil.rmtree(file_path)
            except Exception as e:
                print('Error al eliminar %s. Razón: %s' % (file_path, e))

def resetMD(image_names, metadata_path, var = 'all'):
    for image_path in tqdm(img_names, desc="Reset Metadata:"):
        # Carga la metadata de la imagen
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
            data = json.load(archivo)
        if var == 'all':
            data['offset_E'] = 0
            data['offset_altura'] = 0
            data['offset_E_tot'] = 0
            data['offset_yaw'] = 0
        elif var == 'E':
            data['offset_E'] = 0
            data['offset_E_tot'] = 0
        elif var == 'yaw':
            data['offset_yaw'] = 0
        elif var == 'altura':
            data['offset_altura'] = 0
            
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'w') as f:
            json.dump(data, f, indent=4)
 
def adjustMD(image_names, metadata_path, param, value):
    for image_path in tqdm(img_names, desc="Adjusting Metadata:"):
        # Carga la metadata de la imagen
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'r') as archivo:
            data = json.load(archivo)
        data[param] = value
        if param == 'offset_E': 
            data['offset_E_tot'] = value      
        with open(f'{metadata_path}/{image_path[:-4]}.txt', 'w') as f:
            json.dump(data, f, indent=4)
                

# Función para seleccionar múltiples directorios
def select_directories():
    
    path_root = filedialog.askdirectory(title='Seleccione el directorio raíz')
    while path_root:
        list_folders.append(path_root)
        path_root = filedialog.askdirectory(title='Seleccione otro directorio o cancele para continuar')
    if not list_folders:
        raise Exception("No se seleccionó ningún directorio")

def select_kml():
    print("Seleccione la tabla KML...")
    csv_file_path = filedialog.askopenfile(title='Seleccione Tabla KML')
    if not csv_file_path:
            raise Exception("No se seleccionó ningúna Tabla KML")
    print("Tabla KML seleccionada")
    return csv_file_path
if __name__ == '__main__':
    
    print("Seleccione Tipo de planta que se va a ajustar:")
    print("         1. Finis Terrae (FIT)")
    print("         2. Finis Terrae Extensión (FIX)")
    print("         3. Campos del Sol (CDS)")
    print("         4. Lalakama (LLK)")
    print("         5. Sol de Lila (SDL)")
    print("         x. Salir")
    planta = input("Seleccione una opción: ")
    if planta == '1':
        areaUmb = 10000
        difUmb = 0.002
        csv_file_path = select_kml()
    elif planta == '2':
        areaUmb = 0
        difUmb = 100000000000
        csv_file_path = select_kml()
    elif planta == '3':
        areaUmb = 15000
        difUmb = 0.002
        csv_file_path = 'kmlTables/CDS - Strings.csv'
    elif planta == '4':
        areaUmb = 0
        difUmb = 100000000
    
        csv_file_path = 'kmlTables/LLKCorrection.csv'
        
    elif planta == '5':
        areaUmb = 0
        difUmb = 100000000
    
        csv_file_path = 'kmlTables/SDL - Trackers numerados.csv'
        
    else:
        exit()

    list_folders = []
    list_images = []
    model_path = 'correctScripts/best.pt'


    # Iniciar Tkinter
    root = tk.Tk()
    root.withdraw()

    

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
    ancho = df['ancho'].mean()
    print("Yaw promedio KML: ", yawKML)
    print("Ancho promedio KML: ", ancho)
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
        if planta == '1':
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctH(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctYaw(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, ancho, list_images, areaUmb, difUmb)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctE(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveKML(img_names, path_root)
            deleteGeoNp(geonp_path)
        if planta == '2':
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctH(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctYaw(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, ancho, list_images, areaUmb, difUmb)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctE(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveKML(img_names, path_root)
            deleteGeoNp(geonp_path)
        
        if planta == '3':
            print("Ajustando Planta Campos del Sol...")
            resetMD(img_names, metadata_path)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctHCDS(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, ancho)
            correctYawCDS(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, ancho, list_images, areaUmb, difUmb)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctECDS(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveKML(img_names, path_root)
            deleteGeoNp(geonp_path)
        elif planta == '4':
            print("Ajustando Planta Lalakama...")
            
            resetMD(img_names, metadata_path, 'all')
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctHLLK(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, 0.0025, areaUmb, path_root)
            correctYawLLK(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, ancho, list_images, areaUmb, difUmb)
            adjustMD(img_names, metadata_path, 'offset_E', -5)
            # saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            # correctELLK2(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveKML(img_names, path_root)
            deleteGeoNp(geonp_path)
        
        elif planta == '5':
            print("Ajustando Planta Sol de Lila...")
            resetMD(img_names, metadata_path)
            saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            correctHLLK(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, ancho, areaUmb, path_root)
            correctYawLLK(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, ancho, list_images, areaUmb, difUmb)
            # saveGeoM(img_names, metadata_path, geonp_path, path_root)   
            # correctECDS(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            # adjustMD(img_names, metadata_path, 'offset_E', 0)
            # correctELLK2(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)
            saveKML(img_names, path_root)
            deleteGeoNp(geonp_path)
            
    print("Todas la carpetas OK")