from adentu_solar.utils import get_string_meta_data
from adentu_solar.utils import to_uint8, to_uint8g, gen_color_pallete
from adentu_solar.utils import get_thermal_dji, get_thermal_flir
from adentu_solar.utils import undistort_m2ea_th, undistort_m3e_th, undistort_zh20t, undistort_xt2
from adentu_solar.transform import yolo_2_plot, yolo_2_album
from adentu_solar.display import add_all_bboxes
from adentu_solar.ordenImagenes import ordenar_imagenes
import numpy as np
import json
import cv2
import utm
import string
from osgeo import ogr
import tkinter as tk
from tkinter import filedialog
import os
import simplekml
from datetime import datetime

np.random.seed(10)
from glob import glob

nnnn = 1
print('count ', nnnn)

#get_thermal = get_thermal_zh20t2(2)


def dms2dd(data):
    dd = float(data[0]) + float(data[1]) / 60 + float(data[2]) / (60 * 60)
    if data[3] == 'W' or data[3] == 'S':
        dd *= -1
    return dd


def save_georef_matriz(data, desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):

    metadata = data
    if metadata['Model'] == "MAVIC2-ENTERPRISE-ADVANCED":
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = 9 / 1000  # mavic 2 enterprice
        yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        center = get_image_pos_utm(data)
        if modo_altura == "relativo":
            altura = float(data['RelativeAltitude']) - float(offset_altura)
        else:
            altura = offset_altura
        GSD = tamano_pix * (altura) / dis_focal
        # Cálculo del desplazamiento debido al pitch de la cámara
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
        desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
    elif metadata['Model'] == "M3T":
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = 9 / 1000  # mavic 2 enterprice
        yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        center = get_image_pos_utm(data)
        if modo_altura == "relativo":
            altura = float(data['RelativeAltitude']) - float(offset_altura)
        else:
            altura = offset_altura
        GSD = tamano_pix * (altura) / dis_focal
        # Cálculo del desplazamiento debido al pitch de la cámara
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
        desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
    elif metadata['Model'] == "XT2":
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = 9 / 1000  # mavic 2 enterprice
        yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        center = get_image_pos_utm(data)
        if modo_altura == "relativo":
            altura = float(data['RelativeAltitude']) - float(offset_altura)
        else:
            altura = float(offset_altura)
        GSD = tamano_pix * (altura) / dis_focal
        # Cálculo del desplazamiento debido al pitch de la cámara
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
        desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
    elif metadata['Model'] == "ZH20T":
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = float(data['FocalLength'][:-2]) / 1000
        # yaw = np.pi * (float(data["FlightYawDegree"]) + desp_yaw) / 180
        yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0

        try:
            distancia_laser = float(data["LRFTargetDistance"]) #if dist is not None else dist
            lat_laser = float(data["LRFTargetLat"])
            lon_laser = float(data["LRFTargetLon"])
            altura = distancia_laser * abs(np.sin(pitch))
            GSD = tamano_pix * altura / dis_focal
            if ans is not None and sig is not None:
                if float(sig["LRFTargetLat"]) < lat_laser < float(ans["LRFTargetLat"]):
                    lon_laser += float(sig["LRFTargetLon"]) + float(ans["LRFTargetLon"])
                    lon_laser /= 3
            usar_posicion_laser = False
            if usar_posicion_laser:
                center = utm.from_latlon(lat_laser, lon_laser)
                desp_pitch = 0
            else:
                center = get_image_pos_utm(data)
                desp_pitch = altura * np.tan(-np.pi / 2 + pitch)

        except:

            center = get_image_pos_utm(data)
            if modo_altura == "relativo":
                altura = float(data['RelativeAltitude']) - float(offset_altura)
            else:
                altura = float(offset_altura)
            GSD = tamano_pix * (altura) / dis_focal
            # Cálculo del desplazamiento debido al pitch de la cámara
            pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
            desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
    else:
        print("===================================================")
        print("CÁMARA NO DEFINIDA")
        return

    mid_width = img_width / 2

    Matriz_y = np.zeros((img_height, img_width))
    Matriz_x = np.zeros((img_height, img_width))

    for pixel_y in range(img_height):
        distancia_y = (pixel_y - img_height / 2 + 0.5) * GSD
        Matriz_y[pixel_y, :] = np.ones(img_width) * -1 * distancia_y

    matriz_gsd_y = (np.append(Matriz_y[:, 0], Matriz_y[-1, 0]) - np.append(Matriz_y[0, 0], Matriz_y[:, 0]))
    matriz_gsd_x = matriz_gsd_y[1:-1]  # asumimos pixeles cuadrados
    matriz_gsd_x = np.append(matriz_gsd_x[0], matriz_gsd_x[:])

    for pixel_y in range(img_height):
        gsd_x = matriz_gsd_x[pixel_y]
        distancia_x = -gsd_x * (mid_width - 0.5)
        for pixel_x in range(img_width):
            Matriz_x[pixel_y, pixel_x] = distancia_x
            distancia_x = distancia_x + gsd_x

    # AJUSTAR OFFSET DEL GPS, VALORES REFERENCIALES
    Matriz_Este = Matriz_y * np.sin(yaw) - Matriz_x * np.cos(yaw) + center[0] + float(desp_este) + float(desp_pitch) * np.sin(yaw)
    Matriz_Norte = Matriz_y * np.cos(yaw) + Matriz_x * np.sin(yaw) + center[1] + float(desp_norte) + float(desp_pitch) * np.cos(yaw)

    #print(center[0], center[1])

    Matriz_zonas_1 = np.ones((img_height, img_width)) * center[2]
    Matriz_zonas_2 = np.ones((img_height, img_width)) * string.ascii_uppercase.find(center[3])

    matriz_puntos_utm = np.concatenate(
        [Matriz_Este[..., np.newaxis], Matriz_Norte[..., np.newaxis], Matriz_zonas_1[..., np.newaxis],
         Matriz_zonas_2[..., np.newaxis]], axis=-1)
    return matriz_puntos_utm


def get_image_pos_utm(data):
    # Obtiene las posiciones en el formato que sale con exiftools
    lat = data['GPSLatitude'].replace('\'', '').replace('"', '').split(' ')
    lng = data['GPSLongitude'].replace('\'', '').replace('"', '').split(' ')
    # Elimina la palabra 'deg' de los datos
    for v in lat:
        if v == 'deg':
            lat.pop(lat.index(v))
    for v in lng:
        if v == 'deg':
            lng.pop(lng.index(v))
    # Calcula la posición en coordenadas UTM
    pos = utm.from_latlon(dms2dd(lat), dms2dd(lng))

    return pos

def get_image_time(data):
    # Obtiene l hora de adquisición en el formato que sale con exiftools
    str_time = data['CreateDate'].split('-')[0]
    time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
    return time

def inicializar(in_step = 1):
    root = tk.Tk()
    root.withdraw()

    # Path de las imágenes a procesar (térmicas originales)
    path_images = filedialog.askdirectory(title='Seleccione path imagenes')
    im_path = glob(path_images + '/**/*.JPG', recursive=True)
    im_path = [l.replace('\\', '/') for l in im_path]

    # Path donde se guardarán los resultados
    path_save = path_images + 'PP'

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    if not os.path.exists(path_save + '/metadata'):
        os.mkdir(path_save + '/metadata')

    if not os.path.exists(path_save + '/detect'):
        os.mkdir(path_save + '/detect')

    if not os.path.exists(path_save + '/Temp'):
        os.mkdir(path_save + '/Temp')

    if not os.path.exists(path_save + '/georef_numpy'):
        os.mkdir(path_save + '/georef_numpy')

    if not os.path.exists(path_save + '/cvat'):
        os.mkdir(path_save + '/cvat')

    if not os.path.exists(path_save + '/original_img'):
        os.mkdir(path_save + '/original_img')

    if in_step == 0:
        return [im_path, path_save]

    if in_step == 2:
        return [im_path, path_save,path_images]

    #path_labels = filedialog.askdirectory(title='Seleccione path etiquetas')
    path_obj = filedialog.askopenfilename(title='Seleccione obj.names')
    path_labels = path_obj.replace('obj.names', 'obj_train_data')

    names = ["Tipo I - StringDesconectado",
             "Tipo II - StringCortoCircuito",
             "Tipo III - ModuloCircuitoAbierto",
             "Tipo IV - BusBar",
             "Tipo V - ModuloCortoCircuito",
             "Tipo VI - CelulaCaliente",
             "Tipo VII - ByPass",
             "Tipo VIII - PID",
             #"Tipo X - Múltiples puntos calientes por suciedad", # Se agrega en etapa real_detect
             "Tipo XX - Tracker fuera de posición"]
             #"Tipo IX - JunctionBoxCaliente"]

    return [im_path, path_labels, path_obj, path_save, names]


def detect2(im_path, path_labels, path_obj, path_save, names, desp_este, desp_norte, desp_yaw):
    with open(path_obj) as f:
        names2 = f.readlines()
        names2 = [l.replace('\n', '') for l in names2]

    mapeo = {0: names.index(names2[0]), 1: names.index(names2[1]), 2: names.index(names2[2]),
             3: names.index(names2[3]), 4: names.index(names2[4]), 5: names.index(names2[5]),
             6: names.index(names2[6]), 7: names.index(names2[7]), 8: 8}

    paleta = gen_color_pallete(len(names))
    paleta[3] = [255, 0, 0][::-1]
    paleta[4] = [255, 191, 0][::-1]
    paleta[2] = [128, 255, 0][::-1]
    paleta[0] = [0, 128, 255][::-1]
    paleta[5] = [255, 0, 255][::-1]

    paleta2 = paleta.copy()
    for i in range(len(paleta2)):
        paleta[i] = paleta2[mapeo[i]]

    for i in range(len(im_path)):
        imp = im_path[i]

        file_path = path_save + f'/metadata/{imp.split("/")[-1].replace(".JPG", ".txt")}'

        try:
            file = open(file_path, 'r')
            metadata = json.load(file)
            print("metadata_encontrada ", file_path)
            file.close()

        except:
            metadata = get_string_meta_data(imp)
            file = open(file_path, 'w')
            file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
            file.close()

        metadata_sig = None
        metadata_ans = None
        if i > 0:
            file_path_ans = path_save + f'/metadata/{im_path[i - 1].split("/")[-1].replace(".JPG", ".txt")}'
            try:
                file = open(file_path_ans, 'r')
                metadata_ans = json.load(file)
                print("metadata_encontrada ", file_path)
                file.close()

            except:
                metadata_ans = get_string_meta_data(im_path[i - 1])
                file = open(file_path_ans, 'w')
                file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
                file.close()

        if i < len(im_path) - 2:
            file_path_sig = path_save + f'/metadata/{im_path[i + 1].split("/")[-1].replace(".JPG", ".txt")}'
            try:
                file = open(file_path_sig, 'r')
                metadata_sig = json.load(file)
                print("metadata_encontrada ", file_path)
                file.close()

            except:
                metadata_sig = get_string_meta_data(im_path[i + 1])
                file = open(file_path_ans, 'w')
                file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
                file.close()

        matriz_geo = save_georef_matriz(metadata, desp_este, desp_norte, desp_yaw, ans=metadata_ans, sig=metadata_sig)
        geo_name = path_save + f'/georef_numpy/{imp.split("/")[-1].replace(".JPG", ".npy")}'
        np.save(geo_name, matriz_geo)

        im = get_thermal(imp)
        im_name = path_save + f'/Temp/{imp.split("/")[-1]}'
        cv2.imwrite(im_name, to_uint8g(im))

        geo_matrix = matriz_geo
        geo_matrix_shape = geo_matrix.shape
        conf = 1
        name = imp.split('/')[-1]

        imi = to_uint8g(im)

        # imi = undistort_zh20t(imi)
        im_name = path_save + f'/detect/{imp.split("/")[-1]}'
        cv2.imwrite(im_name, imi)


def avg_dist(im_path):
    lista = []
    for imp in im_path:
        file_path = path_save + f'/metadata/{imp.split("/")[-1].replace(".JPG", ".txt")}'

        try:
            file = open(file_path, 'r')
            data = json.load(file)
            print("metadata_encontrada ", file_path)
            file.close()


        except:
            data = get_string_meta_data(imp)
            file = open(file_path, 'w')
            file.write(json.dumps(data, indent=4, sort_keys=True, default=str))
            file.close()

        try:
            dist = float(data["LRFTargetDistance"])
        except:
            dist = float(data['RelativeAltitude'])
        lista.append(dist)
    return np.median(lista)


def detect(im_path, path_labels, path_obj, path_save, names, desp_este, desp_norte, desp_yaw):
    falla_id = 0
    with open(path_obj) as f:
        names2 = f.readlines()
        names2 = [l.replace('\n', '') for l in names2]

    mapeo = {0: names.index(names2[0]), 1: names.index(names2[1]), 2: names.index(names2[2]),
             3: names.index(names2[3]), 4: names.index(names2[4]), 5: names.index(names2[5]),
             6: names.index(names2[6]), 7: names.index(names2[7]), 8: 8}

    paleta = gen_color_pallete(len(names))
    paleta[3] = [255, 0, 0][::-1]
    paleta[4] = [255, 191, 0][::-1]
    paleta[2] = [128, 255, 0][::-1]
    paleta[0] = [0, 128, 255][::-1]
    paleta[5] = [255, 0, 255][::-1]

    paleta2 = paleta.copy()
    for i in range(len(paleta2)):
        paleta[i] = paleta2[mapeo[i]]

    # bbox_path = [path_labels + '/' + l.split("/")[-1].replace('.JPG', '.txt') for l in im_path]
    # bbox_path = [path_labels + '/' + im_path[0].split('/')[-3] + '_' + im_path[0].split('/')[-2] + '_' + l.split(
    #     "/")[-1].replace('.JPG', '.txt') for l in im_path]

    bbox_path = [path_labels + "/v" + l.split("/")[-2] + "_" + l.split("/")[-1].replace('.JPG', '.txt') for l in im_path]

    i = 0
    for imp, bp in zip(im_path, bbox_path):

        file_path = path_save + f'/metadata/{imp.split("/")[-1].replace(".JPG", ".txt")}'

        try:
            file = open(file_path, 'r')
            metadata = json.load(file)
            print("metadata_encontrada ", file_path)
            file.close()


        except:
            metadata = get_string_meta_data(imp)
            file = open(file_path, 'w')
            file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
            file.close()

        matriz_geo = save_georef_matriz(metadata, desp_este, desp_norte, desp_yaw)
        geo_name = path_save + f'/georef_numpy/{imp.split("/")[-1].replace(".JPG", ".npy")}'
        np.save(geo_name, matriz_geo)

        temperaturas = get_thermal(imp)
        temperaturas = undistort_m2ea_th(temperaturas)
        im_name = path_save + f'/Temp/{imp.split("/")[-1]}'
        cv2.imwrite(im_name, to_uint8g(temperaturas.copy()))

        with open(bp, 'r') as f:
            bboxes = f.readlines()

        geo_matrix = matriz_geo
        geo_matrix_shape = geo_matrix.shape
        conf = 1
        name = imp.split('/')[-1]

        if bboxes.__len__() > 0:
            to_file_bbox = ''
            pdata = []
            tdata = []
            for _b in bboxes:
                cls, xx, yy, ww, hh = _b.split(" ")
                xx = float(xx)
                yy = float(yy)
                ww = float(ww)
                hh = float(hh)
                dd = (xx - 0.5) ** 2 + (yy - 0.5) ** 2

                if xx < 0.10 or xx > 0.9 or yy < 0.1 or yy > 0.9:
                    continue

                # xx = 0.5
                # yy = 0.5
                # ww = 0.99999
                # hh = 0.99999

                pdata.append({'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)})

                c1, c2 = yolo_2_album(xx, yy, ww, hh)
                c1 = (int(c1[0] * 640), int(c1[1] * 512))
                c2 = (int(c2[0] * 640), int(c2[1] * 512))

                if c1[0] >= geo_matrix_shape[1]:
                    c1 = (geo_matrix_shape[1] - 1, c1[1])
                if c2[0] >= geo_matrix_shape[1]:
                    c2 = (geo_matrix_shape[1] - 1, c2[1])

                if c1[1] >= geo_matrix_shape[0]:
                    c1 = (c1[0], geo_matrix_shape[0] - 1)
                if c2[1] >= geo_matrix_shape[0]:
                    c2 = (c2[0], geo_matrix_shape[0] - 1)

                p1_utm = geo_matrix[c1[1]][c1[0]]
                p1_ll = utm.to_latlon(p1_utm[0], p1_utm[1], int(p1_utm[2]), string.ascii_uppercase[int(p1_utm[3])])

                p2_utm = geo_matrix[c1[1]][c2[0]]
                p2_ll = utm.to_latlon(p2_utm[0], p2_utm[1], int(p2_utm[2]), string.ascii_uppercase[int(p2_utm[3])])

                p3_utm = geo_matrix[c2[1]][c2[0]]
                p3_ll = utm.to_latlon(p3_utm[0], p3_utm[1], int(p3_utm[2]), string.ascii_uppercase[int(p3_utm[3])])

                p4_utm = geo_matrix[c2[1]][c1[0]]
                p4_ll = utm.to_latlon(p4_utm[0], p4_utm[1], int(p4_utm[2]), string.ascii_uppercase[int(p4_utm[3])])

                to_file_bbox += '{"poly" : "\'POLYGON(( %s %s, %s %s, %s %s, %s %s, %s %s ))\'","type" : "%s","conf" : "%s" ,' % (
                    p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1],
                    names[mapeo[int(cls)]], conf)

                subtemp = temperaturas[c1[1]:c2[1], c1[0]:c2[0]]
                if np.prod(subtemp.shape) > 0:
                    tmin = np.min(subtemp)
                    tmax = np.max(subtemp)
                    tmean = np.mean(subtemp)
                    tstd = np.std(subtemp)
                else:
                    tmin = 0
                    tmax = 0
                    tmean = 0
                    tstd = 0


                to_file_bbox += f'"tmin": {tmin}, "tmax": {tmax}, "tmean": {tmean}, "tstd": {tstd}, '

                to_file_bbox += '"geo_json" : {"type": "Polygon","coordinates":[[  [%s, %s],[%s, %s],[%s, %s],[%s, %s],[%s, %s] ]] }, ' % (
                    p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1])

                _name = name.replace('.JPG', f'_{falla_id}.JPG')
                to_file_bbox += f'"name" : "{_name}", "dd": {dd}, "falla_id": {falla_id}' + '}'
                falla_id += 1
                to_file_bbox = to_file_bbox.replace('}{', '},{')

                imi = to_uint8g(temperaturas.copy())
                all_boxes = yolo_2_plot([{'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)}],
                                        [None for _ in range(len(names))], colors=paleta)
                imi = add_all_bboxes(imi, all_boxes, 2)
                im_name = path_save + f'/detect/{_name}'
                cv2.imwrite(im_name, imi)

            with open(path_save + f'/detect/{imp.split("/")[-1].replace(".JPG", ".json")}', 'w') as f:
                f.write('[')
                f.write(to_file_bbox)
                f.write(']')
            # imi = to_uint8g(im)

            # all_boxes = yolo_2_plot(pdata, [None for _ in range(len(names))], colors=paleta)
            # imi = add_all_bboxes(imi, all_boxes, 1)
            # im_name = path_save + f'/detect/{imp.split("/")[-1]}'
            # cv2.imwrite(im_name, imi)


def detect_fast(im_path, path_labels, path_obj, path_save, names):
    falla_id = 0
    with open(path_obj) as f:
        names2 = f.readlines()
        names2 = [l.replace('\n', '') for l in names2]

    mapeo = {0: names.index(names2[0]), 1: names.index(names2[1]), 2: names.index(names2[2]),
             3: names.index(names2[3]), 4: names.index(names2[4]), 5: names.index(names2[5]),
             6: names.index(names2[6]), 7: names.index(names2[7]), 8: 8}

    paleta = gen_color_pallete(len(names))
    paleta[3] = [255, 0, 0][::-1]
    paleta[4] = [255, 191, 0][::-1]
    paleta[2] = [128, 255, 0][::-1]
    paleta[0] = [0, 128, 255][::-1]
    paleta[5] = [255, 0, 255][::-1]

    paleta2 = paleta.copy()
    for i in range(len(paleta2)):
        paleta[i] = paleta2[mapeo[i]]

    # bbox_path = [path_labels + '/' + l.split("/")[-1].replace('.JPG', '.txt') for l in im_path]
    # bbox_path = [path_labels + '/' + im_path[0].split('/')[-3] + '_' + im_path[0].split('/')[-2] + '_' + l.split(
    #     "/")[-1].replace('.JPG', '.txt') for l in im_path]

    # TODO: El nombre de la imagen para CVAT tiene el prefijo vN con el numero de vuelo. Hay que eliminar de acá y de la generación de imagenes CVAT

    #Usar uno de los dos 'bbox_path' según el caso
    #bbox_path = [path_labels + "/v" + l.split("/")[-2] + "_" + l.split("/")[-1].replace('.JPG', '.txt') for l in im_path]
    bbox_path = [path_labels + "/v" + l.split("/")[-2] + "_" + l.split("/")[-1].replace('.JPG', '.txt') for l in im_path]

    i = 0
    for imp, bp in zip(im_path, bbox_path):

        with open(bp, 'r') as f:
            bboxes = f.readlines()

        if bboxes.__len__() <= 0:
            continue

        nombre_imagen = imp.split("/")[-1].replace(".JPG", ".txt")
        file_path = path_save + f'/metadata/{nombre_imagen}'

        try:
            file = open(file_path, 'r')
            metadata = json.load(file)
            print("Metadata encontrada ", file_path)
            file.close()


        except:
            print(f"ERROR {nombre_imagen}: Metadata NO encontrada")
            #metadata = get_string_meta_data(imp)
            #file = open(file_path, 'w')
            #file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
            #file.close()


        if 'modo_altura' in metadata:
            matriz_geo = save_georef_matriz(metadata, metadata['offset_E_tot'], metadata['offset_N_tot'],metadata['offset_yaw'], metadata['offset_altura'], metadata['modo_altura'])
        else:
            matriz_geo = save_georef_matriz(metadata, metadata['offset_E_tot'], metadata['offset_N_tot'],metadata['offset_yaw'], metadata['offset_altura'])
        #geo_name = path_save + f'/georef_numpy/{imp.split("/")[-1].replace(".JPG", ".npy")}'
        #np.save(geo_name, matriz_geo)


        if metadata['Model'] == "MAVIC2-ENTERPRISE-ADVANCED":
            temperaturas = get_thermal_dji(imp)
            temperaturas = undistort_m2ea_th(temperaturas)
        elif metadata['Model'] == "ZH20T":
            temperaturas = get_thermal_dji(imp)
            temperaturas = undistort_zh20t(temperaturas)
        elif metadata['Model'] == "XT2":
            temperaturas = get_thermal_flir(imp)
            temperaturas = undistort_xt2(temperaturas)
        else:
            print("===================================================")
            print("CÁMARA NO DEFINIDA")
            return

        geo_matrix = matriz_geo
        geo_matrix_shape = geo_matrix.shape
        conf = 1
        name = imp.split('/')[-1]

        if bboxes.__len__() > 0:
            to_file_bbox = ''
            pdata = []
            tdata = []
            for _b in bboxes:
                cls, xx, yy, ww, hh = _b.split(" ")
                xx = float(xx)
                yy = float(yy)
                ww = float(ww)
                hh = float(hh)
                dd = (xx - 0.5) ** 2 + (yy - 0.5) ** 2

                if xx < 0.10 or xx > 0.9 or yy < 0.1 or yy > 0.9:
                    continue

                # xx = 0.5
                # yy = 0.5
                # ww = 0.99999
                # hh = 0.99999

                pdata.append({'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)})

                c1, c2 = yolo_2_album(xx, yy, ww, hh)
                c1 = (int(c1[0] * 640), int(c1[1] * 512))
                c2 = (int(c2[0] * 640), int(c2[1] * 512))

                if c1[0] >= geo_matrix_shape[1]:
                    c1 = (geo_matrix_shape[1] - 1, c1[1])
                if c2[0] >= geo_matrix_shape[1]:
                    c2 = (geo_matrix_shape[1] - 1, c2[1])

                if c1[1] >= geo_matrix_shape[0]:
                    c1 = (c1[0], geo_matrix_shape[0] - 1)
                if c2[1] >= geo_matrix_shape[0]:
                    c2 = (c2[0], geo_matrix_shape[0] - 1)

                p1_utm = geo_matrix[c1[1]][c1[0]]
                p1_ll = utm.to_latlon(p1_utm[0], p1_utm[1], int(p1_utm[2]), string.ascii_uppercase[int(p1_utm[3])])

                p2_utm = geo_matrix[c1[1]][c2[0]]
                p2_ll = utm.to_latlon(p2_utm[0], p2_utm[1], int(p2_utm[2]), string.ascii_uppercase[int(p2_utm[3])])

                p3_utm = geo_matrix[c2[1]][c2[0]]
                p3_ll = utm.to_latlon(p3_utm[0], p3_utm[1], int(p3_utm[2]), string.ascii_uppercase[int(p3_utm[3])])

                p4_utm = geo_matrix[c2[1]][c1[0]]
                p4_ll = utm.to_latlon(p4_utm[0], p4_utm[1], int(p4_utm[2]), string.ascii_uppercase[int(p4_utm[3])])

                to_file_bbox += '{"poly" : "\'POLYGON(( %s %s, %s %s, %s %s, %s %s, %s %s ))\'","type" : "%s","conf" : "%s" ,' % (
                    p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1],
                    names[mapeo[int(cls)]], conf)

                subtemp = temperaturas[c1[1]:c2[1], c1[0]:c2[0]]
                if np.prod(subtemp.shape) > 0:
                    tmin = np.min(subtemp)
                    tmax = np.max(subtemp)
                    tmean = np.mean(subtemp)
                    tstd = np.std(subtemp)
                else:
                    tmin = 0
                    tmax = 0
                    tmean = 0
                    tstd = 0


                to_file_bbox += f'"tmin": {tmin}, "tmax": {tmax}, "tmean": {tmean}, "tstd": {tstd}, '

                to_file_bbox += '"geo_json" : {"type": "Polygon","coordinates":[[  [%s, %s],[%s, %s],[%s, %s],[%s, %s],[%s, %s] ]] }, ' % (
                    p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1])

                _name = name.replace('.JPG', f'_{falla_id}.JPG')
                to_file_bbox += f'"name" : "{_name}", "dd": {dd}, "falla_id": {falla_id}' + '}'
                falla_id += 1
                to_file_bbox = to_file_bbox.replace('}{', '},{')

                imi = to_uint8g(temperaturas.copy())
                all_boxes = yolo_2_plot([{'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)}],
                                        [None for _ in range(len(names))], colors=paleta)
                imi = add_all_bboxes(imi, all_boxes, 2)
                im_name = path_save + f'/detect/{_name}'
                cv2.imwrite(im_name, imi)

            with open(path_save + f'/detect/{imp.split("/")[-1].replace(".JPG", ".json")}', 'w') as f:
                f.write('[')
                f.write(to_file_bbox)
                f.write(']')
            # imi = to_uint8g(im)

            # all_boxes = yolo_2_plot(pdata, [None for _ in range(len(names))], colors=paleta)
            # imi = add_all_bboxes(imi, all_boxes, 1)
            # im_name = path_save + f'/detect/{imp.split("/")[-1]}'
            # cv2.imwrite(im_name, imi)



def pre_proceso(im_path, path_save, desp_este, desp_norte, desp_yaw):

    i = 0
    #for imp, bp in zip(im_path, bbox_path):
    for imp in im_path:
        print("Procesando imagen: ", imp)
        file_path = path_save + f'/metadata/{imp.split("/")[-1].replace(".JPG", ".txt")}'

        try:
            file = open(file_path, 'r')
            metadata = json.load(file)
            print("metadata_encontrada ", file_path)
            file.close()
        except:
            metadata = get_string_meta_data(imp)

        metadata['offset_N'] = desp_norte
        metadata['offset_E'] = desp_este
        metadata['offset_yaw'] = desp_yaw
        file = open(file_path, 'w')
        file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
        file.close()

        matriz_geo = save_georef_matriz(metadata, desp_este, desp_norte, desp_yaw)
        geo_name = path_save + f'/georef_numpy/{imp.split("/")[-1].replace(".JPG", ".npy")}'
        np.save(geo_name, matriz_geo)

        # Cargar imagenes originales para arreglar distorcion de camara
        original_image = cv2.imread(imp)

        if metadata['Model'] == "MAVIC2-ENTERPRISE-ADVANCED":
            temperaturas = get_thermal_dji(imp)
            temperaturas = undistort_m2ea_th(temperaturas)
            original_image = undistort_m2ea_th(original_image)
        elif metadata['Model'] == "XT2":
            temperaturas = get_thermal_flir(imp)
            temperaturas = undistort_xt2(temperaturas)
            original_image = undistort_xt2(original_image)
        elif metadata['Model'] == "ZH20T":
            temperaturas = get_thermal_dji(imp)
            temperaturas = undistort_zh20t(temperaturas)
            original_image = undistort_zh20t(original_image)
        elif metadata['Model'] == "M3T":
            temperaturas = get_thermal_dji(imp)
            temperaturas = undistort_m3e_th(temperaturas)
            original_image = undistort_m3e_th(original_image)
        else:
            print("===================================================")
            print("CÁMARA NO DEFINIDA")
            return

        # Genera las imágenes para el CVAT
        imCVAT = to_uint8(temperaturas)
        splits = imp.split('/')
        if splits[-2] == 'T':
            name = 'v' + splits[-3].split(' ')[-1] + '_' + splits[-1]
        else:
            name = 'v' + splits[-2].split(' ')[-1] + '_' + splits[-1]
        cv2.imwrite(path_save + "/cvat/" + name, imCVAT)
        im_name_or = path_save + f'/original_img/{imp.split("/")[-1]}'
        cv2.imwrite(im_name_or, original_image, [cv2.IMWRITE_JPEG_QUALITY,100])
        im_name = path_save + f'/Temp/{imp.split("/")[-1]}'
        cv2.imwrite(im_name, to_uint8g(temperaturas.copy()))

        #with open(bp, 'r') as f:
        #    bboxes = f.readlines()

        geo_matrix = matriz_geo
        geo_matrix_shape = geo_matrix.shape
        conf = 1
        name = imp.split('/')[-1]

        #if bboxes.__len__() > 0:
        #    to_file_bbox = ''
        #    pdata = []
        #    tdata = []
        #    for _b in bboxes:
        #        cls, xx, yy, ww, hh = _b.split(" ")
        #        xx = float(xx)
        #        yy = float(yy)
        #        ww = float(ww)
        #        hh = float(hh)
        #        dd = (xx - 0.5) ** 2 + (yy - 0.5) ** 2

        #        if xx < 0.10 or xx > 0.9 or yy < 0.1 or yy > 0.9:
        #            continue

                # xx = 0.5
                # yy = 0.5
                # ww = 0.99999
                # hh = 0.99999

        #        pdata.append({'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)})

        #        c1, c2 = yolo_2_album(xx, yy, ww, hh)
        #        c1 = (int(c1[0] * 640), int(c1[1] * 512))
        #        c2 = (int(c2[0] * 640), int(c2[1] * 512))

        #        if c1[0] >= geo_matrix_shape[1]:
        #            c1 = (geo_matrix_shape[1] - 1, c1[1])
        #        if c2[0] >= geo_matrix_shape[1]:
        #            c2 = (geo_matrix_shape[1] - 1, c2[1])

        #        if c1[1] >= geo_matrix_shape[0]:
        #            c1 = (c1[0], geo_matrix_shape[0] - 1)
        #        if c2[1] >= geo_matrix_shape[0]:
        #            c2 = (c2[0], geo_matrix_shape[0] - 1)

        #        p1_utm = geo_matrix[c1[1]][c1[0]]
        #        p1_ll = utm.to_latlon(p1_utm[0], p1_utm[1], int(p1_utm[2]), string.ascii_uppercase[int(p1_utm[3])])

        #        p2_utm = geo_matrix[c1[1]][c2[0]]
        #        p2_ll = utm.to_latlon(p2_utm[0], p2_utm[1], int(p2_utm[2]), string.ascii_uppercase[int(p2_utm[3])])

        #        p3_utm = geo_matrix[c2[1]][c2[0]]
        #        p3_ll = utm.to_latlon(p3_utm[0], p3_utm[1], int(p3_utm[2]), string.ascii_uppercase[int(p3_utm[3])])

        #        p4_utm = geo_matrix[c2[1]][c1[0]]
        #        p4_ll = utm.to_latlon(p4_utm[0], p4_utm[1], int(p4_utm[2]), string.ascii_uppercase[int(p4_utm[3])])

        #        to_file_bbox += '{"poly" : "\'POLYGON(( %s %s, %s %s, %s %s, %s %s, %s %s ))\'","type" : "%s","conf" : "%s" ,' % (
        #            p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1],
        #            names[mapeo[int(cls)]], conf)

        #        subtemp = temperaturas[c1[1]:c2[1], c1[0]:c2[0]]
        #        if np.prod(subtemp.shape) > 0:
        #            tmin = np.min(subtemp)
        #            tmax = np.max(subtemp)
        #            tmean = np.mean(subtemp)
        #            tstd = np.std(subtemp)
        #        else:
        #            tmin = 0
        #            tmax = 0
        #            tmean = 0
        #            tstd = 0


        #        to_file_bbox += f'"tmin": {tmin}, "tmax": {tmax}, "tmean": {tmean}, "tstd": {tstd}, '

        #        to_file_bbox += '"geo_json" : {"type": "Polygon","coordinates":[[  [%s, %s],[%s, %s],[%s, %s],[%s, %s],[%s, %s] ]] }, ' % (
        #            p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1])

        #        _name = name.replace('.JPG', f'_{falla_id}.JPG')
        #        to_file_bbox += f'"name" : "{_name}", "dd": {dd}, "falla_id": {falla_id}' + '}'
        #        falla_id += 1
        #        to_file_bbox = to_file_bbox.replace('}{', '},{')

        #        imi = to_uint8g(temperaturas.copy())
        #        all_boxes = yolo_2_plot([{'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)}],
        #                                [None for _ in range(len(names))], colors=paleta)
        #        imi = add_all_bboxes(imi, all_boxes, 2)
        #        im_name = path_save + f'/detect/{_name}'
        #        cv2.imwrite(im_name, imi)

        #    with open(path_save + f'/detect/{imp.split("/")[-1].replace(".JPG", ".json")}', 'w') as f:
        #        f.write('[')
        #        f.write(to_file_bbox)
        #        f.write(']')
            # imi = to_uint8g(im)

            # all_boxes = yolo_2_plot(pdata, [None for _ in range(len(names))], colors=paleta)
            # imi = add_all_bboxes(imi, all_boxes, 1)
            # im_name = path_save + f'/detect/{imp.split("/")[-1]}'
            # cv2.imwrite(im_name, imi)




def desplazamiento():
    while True:
        try:
            desp_norte = float(input('Introduzca el desplazamiento hacia el NORTE en metros'))
            break
        except:
            desp_norte = ''
            continue
    while True:
        try:
            desp_este = float(input('Introduzca el desplazamiento hacia el ESTE en metros'))
            break
        except:
            desp_este = ''
            continue
    while True:
        try:
            desp_yaw = float(input('Introduzca la ROTACION en Grados Sentido horario'))
            break
        except:
            desp_yaw = ''
            continue
    while True:
        try:
            offset_vuelo = float(input('Introduzca ajuste de GPS en la dirección de vuelo (0 a 1)'))
            break
        except:
            offset_vuelo = ''
            continue
    while True:
        try:
            #offset_altura = float(input('Introduzca ajuste de ALTURA en metros'))
            offset_altura = input('Introduzca ajuste de ALTURA en metros')
            break
        except:
            offset_altura = ''
            continue
    return [desp_este, desp_norte, desp_yaw, offset_vuelo, offset_altura]


def real_detect_group():
    root = tk.Tk()
    root.withdraw()

    # Path de las imágenes a procesar (térmicas originales)
    path_save = filedialog.askdirectory(title='Seleccione path imagenes')

    real_detect(path_save)
    real_detect_filter(path_save)

    return path_save

def real_detect_filter(path):
    import shutil

    cardinal = cv2.imread('./punto-cardinal.png')
    mask = cardinal > 0

    def add(impath, angle,of_set_ang):
        im = cv2.imread(impath)
        '''
        if -99 < angle < -81:
            pass
        elif 81 < angle < 99:
            print('Rotado')
            im = cv2.rotate(im, cv2.ROTATE_180)
        im[:24, -24:, :] = im[:24, -24:, :] * (1 - mask) + mask * cardinal
        cv2.imwrite(impath, im)
        '''
        angle= (angle+of_set_ang)%360
        print(angle,impath.split('/')[-1])
        if -99 < angle < -81:
            pass
        elif 81 < angle < 99:
            print('Rotado')
            im = cv2.rotate(im, cv2.ROTATE_180)
        im[:24, -24:, :] = im[:24, -24:, :] * (1 - mask) + mask * cardinal
        cv2.imwrite(impath, im)


    # cambiar path
    # path = './reEtiquetasPLS/re_etiquetar_CP/'
    # path = 'E:/ProcesamientoEnel_2023-04/Imagenes/Domeyko/DMK/Z5/PV025PP/'

    path = path + '/'

    if not os.path.exists(path + 'para_subir'):
        os.mkdir(path + 'para_subir')
    else:
        shutil.rmtree(path + 'para_subir')
        os.mkdir(path + 'para_subir')

    with open(path + 'real_detect.json') as f:
        poligonos = json.load(f)

    for p in poligonos:
        name = p['name']
        shutil.copy(path + 'detect/' + name, path + 'para_subir/' + name)

        # TODO: Arreglar manejo de nombres de imágenes (acá y en todas partes) para que soporte cualquier nombre
        if name.find('_T_') >= 0:
            if name.split('_T_')[1].find('_') == -1:
                str_corr = ''
            else:
                str_corr = name.split('_T_')[1].split('_')[0]
                str_corr = '_' + str_corr
            real_name = name.split('_T_')[0] + '_T' + str_corr + '.txt'
        else:
            real_name = '_'.join(name.split('_')[0:-1]) + '.txt'

        with open(path + 'metadata/' + real_name, 'r') as f:
            data = json.load(f)
        angle = float((data['GimbalYawDegree']))
        res_angle=float((data["offset_yaw"]))
        add(path + 'para_subir/' + name, angle,res_angle)

    print('Done')

def real_detect_max_elim(path_save):
    polygons = []

    data_path = f'{path_save}/detect'
    write_path = path_save
    n_images = {}
    for file in os.listdir(data_path):
        ext = file.split('.')
        if ext[-1] == 'json':
            with open(data_path + '/' + file) as json_file:
                data_file = json.load(json_file)
                for data in data_file:
                    polygons.append({"data": data, "poly": ogr.CreateGeometryFromJson(str(data['geo_json']))})
                    if data['name'] in n_images.keys():
                        n_images[data['name']] += 1
                    else:
                        n_images[data['name']] = 1

    # Filtra por tipo de falla que no queremos reportar
    aux_polygons = [i for i in polygons if i['data']['type'] != "Tipo XX - Tracker fuera de posición"]
    polygons = aux_polygons

    # Esto no funciona porque itera sobre el arreglo que modifica
    #for p in polygons:
    #    if p['data']['type'] == 'Tipo IX - JunctionBoxCaliente':
    #        aux_polygons.pop(polygons.index(p))
    #    if p['data']['type'].split("-")[0] == "Tipo XX ":#- Tracker fuera de posición":
    #        aux_polygons.pop(polygons.index(p))

    polygons.sort(reverse=True, key=lambda x: (x['data']['tmax'], 0.5 - x['data']['dd'], n_images[x['data']['name']]))

    deberian = 2.8 * len(polygons) // 4
    aux = polygons.copy()
    for umb in range(1000, 0, -1):
        print(umb)
        umb = umb / 1000
        polygons = aux.copy()
        i = 0
        eliminados = 0
        # print('jiji', len(polygons))
        while i < len(polygons):
            j = i + 1
            while j < len(polygons):
                if polygons[i]["data"]["name"].split('_T')[0] != polygons[j]["data"]["name"].split('_T')[0] and \
                        polygons[i]["data"]["type"] == \
                        polygons[j]["data"]["type"]:
                    dist = polygons[i]['poly'].Distance(polygons[j]['poly'])
                    if -1 < dist < umb / 1000:  #### <----- Umbral de distancia en km
                        polygons.pop(j)
                        eliminados += 1
                        continue
                j += 1
            i += 1
        if eliminados <= deberian:
            break
    print('Poligonos: ', len(polygons))

    read_detec = open(write_path + '/real_detect.json', 'w')
    data = []
    for i, poly in enumerate(polygons):
        poly["data"].__delitem__('dd')
        poly["data"].__delitem__('falla_id')
        data.append(poly["data"])
    read_detec.write(json.dumps(data))
    read_detec.close()

def get_detections(path_save):
    data_path = f'{path_save}/detect'
    write_path = path_save

    polygons = []
    n_images = {}

    for file in os.listdir(data_path):
        ext = file.split('.')
        if ext[-1] == 'json':
            with open(data_path + '/' + file) as json_file:
                data_file = json.load(json_file)
                for data in data_file:
                    polygons.append({"data": data, "poly": ogr.CreateGeometryFromJson(str(data['geo_json']))})
                    i = len(polygons) - 1
                    polygons[i]['Names'] = '_'.join(polygons[i]['data']['name'].split('_')[:-1])
                    polygons[i]['falla'] = polygons[i]['data']['type']
                    polygons[i]['CentroidX'] = float(utm.from_latlon(polygons[i]['poly'].Centroid().GetPoints()[0][0],
                                                               polygons[i]['poly'].Centroid().GetPoints()[0][1])[0])
                    polygons[i]['CentroidY'] = float(utm.from_latlon(polygons[i]['poly'].Centroid().GetPoints()[0][0],
                                                         polygons[i]['poly'].Centroid().GetPoints()[0][1])[1])

                    if data['name'] in n_images.keys():
                        n_images[data['name']] += 1
                    else:
                        n_images[data['name']] = 1

    return polygons, n_images

def real_detect_new(path_save,pc,cpc):
    import pandas as pd
    import shutil
    import os
    from time import sleep

    data_path = f'{path_save}/detect'
    write_path = path_save

    #Parametros de desicion:
    CANT_PUNTOS_CALIENTES = cpc #desde 2 en adelante
    sensX = 3  # parametro sensibilidad X para union de puntos calientes
    sensY = 10  # parametro sensibilidad Y para union de puntos calientes


    polygons, n_images = get_detections(path_save)
    if pc=='y':
        testx = list()
        testy = list()
        name0 = list()
        name1 = list()
        ty = list()
        for n in range(len(polygons)):
            testx.append(polygons[n]['CentroidX'])
            testy.append(polygons[n]['CentroidY'])
            name0.append(str(n))
            name1.append('_'.join(polygons[n]['Names'].split('_')[:-2]))
            ty.append(polygons[n]['falla'])
        dat = pd.DataFrame(list(zip(testx, testy, name0, name1, ty)), columns=['X', 'Y', 'num', 'NAME', 'falla'])
        dat.to_csv('orden_paneles.csv', index=False)
        TABLA = pd.pivot_table(dat, columns=['NAME', 'falla'], aggfunc='size').reset_index()
        TABLA.columns = ['ids', 'falla', 'cant']
        TABLA = TABLA.sort_values('cant').copy()
        TABLA.to_csv('ids_img_and_cant.csv')

        # TODO: detectar puntos caliente spor suciedod en cada imagen
        for idss, fail, can in zip(TABLA.ids, TABLA.falla,
                                   TABLA.cant):  # determinar la cantidad de celulas calientes que tiene la imagen
            for n in range(len(polygons)):
                if (idss in polygons[n]['Names']):
                    if 'Tipo VI - CelulaCaliente' in fail:
                        if can > CANT_PUNTOS_CALIENTES:
                            polygons[n]['band'] = True
                            # polygons[n]['data']['type'] = "Tipo X - Múltiples puntos calientes por suciedad"

                            # polygons[n]['falla'] = "Tipo X - Múltiples puntos calientes por suciedad"
                        else:
                            polygons[n]['band'] = False
                    else:
                        polygons[n]['band'] = False
        aux_ = [x for x in polygons if x['band'] == True]

        aux_.sort(reverse=False, key=lambda x: (x['Names'], x['CentroidY']))
        if len(aux_):
            centrX = []
            centrY = []
            nombre = []
            comparacion = aux_[0]['Names']
            #centrX.append([aux_[0]['CentroidX']])
            #centrY.append([aux_[0]['CentroidY']])
            #nombre.append(['inicial'])
            for pol in aux_:
                if comparacion == pol['Names']:
                    if pol['data']['type'] == "Tipo VI - CelulaCaliente":
                        # Comparar max y min de listas
                        usado = False
                        recuadro = -1
                        for cenX, cenY in zip(centrX, centrY):
                            detec_X = [False, 0]
                            detec_Y = [False, 0]
                            recuadro += 1
                            for cenX_it in cenX:
                                if abs(float(pol['CentroidX']) - float(cenX_it)) < sensX:
                                    detec_X = [True, recuadro]
                            for cenY_it in cenY:
                                if abs(float(pol['CentroidY']) - float(cenY_it)) < sensY:
                                    detec_Y = [True, recuadro]

                            if (detec_X[0] and detec_Y[0]):
                                usado = True
                                print('aumento')
                                centrX[recuadro].append(pol['CentroidX'])
                                centrY[recuadro].append(pol['CentroidY'])
                                nombre[recuadro].append(pol['data']['name'])
                                break
                        if not usado:
                            centrX.append([pol['CentroidX']])
                            centrY.append([pol['CentroidY']])
                            nombre.append([pol['data']['name']])
                else:

                    for centrX_, centrY_, nombre_ in zip(centrX, centrY, nombre):
                        if len(centrX_) > CANT_PUNTOS_CALIENTES:
                            # Si son mas de 3 centroides, crear los poligonos correspondientes a suciedad
                            # tambien eliminar cada uno de estos poligonos previamente usados
                            delin = 0
                            eliminar = []
                            for it in polygons:
                                if it['data']['name'] in nombre_:
                                    eliminar.append(delin)
                                delin += 1
                            borrados = 0
                            for drop in eliminar:
                                polygons.remove(polygons[drop - borrados])
                                borrados += 1

                            Xmax = max(centrX_) + 0.2
                            Xmin = min(centrX_) - 0.2
                            Ymax = max(centrY_) + 0.2
                            Ymin = min(centrY_) - 0.2

                            #
                            #   V1-----------V2
                            #   -            -
                            #   -            -
                            #   -            -
                            #   V4-----------V3
                            #
                            V1 = list(utm.to_latlon(Xmin, Ymax, 19, northern=False))
                            V2 = list(utm.to_latlon(Xmax, Ymax, 19, northern=False))
                            V3 = list(utm.to_latlon(Xmax, Ymin, 19, northern=False))
                            V4 = list(utm.to_latlon(Xmin, Ymin, 19, northern=False))

                            temporal = [x for x in aux_ if (comparacion == x['Names'])]

                            geo_json_new = {'type': 'Polygon', 'coordinates': [[V1, V2, V3, V4, V1]]}
                            temporal[0]['data']['geo_json'] = geo_json_new
                            temporal[0]['poly'] = ogr.CreateGeometryFromJson(str(geo_json_new))
                            # temporal[0]['data']['poly'] = temporal[0]['poly'].ExportToWkt()
                            temporal[0]['data'][
                                'poly'] = f"\'POLYGON(( {V1[0]} {V1[1]}, {V2[0]} {V2[1]}, {V3[0]} {V3[1]}, {V4[0]} {V4[1]}, {V1[0]} {V1[1]} ))\'"
                            temporal[0]['falla'] = "Tipo XI - Múltiples puntos calientes por suciedad"
                            temporal[0]['data']['type'] = "Tipo XI - Múltiples puntos calientes por suciedad"
                            polygons.append(temporal[0])
                            shutil.copy(path_images + f'/{comparacion.split("_")[2]}/{comparacion}.JPG',
                                        path_save + f'/detect/{temporal[0]["data"]["name"]}')
                    # inicializar siguiente imagen
                    comparacion = pol['Names']
                    centrX = [[pol['CentroidX']]]
                    nombre = [[pol['data']['name']]]
                    centrY = [[pol['CentroidY']]]



        dir_new = rf'{path_save}/detect_lix'
        os.makedirs(dir_new, exist_ok=True)
        for move in polygons:
            print(f'Copiando {move["data"]["name"]}')
            shutil.copy(data_path + rf'/{move["data"]["name"]}', dir_new + rf'/{move["data"]["name"]}')
            shutil.copy(data_path + rf'/{move["Names"]}.json', dir_new + rf'/{move["Names"]}.json')

        # TODO: ordenar por lineas y orden de adquisición según línea de vuelo. Inicialmente separar por líneas de tracker según CSV

    # Para ordenar poligonos, es necesario crear areas de cada linea para cada PV, para ello utilizar el archivo DMK_polyxline.csv, luego comparar si pertenecen o no
    deberian = 2.8 * len(polygons) // 4
    aux = polygons.copy()
    umb = 0.000002  # 20cm (si unidad de distancia entre polígonos es 100km)
    polygons = aux.copy()
    i = 0
    eliminados = 0
    # print('jiji', len(polygons))
    while i < len(polygons):
        j = i + 1
        while j < len(polygons):
            # PUNTO DE DEPURACION
            # if polygons[j]["data"]["name"].split('_T')[0] == 'PCP_CT1_1_DJI_0760':
            #    dep = 1
            if polygons[i]["data"]["name"].split('_T')[0] != polygons[j]["data"]["name"].split('_T')[0] and \
                    polygons[i]["data"]["type"] == \
                    polygons[j]["data"]["type"]:

                dist = polygons[i]['poly'].Distance(polygons[j]['poly'])
                if -1 < dist < umb:  #### <----- Distancia parece tener unidad de 100km
                    polygons.pop(j)
                    eliminados += 1
                    continue
            j += 1
        i += 1
    #    if eliminados <= deberian:
    #        break
    print('Poligonos: ', len(polygons))

    # TODO: agregar hallazgo de puntos calientes por suciedad con bounding box que envuelva a todos (eliminar hallazgos puntos calientes)

    # TODO: eliminar duplicados

    read_detec = open(write_path + '/real_detect.json', 'w')
    data = []
    for i, poly in enumerate(polygons):
        if 'Tipo X -' in poly["data"]['type']:
            continue
        try:
            poly["data"].__delitem__('dd')
        except:
            pass
        try:
            poly["data"].__delitem__('falla_id')
        except:
            pass
        data.append(poly["data"])
    read_detec.write(json.dumps(data))
    read_detec.close()
    print(len(polygons))

def real_detect(path_save):
    polygons = []

    data_path = f'{path_save}/detect' #detecciones
    write_path = path_save #lugar de escritura
    n_images = {} #contador de
    for file in os.listdir(data_path): #iterar
        ext = file.split('.')
        if ext[-1] == 'json':
            with open(data_path + '/' + file) as json_file:
                data_file = json.load(json_file)
                for data in data_file:
                    polygons.append({"data": data, "poly": ogr.CreateGeometryFromJson(str(data['geo_json']))})
                    if data['name'] in n_images.keys():
                        n_images[data['name']] += 1
                    else:
                        n_images[data['name']] = 1

    # Filtra por tipo de falla que no queremos reportar
    aux_polygons = [i for i in polygons if i['data']['type'] != "Tipo XX - Tracker fuera de posición"]
    polygons = aux_polygons

    # Esto no funciona porque itera sobre el arreglo que modifica
    #for p in polygons:
    #    if p['data']['type'] == 'Tipo IX - JunctionBoxCaliente':
    #        aux_polygons.pop(polygons.index(p))
    #    if p['data']['type'].split("-")[0] == "Tipo XX ":#- Tracker fuera de posición":
    #        aux_polygons.pop(polygons.index(p))

    polygons.sort(reverse=True, key=lambda x: (x['data']['tmax'], 0.5 - x['data']['dd'], n_images[x['data']['name']]))

    deberian = 2.8 * len(polygons) // 4
    aux = polygons.copy()
    #for umb in range(1000, 0, -1):
    #print(umb)
    #umb = umb / 1000

    umb = 0.000002 # 20cm (si unidad de distancia entre polígonos es 100km)
    polygons = aux.copy()
    i = 0
    eliminados = 0
    # print('jiji', len(polygons))
    while i < len(polygons):
        j = i + 1
        while j < len(polygons):
            # PUNTO DE DEPURACION
            #if polygons[j]["data"]["name"].split('_T')[0] == 'PCP_CT1_1_DJI_0760':
            #    dep = 1
            if polygons[i]["data"]["name"].split('_T')[0] != polygons[j]["data"]["name"].split('_T')[0] and \
                    polygons[i]["data"]["type"] == \
                    polygons[j]["data"]["type"]:

                dist = polygons[i]['poly'].Distance(polygons[j]['poly'])
                if -1 < dist < umb:  #### <----- Distancia parece tener unidad de 100km
                    polygons.pop(j)
                    eliminados += 1
                    continue
            j += 1
        i += 1
    #    if eliminados <= deberian:
    #        break
    print('Poligonos: ', len(polygons))

    read_detec = open(write_path + '/real_detect.json', 'w')
    data = []
    for i, poly in enumerate(polygons):
        poly["data"].__delitem__('dd')
        poly["data"].__delitem__('falla_id')
        data.append(poly["data"])
    read_detec.write(json.dumps(data))
    read_detec.close()

def generar_kml_imagenes(desp_este, desp_norte, desp_yaw, path_save, mode='Temp'):

    apellido = "_DN" + str(desp_norte) + "DE" + str(desp_este)
    if mode == "Temp":
        path_imagenes = f"{path_save}/{mode}/*.JPG"
    else:
        path_imagenes = f"{path_save.replace('PP','P')}/*.JPG"

    with open(path_save + apellido + '.kml', 'w') as file:
        a = f'''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
    <Folder>
        <name>{apellido}</name>
        '''
        file.write(a)
        for f_name in glob(path_imagenes):
            all_path = f_name
            nombre = os.path.basename(os.path.normpath(f_name)).replace(".JPG", "")

            # with open(f"C:/Users/vicen/OneDrive/Escritorio/detect/{nombre}.txt")  as f1:
            # with open(f"C:/Users/vicen/OneDrive/Escritorio/Adentu/Solcito/finis/metadata/{nombre}.txt")  as f1:
            # with open(f"C:/Users/vicen/OneDrive/Escritorio/Adentu/Solcito/la_silla_pros/metadata/{nombre}.txt")  as f1:
            # with open(f"C:/Users/vicen/OneDrive/Escritorio/Adentu/Solcito/superposicion_de_imagenes/metadata T/{nombre}.txt")  as f1:
            with open(f"{path_save}/metadata/{nombre}.txt") as f1:
                data2 = json.load(f1)

            m = save_georef_matriz(data2, desp_este, desp_norte, desp_yaw)
            p1_ll = utm.to_latlon(m[0][0][0], m[0][0][1], int(m[0][0][2]), string.ascii_uppercase[int(m[0][0][3])])
            p2_ll = utm.to_latlon(m[0][-1][0], m[0][-1][1], int(m[0][-1][2]), string.ascii_uppercase[int(m[0][-1][3])])
            p3_ll = utm.to_latlon(m[-1][-1][0], m[-1][-1][1], int(m[-1][-1][2]),
                                  string.ascii_uppercase[int(m[-1][-1][3])])
            p4_ll = utm.to_latlon(m[-1][0][0], m[-1][0][1], int(m[-1][0][2]), string.ascii_uppercase[int(m[-1][0][3])])

            cordinates = f"{str(p4_ll[1])},{str(p4_ll[0])},0 {str(p3_ll[1])},{str(p3_ll[0])},0 {str(p2_ll[1])},{str(p2_ll[0])},0 {str(p1_ll[1])},{str(p1_ll[0])},0 "

            a = f'''<GroundOverlay>
            <name>{apellido + nombre}</name>
            <Icon>
                <href>{all_path}</href>
                <viewBoundScale>0.75</viewBoundScale>
            </Icon>
            <gx:LatLonQuad>
                <coordinates>
                    {cordinates} 
                </coordinates>
            </gx:LatLonQuad>
        </GroundOverlay>
        '''
            file.write(a)
        a = '''</Folder>
    </kml>'''
        file.write(a)


def ajuste_posicion_imagenes(path_im, desp_este, desp_norte, desp_yaw, offset_vuelo, offset_altura, in_vuelo='', mode='Temp'):

    # Si no se recibe el número de vuelo se solicita al usuario a través de la consola
    if in_vuelo == '':
        seleccion = input('''
            ============================================================================
            GENERAR KML
            ============================================================================
            Elija vuelo a procesar:
                0 Todos
                1, 2, ...
                x Salir 
    
                Introduzca su opción:  ''')

        if (seleccion == "x"):
            return
        vuelo_sel = seleccion
    else:
        vuelo_sel = in_vuelo


    if mode == "Temp":
        path_imagenes = f"{path_save}/{mode}/*.JPG"
    else:
        path_imagenes = f"{path_save.replace('PP','P')}/*.JPG"

    with open(path_save + '/' + path_save.split('/')[-1] + '.kml', 'w') as file:
        a = f'''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
    <Folder>
        <name>{path_save.split('/')[-1]}</name>
        '''
        file.write(a)
        vuelo_ant = ''
        im_pos_ant = None
        im_time_ant = None
        for f_name in glob(path_imagenes):
            all_path = f_name
            nombre = os.path.basename(os.path.normpath(f_name)).replace(".JPG", "")

            # Obtiene el número de vuelo desde el nombre de archivo
            vuelo = nombre.split('_')[3]
            if vuelo == 'DJI':
                vuelo = nombre.split('_')[2]

            try:
                file = open(file_path, 'r')
                metadata = json.load(file)
                print("metadata_encontrada ", file_path)
                file.close()


            except:
                metadata = get_string_meta_data(imp)

            # Carga la metadata de la imagen
            str_metada_file = f"{path_save}/metadata/{nombre}.txt"
            with open(str_metada_file) as metadata_file:
                data2 = json.load(metadata_file)

            # Se calcula la dirección del vuelo
            im_pos = get_image_pos_utm(data2)
            im_time = get_image_time(data2)
            if im_pos_ant is not None:
                delta_x = (im_pos[0] - im_pos_ant[0])
                delta_y = (im_pos[1] - im_pos_ant[1])
                delta_t = (im_time - im_time_ant).total_seconds()
                speed_x = delta_x/delta_t
                speed_y = delta_y / delta_t
            else:
                delta_x = 0
                delta_y = 0
                speed_x = 0
                speed_y = 0
            im_pos_ant = im_pos
            im_time_ant = im_time

            if abs(speed_x) < 0.5:
                if speed_y < -0.5:
                    dir_vuelo = 'S'
                elif speed_y > 0.5:
                    dir_vuelo = 'N'
                else:
                    dir_vuelo = '-'
            else:
                dir_vuelo = '-'


            # Se actualiza el desplazamiento a los vuelos que correspondan
            if (desp_este != '') and (desp_norte != '') and (desp_yaw != ''):
                if (vuelo == vuelo_sel) or (vuelo_sel == '0'):
                    data2['offset_N'] = desp_norte
                    data2['offset_E'] = desp_este
                    data2['offset_yaw'] = desp_yaw
                    data2['offset_altura'] = offset_altura
                    data2['desface_gps'] = offset_vuelo

                    # Actualiza los valores en el archivo de metada
                    metadata_file = open(str_metada_file, 'w')
                    metadata_file.write(json.dumps(data2, indent=4, sort_keys=True, default=str))
                    metadata_file.close()

            # Ajuste de error GPS según dirección de vuelo
            offset_N = data2['offset_N'] - delta_y * data2['desface_gps']
            offset_E = data2['offset_E'] - delta_x * data2['desface_gps']

            # Calcula la matriz de posición
            m = save_georef_matriz(data2, offset_E, offset_N, data2['offset_yaw'], data2['offset_altura'])
            p1_ll = utm.to_latlon(m[0][0][0], m[0][0][1], int(m[0][0][2]), string.ascii_uppercase[int(m[0][0][3])])
            p2_ll = utm.to_latlon(m[0][-1][0], m[0][-1][1], int(m[0][-1][2]), string.ascii_uppercase[int(m[0][-1][3])])
            p3_ll = utm.to_latlon(m[-1][-1][0], m[-1][-1][1], int(m[-1][-1][2]),
                                  string.ascii_uppercase[int(m[-1][-1][3])])
            p4_ll = utm.to_latlon(m[-1][0][0], m[-1][0][1], int(m[-1][0][2]), string.ascii_uppercase[int(m[-1][0][3])])

            # Coordenadas para el kml
            cordinates = f"{str(p4_ll[1])},{str(p4_ll[0])},0 {str(p3_ll[1])},{str(p3_ll[0])},0 {str(p2_ll[1])},{str(p2_ll[0])},0 {str(p1_ll[1])},{str(p1_ll[0])},0 "

            txt_desplazamiento = "_DN" + str(data2['offset_N']) + "DE" + str(data2['offset_E']) + "DY" + str(
                data2['offset_yaw'])
            if vuelo != vuelo_ant:
                if vuelo_ant != '':
                    a = f'''</Folder>'''
                    file.write(a)

                a = f'''<Folder>
                        <name>{vuelo} - {txt_desplazamiento}</name>
                        '''
                file.write(a)
                vuelo_ant = vuelo

            txt_href = all_path.split('/')[-1]
            a = f'''<GroundOverlay>
            <name>{nombre + txt_desplazamiento}</name>
            <Icon>
                <href>{txt_href}</href>
                <viewBoundScale>0.75</viewBoundScale>
            </Icon>
            <gx:LatLonQuad>
                <coordinates>
                    {cordinates} 
                </coordinates>
            </gx:LatLonQuad>
        </GroundOverlay>
        '''
            file.write(a)
        a = '''</Folder>
        </Folder>
    </kml>'''
        file.write(a)


def generar_kml_imagenes_vuelos(path_save, desp_este, desp_norte, desp_yaw, offset_vuelo, offset_altura, in_vuelo='',
                                mode='Temp'):
    # Si no se recibe el número de vuelo se solicita al usuario a través de la consola
    if in_vuelo == '':
        seleccion = input('''
            ============================================================================
            GENERAR KML
            ============================================================================
            Elija vuelo a procesar:
                0 Todos
                1, 2, ...
                x Salir 

                Introduzca su opción:  ''')

        if (seleccion == "x"):
            return
        vuelo_sel = seleccion
    else:
        vuelo_sel = in_vuelo

    modo_altura = "relativo"
    if offset_altura != '':
        if offset_altura.find("f") >= 0:
            modo_altura = "fijo"
            offset_altura = float(offset_altura.split("f")[-1])
        else:
            offset_altura = float(offset_altura)

    path_imagenes_temp = f"{path_save}/{mode}/*.JPG"
    path_imagenes = f"{path_save.replace('PP', '')}/**/*.JPG"

    with open(path_save + '/' + path_save.split('/')[-1] + '.kml', 'w') as file:
        a = f'''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
    <Folder>
        <name>{path_save.split('/')[-1]}</name>
        '''
        file.write(a)
        vuelo_ant = ''
        im_pos_ant = None
        im_time_ant = None
        delta_t = None
        for f_name in glob(path_imagenes):
            #all_path = f_name
            all_path = os.path.normpath(f_name)
            nombre = os.path.basename(os.path.normpath(f_name)).replace(".JPG", "")

            # Obtiene el número de vuelo desde la carpeta original
            vuelo = all_path.split('\\')[-2]
            #if vuelo == 'DJI':
            #    vuelo = nombre.split('_')[2]

            # Carga la metadata de la imagen
            str_metada_file = f"{path_save}/metadata/{nombre}.txt"
            with open(str_metada_file) as metadata_file:
                data2 = json.load(metadata_file)

            # Se calcula la dirección del vuelo
            im_pos = get_image_pos_utm(data2)
            im_time = get_image_time(data2)
            if im_pos_ant is not None:
                delta_x = (im_pos[0] - im_pos_ant[0])
                delta_y = (im_pos[1] - im_pos_ant[1])
                delta_t = (im_time - im_time_ant).total_seconds()
                speed_x = delta_x / delta_t
                speed_y = delta_y / delta_t
            else:
                delta_x = 0
                delta_y = 0
                speed_x = 0
                speed_y = 0
            im_pos_ant = im_pos
            im_time_ant = im_time

            if abs(speed_x) < 0.5:
                if speed_y < -0.5:
                    dir_vuelo = 'S'
                elif speed_y > 0.5:
                    dir_vuelo = 'N'
                else:
                    dir_vuelo = '-'
            else:
                dir_vuelo = '-'

            # Se actualiza el desplazamiento a los vuelos que correspondan
            if (desp_este != '') and (desp_norte != '') and (desp_yaw != ''):
                if (vuelo == vuelo_sel) or (vuelo_sel == '0'):
                    data2['offset_N'] = desp_norte
                    data2['offset_E'] = desp_este
                    data2['offset_yaw'] = desp_yaw
                    data2['offset_altura'] = offset_altura
                    data2['desface_gps'] = offset_vuelo
                    data2['modo_altura'] = modo_altura
                    if (delta_t is not None) and (abs(delta_t) < 8):
                        data2['offset_N_tot'] = data2['offset_N'] - delta_y * data2['desface_gps']
                        data2['offset_E_tot'] = data2['offset_E'] - delta_x * data2['desface_gps']
                    else:
                        data2['offset_N_tot'] = data2['offset_N']
                        data2['offset_E_tot'] = data2['offset_E']

                    # Actualiza los valores en el archivo de metada
                    metadata_file = open(str_metada_file, 'w')
                    metadata_file.write(json.dumps(data2, indent=4, sort_keys=True, default=str))
                    metadata_file.close()

            #if abs(delta_t) < 8:
            #    # Ajuste de error GPS según dirección de vuelo
            #    offset_N = data2['offset_N'] - delta_y * data2['desface_gps']
            #    offset_E = data2['offset_E'] - delta_x * data2['desface_gps']

            #PUNTO DE DEPURACIÓN
            if nombre == "PCP_CT6-7_1_DJI_0600_T":
                dep = 0

            # Calcula la matriz de posición
            m = save_georef_matriz(data2, data2['offset_E_tot'], data2['offset_N_tot'], data2['offset_yaw'], data2['offset_altura'], modo_altura)
            p1_ll = utm.to_latlon(m[0][0][0], m[0][0][1], int(m[0][0][2]), string.ascii_uppercase[int(m[0][0][3])])
            p2_ll = utm.to_latlon(m[0][-1][0], m[0][-1][1], int(m[0][-1][2]), string.ascii_uppercase[int(m[0][-1][3])])
            p3_ll = utm.to_latlon(m[-1][-1][0], m[-1][-1][1], int(m[-1][-1][2]),
                                  string.ascii_uppercase[int(m[-1][-1][3])])
            p4_ll = utm.to_latlon(m[-1][0][0], m[-1][0][1], int(m[-1][0][2]), string.ascii_uppercase[int(m[-1][0][3])])

            # Coordenadas para el kml
            cordinates = f"{str(p4_ll[1])},{str(p4_ll[0])},0 {str(p3_ll[1])},{str(p3_ll[0])},0 {str(p2_ll[1])},{str(p2_ll[0])},0 {str(p1_ll[1])},{str(p1_ll[0])},0 "

            txt_desplazamiento = "_DN" + str(data2['offset_N']) + \
                                 "_DE" + str(data2['offset_E']) + \
                                 "_DY" + str(data2['offset_yaw']) + \
                                 "_DV" + str(data2['desface_gps']) + \
                                 "_DA" + str(data2['offset_altura']) + \
                                 "_MA" + str(data2['modo_altura'])
            if vuelo != vuelo_ant:
                if vuelo_ant != '':
                    a = f'''</Folder>'''
                    file.write(a)

                a = f'''<Folder>
                        <name>{vuelo} - {txt_desplazamiento}</name>
                        '''
                file.write(a)
                vuelo_ant = vuelo

            txt_href = f'{mode}/{nombre}.JPG'
            print(nombre)
            a = f'''<GroundOverlay>
            <name>{nombre + txt_desplazamiento}</name>
            <Icon>
                <href>{txt_href}</href>
                <viewBoundScale>0.75</viewBoundScale>
            </Icon>
            <gx:LatLonQuad>
                <coordinates>
                    {cordinates} 
                </coordinates>
            </gx:LatLonQuad>
        </GroundOverlay>
        '''
            file.write(a)
        a = '''</Folder>
        </Folder>
    </kml>'''
        file.write(a)

    print("FIN GENERACIÓN KML")
    print("Parámetros: " + txt_desplazamiento)
    print("----------------------------------------------------------------------")

def generar_kml_fallas(path_save):
    kml = simplekml.Kml()

    for f_name in glob(f'{path_save}/detect/*.json'):

        with open(f_name) as file:
            data = json.load(file)
            numero_de_falla = 1
            for poligono in data:
                # print(poligono["name"])
                #
                # print(poligono["type"])

                cordenadas = poligono["poly"]

                listcordenadas_2 = []

                listcordenadas = cordenadas.split()
                listcordenadas = listcordenadas[1:-1]
                for i in range(int(len(listcordenadas) / 2)):
                    listcordenadas_2.append((float(listcordenadas[2 * i + 1].replace(",", "")),
                                             (float(listcordenadas[2 * i].replace(",", "")))))

                print(listcordenadas_2)
                pol = kml.newpolygon(name=poligono["name"] + "_" + str(numero_de_falla),

                                     outerboundaryis=listcordenadas_2)
                # innerboundaryis=[(18.43348,-33.98985),(18.43387,-33.99004),(18.43410,-33.98972),
                #                       (18.43371,-33.98952),(18.43348,-33.98985)])

                pol.style.polystyle.color = simplekml.Color.red
                pol.style.polystyle.outline = 0
                pol.style.polystyle.fill = 1

                numero_de_falla = numero_de_falla + 1

    kml.save(path_save + "/fallas.kml")


def generar_kml_fallas2(path_save):
    kml = simplekml.Kml()

    with open(f'{path_save}/real_detect.json') as file:
        datas = json.load(file)
        numero_de_falla = 1
        for poligono in datas:
            # print(poligono["name"])
            #
            # print(poligono["type"])

            cordenadas = poligono["poly"]

            listcordenadas_2 = []

            listcordenadas = cordenadas.split()
            listcordenadas = listcordenadas[1:-1]
            for i in range(int(len(listcordenadas) / 2)):
                try:
                    listcordenadas_2.append((float(listcordenadas[2 * i + 1].replace(",", "")),
                                             (float(listcordenadas[2 * i].replace(",", "")))))
                except: continue

            print(listcordenadas_2,poligono["name"])
            pol = kml.newpolygon(name=poligono["name"] + "_" + str(numero_de_falla),

                                 outerboundaryis=listcordenadas_2)
            # innerboundaryis=[(18.43348,-33.98985),(18.43387,-33.99004),(18.43410,-33.98972),
            #                       (18.43371,-33.98952),(18.43348,-33.98985)])

            pol.style.polystyle.color = simplekml.Color.red
            pol.style.polystyle.outline = 0
            pol.style.polystyle.fill = 1

            numero_de_falla = numero_de_falla + 1

    kml.save(path_save + "/fallas2.kml")


if __name__ == '__main__':

    im_path = ''
    path_obj = ''

    while True:
        seleccion = input('''
        
        
    Elija su opcion:
        h Ayuda
        00 Ordenar Imágenes
        0 Pre-Proceso (CVAT + metadata + matrices)
        01 Ajustar posiciones (KML)
        02 Revisar posiciones imágenes (KML)
        1 Procesar
        2 Generar Real detect
        9 Cambiar Directorio 
        x Salir 
        
        
        Introduzca su opción:  ''')

        if (seleccion == "h"):
            print('''En este Script realiza las siguientes funcionalidades 
            
00 Ordenar imágenes:

    Separa y ordena en zonas y vuelos las imágenes descargadas para poder procerlas.
    
0 Pre-Proceso (CVAT + metadata + matrices):

    Ejecuta el preproceso de las imágenes generando todo lo necesario para el procesamiento posterior.

01 Ajustar posiciones (KML):

    Recibe valores de desplazamiento para ajustar la posición de las imágenes. Crea un KML con las imagenes ubicadas segun la metadata y el ajuste agregado. Los desplazamiento se actualizan en la metada de la imagen.
    
02 Revisar posiciones imágenes (KML)

    Crea un KML con las imagenes ubicadas segun la metadata de las imagenes, incluyendo el desplazamiento. Las imágenes se muestran en la posición en que serán procesadas.
    
    
1 Procesar: 

    El script genera una carpeta con sufijo 'P'(nombre de carpeta igual al de las originales) para guardar los las
    detecciones, metadata, georef y json de detecciones. 

    Crea un KML con las imagenes ubicadas segun la metadata de las imagenes con el desplazamiento agregado
    
            
2 Generar Real detect:

    Genera un unico json con los poligonos filtrados ppor ubicacion e imagen para cargarlo a la plataforma

    Crea un KML de las Fallas sin desplazamiento a partir del detect se guarda en la misma carpeta en la que se corre el codigo 


9 Cambiar Directorio:

    Seleccionar imagenes a procesar, originales(no las del cvat, del cvat deben descargarse solo las etiquetas, de lo contrario genera un error al manejar grandes cantidades de imagenes)
    Seleccionar la carpeta de etiquedas descagadas, (en formato yolo la carpeta se llama obj_train_data)
    Seleccionar archivo obj.name descargado del cvat            
            ''')

        if (seleccion == "100"):
            print('jijijiji')
            print("Iniciando Procesamiento")
            [desp_este, desp_norte, desp_yaw] = desplazamiento()
            detect2(im_path, path_labels, path_obj, path_save, names, desp_este, desp_norte, desp_yaw)

            print("Generarando KML imagenes")
            generar_kml_imagenes(desp_este, desp_norte, desp_yaw, path_save)
            print("KML imagenes Listo")

        if (seleccion == "00"):
            #if im_path == '':
            #    [im_path, path_save] = inicializar(0)
            print("Ordenando Imágenes...")
            ordenar_imagenes()
            print("Imágenes Ordenadas")

        if (seleccion == "0"):
            [im_path, path_save] = inicializar(0)
            #[desp_este, desp_norte, desp_yaw] = desplazamiento()
            pre_proceso(im_path, path_save, 0, 0, 0)
            print("Generando KML imagenes")
            generar_kml_imagenes_vuelos(path_save, 0, 0, 0, 0, '0', '0')
            print("KML imagenes Listo")

        if (seleccion == "01"):
            if im_path == '':
                [im_path, path_save] = inicializar(0)
            [desp_este, desp_norte, desp_yaw, offset_vuelo, offset_altura] = desplazamiento()
            print("Generando KML imagenes")
            generar_kml_imagenes_vuelos(path_save, desp_este, desp_norte, desp_yaw, offset_vuelo, offset_altura)
            print("KML imagenes Listo")

        if (seleccion == "02"):
            if im_path == '':
                [im_path, path_save] = inicializar(0)
            #[desp_este, desp_norte, desp_yaw] = desplazamiento()
            print("Generando KML imagenes")
            # KML sin modificaciones a los desplazamientos y para todos los vuelos
            generar_kml_imagenes_vuelos(path_save, '', '', '', '', '', '0')
            print("KML imagenes Listo")

        if (seleccion == "1"):

            [im_path, path_labels, path_obj, path_save, names] = inicializar()
            print("Iniciando Procesamiento")
            detect_fast(im_path, path_labels, path_obj, path_save, names)
            print("Procesamiento Listo")
            print("Generando KML fallas")
            generar_kml_fallas(path_save)
            print("KML fallas Listo")

        if (seleccion == "2"):
            if im_path == '':
                [im_path, path_labels, path_obj, path_save, names] = inicializar()
            print("Iniciando REAL_DETECT")
            real_detect(path_save)
            print("REAL_DETECT Listo")
            print("Generarando KML fallas")
            #generar_kml_fallas(path_save)
            generar_kml_fallas2(path_save)
            # generar_kml_imagenes(desp_este, desp_norte, desp_yaw, path_save, mode='detect')
            print("KML fallas Listo")

        if (seleccion == "2g"):

            print("Iniciando REAL_DETECT")
            path_save = real_detect_group()
            print("REAL_DETECT Listo")
            print("Generarando KML fallas")
            generar_kml_fallas(path_save)
            generar_kml_fallas2(path_save)
            # generar_kml_imagenes(desp_este, desp_norte, desp_yaw, path_save, mode='detect')
            print("KML fallas Listo")

        if (seleccion == "2test"):
            while True:
                p_c=input('¿Desea agrupar puntos calientes por suciedad? (y/n): ')
                if p_c=='y':
                    while True:
                        cp_c=input('Ingrese cantidad minima de puntos calientes cercanos considerados suciedad: ')
                        try:
                            print(int(cp_c))
                            break
                        except:
                            print('Ingrese un valor numerico entero')
                    break


                if p_c=='n':
                    cp_c=0
                    break
            print("Selección de archivos...")
            [im_path, path_save,path_images] = inicializar(2)
            print("Path: ", path_save)
            print("Iniciando REAL_DETECT...")
            real_detect_new(path_save,p_c,cp_c)
            real_detect_filter(path_save)
            #print("REAL_DETECT Listo")
            #print("Generarando KML fallas")
            generar_kml_fallas(path_save)
            generar_kml_fallas2(path_save)
            # generar_kml_imagenes(desp_este, desp_norte, desp_yaw, path_save, mode='detect')
            #print("KML fallas Listo")

        if (seleccion == "9"):
            [im_path, path_labels, path_obj, path_save, names] = inicializar()

        if (seleccion == "x"):
            break
