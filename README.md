
# Ajuste Automático


## 1. correctE.py
### Funcionalidad General
`correctE.py` es un script de Python diseñado para el procesamiento de coordenadas Este (E) en sistemas geográficos, incluyendo correcciones y validaciones. Facilita la corrección automática de errores comunes en las coordenadas, aplicando ajustes para mejorar la precisión de los datos geográficos.

### Funciones del Script

1.1 **estan_en_linea(long1, long2, umbral=0.5)**:
   - **Descripción**: Determina si dos longitudes están alineadas dentro de un umbral de tolerancia especificado.
   - **Parámetros**:
     - `long1`: Longitud del primer punto.
     - `long2`: Longitud del segundo punto.
     - `umbral`: Valor de tolerancia para determinar la alineación.
   - **Retorno**: Booleano que indica si las longitudes están alineadas o no.

1.2 **dms_a_decimal(grados, minutos, segundos)**:
   - **Descripción**: Convierte coordenadas de grados, minutos y segundos a formato decimal.
   - **Parámetros**:
     - `grados`: Grados de la coordenada.
     - `minutos`: Minutos de la coordenada.
     - `segundos`: Segundos de la coordenada.
   - **Retorno**: Valor de la coordenada en formato decimal.

1.3 **correctE(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)**:
   - **Descripción**: Realiza la corrección de las coordenadas Este (E) para un conjunto de imágenes georreferenciadas. Procesa metadatos y utiliza un modelo para identificar y corregir errores.
   - **Parámetros**:
     - `folder_path`: Ruta a la carpeta con imágenes originales.
     - `img_names`: Nombres de las imágenes a procesar.
     - `geonp_path`: Ruta a archivos numpy georreferenciados.
     - `metadata_path`: Ruta a archivos JSON de metadatos.
     - `metadatanew_path`: Ruta a archivos JSON con metadatos corregidos.
     - `df`: DataFrame con datos georreferenciados.
     - `transformer`: Objeto para transformación de coordenadas.
     - `model`: Modelo de aprendizaje profundo para detección en imágenes.
   - **Retorno**: No hay, implica la actualización de metadatos ygeneración de nuevos archivos con coordenadas corregidas.


## 2. correctH.py
### Descripción General
`correctH.py` es un script de Python diseñado para el procesamiento de datos geográficos y de imágenes. Realiza operaciones como el cálculo de distancias geográficas, el análisis geométrico de puntos y la manipulación de imágenes.

### Funciones del Script

2.1 **haversine_distance(lat1, lon1, lat2, lon2)**:
   - **Descripción**: Calcula la distancia Haversine entre dos puntos geográficos.
   - **Parámetros**:
     - `lat1`, `lon1`: Latitud y longitud del primer punto.
     - `lat2`, `lon2`: Latitud y longitud del segundo punto.
   - **Retorno**: Distancia Haversine entre los dos puntos.

2.2 **calcular_centroide(puntos)**:
   - **Descripción**: Calcula el centroide de un conjunto de puntos.
   - **Parámetros**:
     - `puntos`: Lista de puntos (x, y).
   - **Retorno**: Centroide (x, y) del conjunto de puntos.

2.3 **calcular_area_poligono(puntos)**:
   - **Descripción**: Calcula el área de un polígono definido por un conjunto de puntos.
   - **Parámetros**:
     - `puntos`: Lista de puntos (x, y) que definen el polígono.
   - **Retorno**: Área del polígono.

2.4 **correctH(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model)**:
   - **Descripción**: [Descripción de la función correctH].
   - **Parámetros**:
     - `folder_path`: Ruta a la carpeta con imágenes originales.
     - `img_names`: Nombres de las imágenes a procesar.
     - `geonp_path`: Ruta a archivos numpy georreferenciados.
     - `metadata_path`: Ruta a archivos JSON de metadatos.
     - `metadatanew_path`: Ruta a archivos JSON con offset_yaw modificado.
     - `df`: DataFrame con datos georreferenciados.
     - `transformer`: Objeto para transformación de coordenadas.
     - `model`: Modelo de aprendizaje profundo para detección en imágenes.
   - **Retorno**: No hay, implica la actualización de metadatos ygeneración de nuevos archivos con coordenadas corregidas.


## 3. correctYaw.py
### Descripción General
`correctYaw.py` es un script de Python diseñado para procesar imágenes georreferenciadas. Este script realiza operaciones como calcular distancias entre puntos geográficos, ordenar puntos y corregir la orientación (yaw) de las imágenes basándose en ciertos parámetros y metadatos.

### Funciones del Script

3.1 **haversine_distance(lat1, lon1, lat2, lon2)**:
   - **Descripción**: Calcula la distancia Haversine entre dos puntos geográficos.
   - **Parámetros**:
     - `lat1`, `lon1`: Latitud y longitud del primer punto.
     - `lat2`, `lon2`: Latitud y longitud del segundo punto.
   - **Retorno**: Distancia Haversine entre los dos puntos.

3.2 **ordenar_puntos(puntos)**:
   - **Descripción**: Ordena un conjunto de puntos en un orden específico.
   - **Parámetros**:
     - `puntos`: Lista de puntos para ordenar.
   - **Retorno**: Lista de puntos ordenados.

3.3 **correctYaw(folder_path, img_names, geonp_path, metadata_path, metadatanew_path, df, transformer, model, yawKML, ancho, list_images)**:
   - **Descripción**: Corrige la orientación (yaw) de imágenes georreferenciadas.
   - **Parámetros**:
     - `folder_path`: Ruta a la carpeta con imágenes originales.
     - `img_names`: Nombres de las imágenes a procesar.
     - `geonp_path`: Ruta a archivos numpy georreferenciados.
     - `metadata_path`: Ruta a archivos JSON de metadatos.
     - `metadatanew_path`: Ruta a archivos JSON con offset_yaw modificado.
     - `df`: DataFrame con datos georreferenciados.
     - `transformer`: Objeto para transformación de coordenadas.
     - `model`: Modelo de aprendizaje profundo para detección en imágenes.
     - `yawKML`: Valor medio del ángulo yaw obtenido de KML.
     - `ancho`: Ancho promedio de ciertos elementos en las imágenes.
     - `list_images`: Lista opcional de imágenes a procesar.
   - **Retorno**: No hay, implica la actualización de metadatos ygeneración de nuevos archivos con coordenadas corregidas.



## Ejecucion General

### Ajuste Automático con `corrector.py`

Este documento proporciona una guía detallada para ejecutar el programa de ajuste automático utilizando el archivo `corrector.py`. Este script es esencial para el procesamiento y corrección de datos geoespaciales, especialmente diseñado para ajustar automáticamente las mediciones basadas en la configuración específica de cada planta solar.

#### Requisitos Previos

Antes de ejecutar `corrector.py`, asegúrate de tener instalado Python 3.6 o superior y todas las dependencias requeridas. Las dependencias incluyen librerías como `opencv-python` (cv2), `numpy`, `pandas`, `pyproj`, y `ultralytics` (YOLO). Puedes instalar estas dependencias utilizando `pip`:

```
pip install -r requirements.txt
```

#### Ejecución del Programa

Para ejecutar `corrector.py` y comenzar el proceso de ajuste, sigue estos pasos:
1. **Abrir Carpeta con Codigos**: Tanto en la nave como el predator, la carpeta YOLOPredict esta en el escritorio y es donde se encuentran todos los scripts.

1. **Abrir la Terminal o Línea de Comandos**: Abrir la terminal de la carpeta para poder ejecutar el programa.

2. **Ejecutar el Script**: Utiliza el siguiente comando para iniciar el script. 

    - Para correrlo en el predator usar: `python corrector.py`

    - Para correrlo en la nave parado en la carpeta del repo correr: `venv\Script\activate` y luego correr: `python corrector.py`



3.  Las opciones disponibles están diseñadas para adaptarse a las especificaciones de diferentes plantas solares. Elige la opción que mejor se ajuste a tu necesidad basada en la planta solar que estás analizando.
Hasta el momento las plantas solares que estan disponibles son:
    - Finis Terrae (FIT)
    - Finis Terrae Extensión (FIX)
    - Campos del Sol (CDS)
    - Lalakama (LLK)
    - Sol de Lila (SDL)

#### Funcionalidad de `corrector.py`

`corrector.py` sirve como el núcleo del programa de ajuste automático, integrando múltiples scripts de corrección y herramientas analíticas. Sus principales funciones incluyen:

- **Integración de Scripts de Corrección**: Coordina la ejecución de scripts específicos de corrección (como `correctH`, `correctE`, y `correctYaw`) en función de la opción seleccionada para la planta solar.
- **Procesamiento de Imágenes y Datos Geoespaciales**: Utiliza `opencv-python` y `numpy` para manipular imágenes y datos geoespaciales, preparándolos para el análisis y corrección.
- **Transformación de Coordenadas**: Aplica transformaciones de coordenadas y ajustes utilizando `pyproj`, adecuados para las especificaciones geográficas de la planta solar.
- **Generación de Resultados**: Produce matrices geoespaciales y archivos KML (`saveGeoM`, `saveKML`) para visualización y análisis posterior.


## Uso de Git

### Colaboración con Git en el Proyecto

Git es una herramienta esencial para la colaboración y gestión de versiones en proyectos de software. Para contribuir eficazmente al repositorio `YOLOpredict`, sigue los siguientes pasos y buenas prácticas:

#### Configuración Inicial

1. **Clonar el Repositorio**: Para empezar, clona el repositorio a tu máquina local usando:
   ```
   git clone https://github.com/mgnavarrete/YOLOpredict
   ```
   En la Nave y el Predator las carpetas ya esta instaladas en el escritorio, hacer un `git pull` para tener la ultima version del repositorio.

#### Trabajando con Ramas

2. **Crear una Nueva Rama**: Antes de empezar a trabajar en nuevas características o correcciones, crea una rama para aislar tus cambios:
   ```
   git checkout -b nombre_de_la_rama
   ```
   Reemplaza `nombre_de_la_rama` con un nombre descriptivo basado en la característica o corrección que estés implementando.

3. **Cambio entre Ramas**: Para cambiar de una rama a otra, utiliza:
   ```
   git checkout nombre_de_la_rama
   ```

#### Añadir y Confirmar Cambios

4. **Añadir Cambios**: Una vez que hayas hecho cambios en tu código, añádelos al área de preparación (staging area) con:
   ```
   git add .
   ```
   o
   ```
   git add nombre_del_archivo
   ```
   para añadir archivos específicos.

5. **Confirmar Cambios**: Para confirmar los cambios añadidos, utiliza:
   ```
   git commit -m "Mensaje descriptivo del cambio"
   ```
   El mensaje del commit debe ser claro y conciso, describiendo qué cambios has realizado y por qué.

#### Actualizar y Sincronizar

6. **Actualizar tu Rama Local**: Regularmente, actualiza tu rama local con los cambios del repositorio remoto para mantener tu trabajo sincronizado:
   ```
   git pull origin nombre_de_la_rama
   ```

#### Subir Cambios

7. **Subir Cambios al Repositorio Remoto**: Una vez que tus cambios estén listos y confirmados, súbelos al repositorio remoto para que otros puedan acceder a ellos:
   ```
   git push origin nombre_de_la_rama
   ```

#### Colaboración y Revisiones

8. **Solicitar la Integración de Cambios (Pull Request)**: Utiliza la interfaz de GitHub para crear una solicitud de integración (Pull Request) cuando estés listo para que tus cambios sean revisados y eventualmente integrados al proyecto principal.

9. **Revisión de Código**: Participa en la revisión de código de las Pull Requests de tus compañeros de equipo, proporcionando comentarios constructivos y aprobaciones cuando sea apropiado.
