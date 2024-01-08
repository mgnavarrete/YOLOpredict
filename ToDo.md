# Reunion 08-01:

## Cosas Generales:
- Hacer pruebas con KML y no en plataforma.
- Unir paneles por fila para poder tener un solo poligono para analizar.
- Ver como usar el kml de los trackers y no el kml de los paneles, para ver si mejora.
- Mejorar modelo de Yolo para que no detecte la sombra.
- Guardar resultados de imágenes anteriores, para así comparar y mejorar los offset. (ver casos de primera imagen y cuando cambia de fila)

## Corrección de Yaw
- En general está bien funciona con imágenes probadas hasta ahora.
- Probar con otras imágenes para poder encontrar otros casos bordes.
- Ver como mejoraría si agregamos resultados de valores anteriores y posteriores.
- Implementar el uso del KML de trackers y no el de paneles.

## Corrección de  Altura
- No comparar por ancho de paneles, comparar por distancia entre filas:
    * Unir paneles por fila en un solo poligono.
    * Sacar centro de cada poligono.
    * Encontrar punto más cercano del kml de cada poligono
    * Comparar distancia entre poligonos imagen y el kml
- Guardar valores anteriores y ir comparando con los acutales, para mejorar el offset.

## Corrección de Este
- Guardar offset anteriores y ver los posteriores para poder mejorar el offset.
- Ver si existe alguna mejora al entrenar con más imágenes en modelo de yolo.
- Agrupar los panles en grupos por fila, y no analizar panel por panel.
- Ver caso de imágenes que no se estan moviendo como deberían.




