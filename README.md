ventas_original.csv =  Contiene el conjunto de datos originales exportados desde el sistema de ventas de la tienda departamental. Incluye campos como fecha, producto, cantidad, precio y cliente. Este archivo presenta inconsistencias como valores nulos y formatos no estandarizados.

ventas_limpio.csv = Es el resultado del proceso ETL, donde los datos han sido depurados, estandarizados y transformados para su posterior análisis. Se agregan columnas derivadas, como el total de venta (cantidad × precio), y se corrigen los formatos de fecha.



Extracción:

Los datos fueron obtenidos del archivo ventas_original.csv,
 el cual provenía del registro de ventas diarias de la tienda departamental.
Los campos principales fueron:

Fecha de venta

Nombre del producto

Cantidad vendida

Precio unitario

Cliente




Transformación:

Durante esta etapa se aplicaron las siguientes acciones:

Eliminación de filas con valores nulos o duplicados.

Estandarización de nombres de columnas (por ejemplo: FechaVenta → fecha).

Conversión del formato de fecha de texto a formato ISO (AAAA-MM-DD).

Creación del campo calculado total_venta = cantidad × precio.

Normalización de nombres de clientes y productos (mayúsculas/minúsculas consistentes).



Carga:

El resultado del proceso fue almacenado en el archivo ventas_limpio.csv,
 que contiene los datos estructurados, 
corregidos y listos para integrarse a un Data Warehouse o analizarse con herramientas de inteligencia de negocios (BI).