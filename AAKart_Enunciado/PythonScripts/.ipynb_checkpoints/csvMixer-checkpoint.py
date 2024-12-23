import pandas as pd
import os
import glob

ruta_carpeta = Path("../ML/")

if not ruta_carpeta.exists():
    print(f"La carpeta especificada '{ruta_carpeta}' no existe. Ruta absoluta: {ruta_carpeta.resolve()}")
else:
    print(f"Carpeta encontrada: {ruta_carpeta.resolve()}")
    
# Buscar todos los archivos que empiecen por "Kart_" y tengan un número
csv_files = glob.glob(os.path.join(ruta_carpeta, "Kart_*.csv"))

# Verificar si se encontraron archivos
if not csv_files:
    print("No se encontraron archivos que cumplan con el patrón 'Kart_*.csv'.")
else:
    print(f"Archivos encontrados: {csv_files}")

# Lista donde se almacenarán los DataFrames
dataframes = []

# Leer cada archivo, omitir la primera fila, y añadirlo a la lista
for file in csv_files:
    try:
        df = pd.read_csv(file, skiprows=1)  # Omitir la primera fila
        dataframes.append(df)
        print(f"Archivo '{file}' leído con éxito.")
    except Exception as e:
        print(f"Error leyendo el archivo '{file}': {e}")

# Concatenar todos los DataFrames
if dataframes:
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Guardar el archivo concatenado en un nuevo archivo CSV
    output_file = os.path.join(ruta_carpeta, "concatenado.csv")
    concatenated_df.to_csv(output_file, index=False)

    print(f"Archivos concatenados con éxito. Guardado en '{output_file}'.")
else:
    print("No se encontraron archivos válidos para concatenar.")
