import pandas as pd
import os
import glob
from pathlib import Path

ruta_carpeta = Path("../ML/")
csv_files = glob.glob(os.path.join(ruta_carpeta, "Kart_*.csv"))

kartData = []

for file in csv_files:
    kartData.append(pd.read_csv(file))
    
kartData = pd.concat(kartData, ignore_index = True)

output_file = os.path.join(ruta_carpeta, "concatenado.csv")
kartData.to_csv(output_file, index=False)
