"""Descarga datasets desde Kaggle usando la API oficial.
Requiere tener ~/.kaggle/kaggle.json con tus credenciales.
"""
import os, subprocess, sys, shutil, zipfile

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

def download_insurance():
    # Many versions exist; the canonical insurance dataset is at "mirichoi0218/insurance"
    cmd = ["kaggle", "datasets", "download", "-d", "mirichoi0218/insurance", "-p", DATA_DIR, "-f", "insurance.csv"]
    subprocess.check_call(cmd)

def download_diabetes():
    # Pima Indians Diabetes Database
    cmd = ["kaggle", "datasets", "download", "-d", "uciml/pima-indians-diabetes-database", "-p", DATA_DIR, "-f", "diabetes.csv"]
    subprocess.check_call(cmd)

def main():
    try:
        download_insurance()
    except Exception as e:
        print("Fallo descarga insurance:", e)
    try:
        download_diabetes()
    except Exception as e:
        print("Fallo descarga diabetes:", e)
    # Unzip any zips that might have been downloaded
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".zip"):
            zpath = os.path.join(DATA_DIR, fname)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(DATA_DIR)
            os.remove(zpath)
    print("Listo. Archivos en data/.")

if __name__ == "__main__":
    main()
