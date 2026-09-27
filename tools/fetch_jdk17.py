"""Descarga Temurin JDK 17 (Windows x64) a ``jdk17/`` en la raíz del proyecto.

El conftest de los tests fija ``JAVA_HOME`` a ``jdk17/`` si existe (Spark 3.5
rompe ``mapInPandas`` en JDK 21 por la Foreign Memory API de Arrow 12).
"""

import os
import tempfile
import urllib.request

URL = (
    "https://github.com/adoptium/temurin17-binaries/releases/download/"
    "jdk-17.0.20.1%2B1/OpenJDK17U-jdk_x64_windows_hotspot_17.0.20.1_1.zip"
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "jdk17")


def main() -> None:
    import zipfile

    tmp = os.path.join(tempfile.gettempdir(), "jdk17.zip")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"descargado {os.path.getsize(tmp)} bytes")

    if os.path.isdir(DEST):
        import shutil

        shutil.rmtree(DEST)
    os.makedirs(DEST)
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(DEST)
    os.remove(tmp)

    inner = next(d for d in os.listdir(DEST) if os.path.isdir(os.path.join(DEST, d)))
    for name in os.listdir(os.path.join(DEST, inner)):
        os.replace(os.path.join(DEST, inner, name), os.path.join(DEST, name))
    os.rmdir(os.path.join(DEST, inner))
    print("extraído a", DEST)


if __name__ == "__main__":
    main()
