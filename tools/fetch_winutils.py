"""Descarga winutils.exe + hadoop.dll (y hdfs.dll) a winutils/bin/ (gitignoreado).

Por qué es necesario (solo Windows):
  Spark 3.5 empaqueta Hadoop 3.3.4. Sin la librería nativa ``hadoop.dll``
  (cargada vía ``java.library.path``/PATH) el native ``NativeIO$Windows`` no
  se enlaza, y ``FileUtil.canRead`` (Hadoop 3.3.x) llama a
  ``NativeIO$Windows.access0`` **sin fallback en Java** — así que Spark no
  puede ni leer ni escribir en el filesystem local. Y ``winutils.exe`` es el
  ejecutable que Hadoop usa para ``setPermission`` (chmod) al crear
  directorios.

Fuente (la práctica, documentada):
  Apache **ya no publica** binarios Windows de Hadoop 3.x (ni zip en
  dlcdn ni en archive; solo fuentes y tarballs Linux). El binario estándar
  de la comunidad es la release ``hadoop-2.8.1`` de ``steveloughran/winutils``
  (Steve Loughran, comitente de Apache Hadoop), construida desde el commit
  oficial de ASF. Los símbolos JNI que el camino local de Hadoop 3.3.4
  realmente invoca (``initNative``, ``access0``,
  ``createDirectoryWithMode0``, ``createFileWithMode0``, ``setFilePointer``,
  ``getOwner``, ``mmap``/``munmap``) están declarados con idéntica firma
  tanto en 2.8.1 como en 3.3.4 — verificado contra
  ``hadoop-common/.../io/nativeio/NativeIO.java`` de ambas releases.

Resultado esperado (``HADOOP_HOME`` = <repo>/winutils):
  winutils/bin/winutils.exe
  winutils/bin/hadoop.dll
  winutils/bin/hdfs.dll

Uso:
  python tools/fetch_winutils.py

Uso en tests/benchmarks:
  - En Windows: ``conftest.py`` fija ``HADOOP_HOME``/``PATH`` a este
    directorio si existe (y respeta un ``HADOOP_HOME`` propio del usuario).
  - En Databricks: no hace falta (los workers Linux ya traen la librería);
    ver benchmarks/README.md.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile

# Release "Hadoop 2.8.1 binaries" de steveloughran/winutils
# (construida desde el commit oficial de ASF).
URL = (
    "https://github.com/steveloughran/winutils/releases/download/"
    "tag_2017-08-29-hadoop-2.8.1-native/hadoop-2.8.1.zip"
)
SHA256 = "ff099578a2bf618c3e3df95a76fdd2763d98f0e815560af6e7cfd2c92c04ed57"

# Miembros del zip (layout plano) -> destino bajo winutils/bin/.
WANTED = {
    "hadoop-2.8.1/winutils.exe": "winutils.exe",
    "hadoop-2.8.1/hadoop.dll": "hadoop.dll",
    "hadoop-2.8.1/hdfs.dll": "hdfs.dll",
}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "winutils", "bin")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(os.path.dirname(OUT), "hadoop-2.8.1.zip")
    if os.path.isfile(tmp) and os.path.getsize(tmp) > 1000:
        print(f"usando {tmp} ya descargado")
    else:
        print(f"descargando {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"descargado {os.path.getsize(tmp)} bytes")
    digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    if digest != SHA256:
        print(f"ERROR: sha256 inesperado {digest} (esperado {SHA256})")
        return 1

    with zipfile.ZipFile(tmp) as zf:
        for member, name in WANTED.items():
            out = os.path.join(OUT, name)
            with zf.open(member) as src, open(out, "wb") as dst:
                dst.write(src.read())
            print(f"extraído: {member} -> {out} ({os.path.getsize(out)} bytes)")
    os.remove(tmp)
    print(f"hecho. Fija HADOOP_HOME={os.path.dirname(OUT)} en Windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
