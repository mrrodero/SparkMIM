"""Conftest compartido de SparkMIM (configuración del entorno de tests).

1. En Windows, el alias de ejecución de aplicaciones de WindowsApps
   (`%LOCALAPPDATA%\\Microsoft\\WindowsApps\\python.exe`) intercepta las
   llamadas a `python`/`python3` y muestra "no se encontró Python". Spark
   usa ese nombre por defecto para lanzar sus workers de Python (ver
   `pyspark.context.SparkContext.pythonExec`), así que fijamos
   `PYSPARK_PYTHON` al intérprete activo ANTES de crear cualquier sesión.

2. Spark 3.5 empaqueta Arrow 12, que asigna buffers directos mediante
   reflexión sobre `sun.misc.Unsafe.allocateDirect` /
   `DirectByteBuffer(long,int)`. En los JDK 21 recientes (con la Foreign
   Memory API) esas APIs desaparecieron y `mapInPandas` falla. Por eso los
   tests usan el JDK 17 descargado en la raíz del proyecto (`jdk17/`), que
   es la versión soportada oficialmente por Spark 3.5.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

_JDK17 = Path(__file__).resolve().parent / "jdk17"
if _JDK17.exists():
    os.environ["JAVA_HOME"] = str(_JDK17)
    os.environ["PATH"] = str(_JDK17 / "bin") + os.pathsep + os.environ.get("PATH", "")
