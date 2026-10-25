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

3. winutils + hadoop.dll: sin la librería nativa de Hadoop, Spark no puede
   leer NI escribir en el filesystem local (en Hadoop 3.3.x,
   ``FileUtil.canRead`` llama al native ``NativeIO$Windows.access0`` sin
   fallback en Java). Ver tools/fetch_winutils.py.

4. En Windows, el launcher de PySpark escribe a ``%TEMP%``
   (``spark-class.cmd`` redirige la salida del launcher ahí). Si ese
   directorio no es escribible (AV/restricciones), el arranque de la sesión
   se queda a la espera indefinidamente. Por eso apuntamos TEMP/TMP a un
   directorio local del repo (gitignoreado): determinista en cualquier
   máquina.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

_JDK17 = Path(__file__).resolve().parent / "jdk17"
if _JDK17.exists():
    os.environ["JAVA_HOME"] = str(_JDK17)
    os.environ["PATH"] = str(_JDK17 / "bin") + os.pathsep + os.environ.get("PATH", "")

# winutils + hadoop.dll (solo Windows): ver tools/fetch_winutils.py. Se
# respeta un HADOOP_HOME propio del usuario.
_WINUTILS = Path(__file__).resolve().parent / "winutils"
if (_WINUTILS / "bin").exists() and not os.environ.get("HADOOP_HOME"):
    os.environ["HADOOP_HOME"] = str(_WINUTILS)
    os.environ["PATH"] = str(_WINUTILS / "bin") + os.pathsep + os.environ.get("PATH", "")

# TEMP/TMP local (solo Windows): el launcher de PySpark (spark-class.cmd)
# redirige la salida del launcher a %TEMP%; si no es escribible, la sesión
# se queda colgada al arrancar. Un directorio del repo es siempre válido.
if os.name == "nt":
    _SPARK_TMP = Path(__file__).resolve().parent / ".tmp_spark"
    _SPARK_TMP.mkdir(exist_ok=True)
    os.environ["TEMP"] = str(_SPARK_TMP)
    os.environ["TMP"] = str(_SPARK_TMP)
