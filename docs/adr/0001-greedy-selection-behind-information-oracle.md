# Selección greedy detrás de un oráculo de información

**Status:** accepted

La selección greedy (etapa 3) existía duplicada: un bucle sobre la
`TableCache` de tablas conjuntas (histograma) y otro sobre arrays crudos con
cachés ad-hoc (KSG), cada uno con su propia convención de índices (posiciones
de candidata vs. índices globales de feature) — una mezcla que ya había
producido un bug real de selección incorrecta. Decidimos que el bucle greedy
(`selection.greedy_select`) consulta una única interfaz de cuatro métodos
(`InformationOracle`: `mi_all`, `mi_pair`, `cmi_single`, `cmi_set_all`) en
posiciones de candidata, con dos adaptadores: `HistogramOracle` (tablas
conjuntas; `cmi_set_all` ejecuta el pase CMIM distribuido por ronda) y
`KsgOracle` (kNN sobre el subsample en el driver).

## Considered Options

- **Extender `TableCache` a tablas de 4 vías** para que CMIM se sirviera del
  caché: rechazado — el coste de memoria crece con el producto de los códigos
  de `S_m` (inabordable), frente al coste de un pase `mapInPandas` extra por
  ronda CMIM, que es acotado por diseño (`cmim_m ≤ 2`).
- **Mantener los dos bucles y unificar solo la convención de índices**:
  rechazado — la duplicación del bucle greedy era la raíz del bug; unificar
  la convención no elimina la duplicación.

## Consequences

- El bucle greedy vive una sola vez y trabaja solo en posiciones de
  candidata: la clase de bugs de mezcla de índices desaparece por
  construcción.
- Los criterios rápidos (mrmr/mim/jmi/jmim) son aritmética pura sobre el
  oráculo; `cmim` es el único criterio que toca Spark (pase distribuido por
  ronda), como consecuencia directa de la interfaz.
- El subsample se calcula una sola vez (etapa 2) y se comparte entre el
  caché de tablas y el pase CMIM, en lugar de dos veces.
- El ranking cubre la población completa (N features por MI univariante
  descendente): el modo histograma reutiliza la MI de screening sin coste
  adicional y el modo KSG usa la MI de kNN sobre el subsample.
