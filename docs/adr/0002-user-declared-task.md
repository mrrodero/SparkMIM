# Task declarada por el usuario con fallback auto

**Status:** accepted

SparkMIM infirió el tipo de problema desde el dtype del Target en el momento del reporte (`detect_task`), lo cual discrepaba con la etapa 0: un Target entero con 11–20 valores distintos se binaba como regresión para la estimación pero se puntuaba como clasificación, y la resolución no sobrevivía en el `SelectorModel` (solo el nombre del Target). Decidimos que la Task se declara en `SelectorConfig` (`classifier_binary`, `classifier_multiclass` o `continuous`, por defecto `auto`), se resuelve una sola vez en la etapa 0 validando contra los datos, y viaja en el `SelectorModel`; `report` mide el Target crudo con la Task llevada.

## Considered Options

- **Solo inferencia** (mantener `detect_task` como fuente única): rechazado — los datos solo muestran cómo se ve el Target; el usuario sabe cuál es el problema (un Target entero con 30 valores distintos que son en realidad 30 clases no es inferible desde los datos).
- **Ganar las semánticas de la etapa 0** (Target numérico → siempre regresión): rechazado — convierte el artefacto de binning en la definición del problema (una escala Likert de 5 niveles guardada como float se volvería regresión).
- **Re-codificar el Target a los bins de la etapa 0 en `report`**: rechazado — la métrica mediría un proxy binado, no el Target que le importa al usuario.

## Consequences

- `detect_task` se elimina de la superficie pública; su regla de inferencia sobrevive como la rama `auto` de la regla de resolución compartida en `schema.py`.
- Las funciones públicas de `evaluate` (`report`, `train_and_evaluate`, `efficiency_curve`) aceptan los strings legacy de 2 valores (`classification`/`regression`) mapeados a `classifier_multiclass`/`continuous` — conservador en comportamiento, porque el AUC macro sobre 2 clases es igual al AUC binario.
- El modo KSG gana una etapa 0 mínima: validación de dtype por Task + resolución de la Task (misma regla compartida), y la Task viaja en el modelo.
