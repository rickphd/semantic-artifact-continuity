# Auditoría de admisibilidad de la sensibilidad F1+F2+F3

Estado: passed. Configuraciones planificadas: 125.

Clasificación: {'admissible_completed': 97, 'outside_current_loader_contract': 28}. Motivos de rechazo: {'negative_empty': 22, 'both_polarities_empty': 6}.

La restricción que exige diccionarios positivo y negativo no vacíos está en OntologyEnricher._load_sentiment_lexicons. La operación posterior suma puntuaciones mediante consultas de pertenencia a esos diccionarios; no hay una necesidad matemática de que ambos tengan entradas. Por tanto, los rechazos se clasifican respecto al contrato operativo congelado, no como imposibilidad científica universal.

Se conserva el contrato durante esta campaña. Permitir léxicos de una sola polaridad o vacíos definiría una variante de implementación adicional, con sus propias pruebas y ejecución; no se ha introducido ese cambio.

Para cada punto se reconstruye el léxico con el texto de entrenamiento y los parámetros numéricos originales. Se comparan exactamente entradas, estadísticas, parámetros, IDs, hashes de origen y recurso VADER. Cada exclusión requiere reproducir el ValueError exacto, concordancia con el error registrado y ausencia de artefactos downstream. Cada admisible requiere los artefactos completos, hashes de dataset/léxico y SHACL conforme con cero resultados.

## Cobertura de la rejilla por ventana

| Ventana | Admisibles completas | Fuera del contrato | Pendientes o errores |
|---|---:|---:|---:|
| 2 | 14 | 11 | 0 |
| 3 | 15 | 10 | 0 |
| 5 | 20 | 5 | 0 |
| 7 | 24 | 1 | 0 |
| 10 | 24 | 1 | 0 |

Las tasas de admisibilidad se expresan sobre 125. Las estadísticas de cobertura, selección y conformidad se calculan exclusivamente sobre las configuraciones admisibles, cuyo denominador debe acompañar cada resultado. Las configuraciones rechazadas no son ceros de cobertura ni fallos SHACL: no alcanzaron esas etapas.

El texto selection_rule de los manifiestos dice at least three times incluso para otras frecuencias: es una etiqueta fija del generador. La auditoría utiliza y verifica minimum_frequency numérico y reconstruye los términos; no interpreta esa etiqueta como el parámetro ejecutado. Los manifiestos originales se conservan.

Archivos: configuration_audit.csv conserva los 125 puntos, sus parámetros, conteos, clasificación, error y hash. admissibility_by_window_frequency.csv muestra dónde se pierden las polaridades. audit.json contiene la evidencia verificable. No se han reentrenado modelos ni modificado el .tex, freeze, piloto o release.
