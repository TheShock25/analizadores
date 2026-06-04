# Analizadores de emociones para letras de canciones

Repositorio de respaldo con prototipos independientes para analizar emociones en letras de canciones usando modelos Transformer y reglas contextuales.

Estos archivos fueron desarrollados como pruebas aisladas antes de integrar un analizador definitivo al proyecto principal de tesis.

## Objetivo

Comparar diferentes formas de clasificar emociones en letras de canciones sin modificar el `main.py` ni el pipeline principal.

El enfoque principal usa modelos Transformer preentrenados y capas de ajuste contextual para interpretar letras complejas, metáforas, contradicciones, coros, estrofas, jerga y funciones narrativas.

## Archivos incluidos

### `ekman_transformer_prueba.py`

Clasificador básico basado en emociones de Ekman.

Clasifica canciones en emociones generales como alegría, tristeza, miedo, ira, asco y sorpresa.

Uso recomendado: prueba inicial o comparación simple.

### `goemotions_transformer_prueba.py`

Clasificador general basado en una taxonomía tipo GoEmotions adaptada a letras musicales.

Analiza la canción completa y genera categorías emocionales más detalladas que Ekman.

Uso recomendado: comparación de clasificación global por canción.

### `goemotions_frases_24h_prueba.py`

Analizador por frases o segmentos.

Divide letras en fragmentos pequeños y clasifica cada parte para obtener un resumen emocional por canción.

Uso recomendado: observar cómo cambian las emociones dentro de una letra.

### `goemotions_jerarquico_prueba.py`

Analizador jerárquico frase -> estrofa -> canción.

Es la prueba más completa. Separa:

- emoción principal;
- perfil emocional;
- estado de confianza;
- función narrativa;
- aporte narrativo global;
- resumen emocional;
- resumen narrativo.

También maneja casos como:

- canciones tristes o de desamor;
- amor correspondido;
- nostalgia;
- contradicción emocional;
- placer dañino;
- no compromiso;
- alarde;
- jerga urbana;
- enumeraciones;
- advertencia afectiva;
- autoconciencia problemática;
- baja densidad emocional.

Uso recomendado: candidato principal para futura integración al sistema.

## Entrada de prueba

El archivo `prueba_cancion_24h.txt` puede usarse como entrada manual para los analizadores que aceptan `--archivo-txt`.

Ejemplo:

```bash
python goemotions_jerarquico_prueba.py --archivo-txt prueba_cancion_24h.txt --frases-por-estrofa 4

Crear entorno virtual:
python -m venv venv

Activar entorno en Windows:
venv\Scripts\activate

Instalar dependencias:
pip install -r requirements.txt

Ejecución:

Clasificador Ekman
python ekman_transformer_prueba.py

Clasificador GoEmotions global
python goemotions_transformer_prueba.py

Clasificador por frases o últimas 24 horas
python goemotions_frases_24h_prueba.py

Con un TXT manual:
python goemotions_frases_24h_prueba.py --archivo-txt prueba_cancion_24h.txt

Clasificador jerárquico

python goemotions_jerarquico_prueba.py --archivo-txt prueba_cancion_24h.txt --frases-por-estrofa 4