#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goemotions_frases_24h_prueba.py
================================
Experimento aislado para analizar letras por frases/estrofas.

No modifica main.py, moodtracker.db, matrices ni canciones_*.txt.

Metodologia:
  1. Toma canciones con letra que pasaron el filtro de matriz (usar_en_matriz=1).
  2. Usa como ancla la fecha mas reciente de la base y conserva solo las
     ultimas 24 horas.
  3. Deduplica por spotify_track_id.
  4. Descarta letras evidentemente ruidosas (listas de albums, fechas, etc.).
  5. Divide cada letra en segmentos pequenos: frases/versos agrupados.
  6. Clasifica cada segmento con la taxonomia GoEmotions adaptada.
  7. Resume por cancion que emociones dominan y cuales acompanan.

Salidas:
    prueba_transformers_frases_24h/
        frases_clasificadas.csv
        resumen_canciones.csv
        letras_usadas_meta.csv
        letras_descartadas_ruido.csv
        categorias/frase_XX_categoria.txt
        por_cancion/NN_artista_cancion.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from goemotions_transformer_prueba import (
    BASE_DIR,
    DB_PATH,
    GOEMOTIONS_CANCIONES,
    MODELO_ZERO_SHOT,
    calcular_score_lexico,
    cargar_zero_shot,
    limpiar_letra_cruda,
    limpiar_nombre_archivo,
    leer_descartadas,
    reparar_mojibake,
)


OUT_DIR = os.path.join(BASE_DIR, "prueba_transformers_frases_24h")
OUT_DIR_TXT = os.path.join(BASE_DIR, "prueba_transformers_frases_txt")

CATEGORIAS_EXTRA_MUSICA = [
    {
        "id": 28,
        "clave": "melancolia_nostalgia",
        "label": "melancolia, nostalgia o recuerdo distante",
        "prompt": "La frase expresa melancolia, nostalgia, recuerdo, ausencia o una emocion del pasado.",
        "grupo": "especial_musica",
        "descripcion": "memoria afectiva, distancia, pasado o nostalgia contemplativa",
    },
    {
        "id": 29,
        "clave": "redencion_renacer",
        "label": "redencion, renacer o esperanza despues del dolor",
        "prompt": "La frase expresa redencion, renacer, aprendizaje, nuevo comienzo o esperanza despues del dolor.",
        "grupo": "especial_musica",
        "descripcion": "transformacion positiva despues de dolor, despedida o crisis",
    },
    {
        "id": 30,
        "clave": "aceptacion_desapego",
        "label": "aceptacion, desapego o despedida madura",
        "prompt": "La frase expresa aceptacion, desapego, despedida madura o soltar a alguien.",
        "grupo": "especial_musica",
        "descripcion": "aceptar una separacion, soltar el rencor o comprender una despedida",
    },
]

CATEGORIAS_FRASES = GOEMOTIONS_CANCIONES + CATEGORIAS_EXTRA_MUSICA

# Lexico musical manual desactivado como clasificador.
# El analisis por frases debe basarse en transformer + contexto jerarquico,
# no en listas grandes de canciones o frases especificas.
LEXICOS_EXTRA_MUSICA = {}
LEXICO_FRASE_AUXILIAR_PESO = 0.08
FILLER_RE = re.compile(
    r"^\s*(uh+|oh+|ah+|eh+|mm+|hmm+|na+|la+|woh+|yeah+|hey+|"
    r"uh[-\suh]*|oh[-\soh]*|ah[-\sah]*)\s*$",
    re.IGNORECASE,
)


def parse_fecha(fecha: str) -> datetime | None:
    if not fecha:
        return None
    try:
        if fecha.endswith("Z"):
            fecha = fecha[:-1] + "+00:00"
        dt = datetime.fromisoformat(fecha)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def obtener_ancla_db() -> datetime:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT MAX(fecha_reproduccion) FROM canciones")
    row = cur.fetchone()
    conn.close()
    ancla = parse_fecha(row[0] if row else "")
    if ancla is None:
        raise RuntimeError("No se pudo obtener fecha maxima de moodtracker.db")
    return ancla


def es_letra_ruidosa(letra: str) -> tuple[bool, str]:
    texto = limpiar_letra_cruda(letra)
    largo = len(texto)
    fechas = len(re.findall(r"\b\d{1,2}/\d{1,2}\b", texto))
    fracciones = len(re.findall(r"\b\d{1,3}/\d{1,3}\b", texto))
    guiones_album = texto.count(" - ")
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    lineas_catalogo = sum(
        1
        for linea in lineas
        if re.search(r"\b\d{1,3}/\d{1,3}\b", linea) and " - " in linea
    )

    if largo > 15000 and (fracciones > 20 or guiones_album > 80):
        return True, "catalogo_musical_extenso"
    if fechas > 15 and fracciones > 20:
        return True, "lista_fechas_albums"
    if lineas and lineas_catalogo / len(lineas) > 0.35:
        return True, "lineas_catalogo"
    if len(lineas) <= 1 and largo > 7000 and guiones_album > 40:
        return True, "texto_largo_no_lirico"
    return False, ""


def cargar_canciones_ultimas_horas(horas: int, limite: int | None = None) -> tuple[list[dict], list[dict], datetime]:
    ancla = obtener_ancla_db()
    inicio = ancla - timedelta(hours=horas)
    descartadas = leer_descartadas()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT spotify_track_id, nombre, artista, album, fecha_reproduccion,
               letra, relevancia_letra, patron_escucha, motivo_matriz, score_matriz
        FROM canciones
        WHERE letra IS NOT NULL
          AND letra != ''
          AND COALESCE(usar_en_matriz, 1) = 1
        ORDER BY fecha_reproduccion DESC
    """)

    canciones = []
    ruido = []
    vistos = set()

    for row in cur.fetchall():
        fecha = parse_fecha(row["fecha_reproduccion"])
        if fecha is None or fecha < inicio or fecha > ancla:
            continue

        track_id = row["spotify_track_id"]
        if not track_id or track_id in vistos or track_id in descartadas:
            continue
        vistos.add(track_id)

        letra = limpiar_letra_cruda(row["letra"])
        if len(letra) < 20:
            continue

        es_ruido, motivo_ruido = es_letra_ruidosa(letra)
        base = {
            "indice": len(canciones) + 1,
            "spotify_track_id": track_id,
            "nombre": reparar_mojibake(row["nombre"] or ""),
            "artista": reparar_mojibake(row["artista"] or ""),
            "album": reparar_mojibake(row["album"] or ""),
            "fecha_reproduccion": row["fecha_reproduccion"] or "",
            "relevancia_letra": row["relevancia_letra"] or "",
            "patron_escucha": row["patron_escucha"] or "",
            "motivo_matriz": row["motivo_matriz"] or "",
            "score_matriz": row["score_matriz"] if row["score_matriz"] is not None else "",
            "letra": letra,
        }

        if es_ruido:
            ruido.append({**base, "motivo_ruido": motivo_ruido, "longitud": len(letra)})
            continue

        base["indice"] = len(canciones) + 1
        canciones.append(base)
        if limite and len(canciones) >= limite:
            break

    conn.close()
    return canciones, ruido, ancla


def preparar_salida(out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "categorias"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "por_cancion"), exist_ok=True)


def limpiar_linea(linea: str) -> str:
    linea = re.sub(r"\s+", " ", linea).strip()
    linea = re.sub(r"^\W+|\W+$", "", linea).strip()
    return linea


def normalizar_simple(texto: str) -> str:
    texto = reparar_mojibake(texto or "").lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tiene_marcador(texto: str, marcadores: Iterable[str]) -> bool:
    return any(marcador in texto for marcador in marcadores)


def contar_marcadores(texto: str, marcadores: Iterable[str]) -> int:
    return sum(1 for marcador in marcadores if marcador in texto)


def primera_posicion(texto: str, marcadores: Iterable[str]) -> int | None:
    posiciones = [texto.find(marcador) for marcador in marcadores if marcador in texto]
    posiciones = [pos for pos in posiciones if pos >= 0]
    return min(posiciones) if posiciones else None


MARCADORES_AFECTO = [
    "amor", "amar", "amo", "te quiero", "querer", "carino", "corazon",
    "beso", "besar", "labios", "caricias", "ternura", "contigo",
    "enamore", "enamorado", "enamorando", "mi vida", "mi cielo",
    "cuerpo", "abrazos", "abrazar", "manos", "entrega", "pasion",
    "love", "heart", "kiss", "darling", "baby",
]

MARCADORES_RELACION_DANADA = [
    "infiel", "infidel", "traicion", "traicionar", "engano", "enganar",
    "mentira", "mentiste", "olvidaste", "olvido", "amor ajeno",
    "otro amor", "otra amor", "ya no te amo", "ya no siento amor",
    "ya no quiero nada", "no quiero nada contigo", "no seremos",
    "ni amigos", "enemigo", "rencor", "reproche", "ruptura",
    "terminar", "terminamos", "se acabo", "se termino", "adios",
    "alejare", "me ire", "dejaste", "me dejas", "abandono",
    "rechazo", "fallaste", "dano", "danaste", "maltrato",
    "ausente", "distante", "distinta", "ya no sientes", "no sientes",
    "no te sale", "finges", "fingir", "mientele", "miente",
    "mentir", "otro hombre", "otra persona", "me ha robado",
    "robado tu calor", "robado tu pasion", "me voy", "te vas",
    "no eres mia", "no eres mio", "no es mia", "no es mio",
    "irreparable", "inestable", "me alejo", "no eres para mi",
    "no me pertenece", "no me perteneces",
    "forbidden", "betray", "cheat", "lie", "goodbye", "leave me",
]

MARCADORES_CIERRE_DECISION = [
    "me alejare", "me voy", "me ire", "alejarme", "alejarte",
    "ya no quiero", "no quiero nada", "no seremos", "terminar",
    "terminamos", "adios", "decir adios", "dar la vuelta",
    "dar la media vuelta", "no pienso reprochar", "soltar",
    "dejar atras", "move on", "let go", "goodbye",
]

MARCADORES_DESEO = [
    "deseo", "te deseo", "quiero verte", "quiero tenerte",
    "necesito", "te necesito", "beso", "besar", "labios",
    "piel", "caricias", "tocarte", "te toque", "abrazarte",
    "tus manos", "tu cuerpo", "placer", "anhelo", "extrano",
    "miss you", "want you", "need you", "desire",
]

MARCADORES_BLOQUEO_AFECTIVO = [
    "no me dejas", "no dejas", "retiras", "te alejas", "distancia",
    "miedo", "temor", "no puedes", "no puedo", "no se logra",
    "no logro", "no alcanz", "rechazas", "rechazo", "bloqueas",
    "te escondes", "callas", "silencio", "no respondes",
    "aguantar", "soportar", "hasta aqui", "ya esta bien",
    "ya basta", "cansado", "cansada", "harto", "harta",
    "ausente", "distante", "distinta", "no sientes", "no te sale",
    "finges", "mientele", "prefiero eso a no tenerte",
]

MARCADORES_CONFLICTO = [
    "enojo", "enoj", "ira", "rabia", "furia", "pelea", "pelear",
    "peleando", "discut", "maltrat", "gritar", "reclamo",
    "odio", "odi", "enemigo", "rencor", "maldigo", "maldicion",
    "fight", "anger", "rage", "hate",
]

MARCADORES_REFLEXION = [
    "no vale la pena", "no valio la pena", "no es asi",
    "tomemos cuidado", "tener cuidado", "cuidado", "advertirte",
    "aprendi", "aprendi", "comprendi", "comprender", "me di cuenta",
    "descubri", "descubriendo", "si seguimos", "evitar",
    "para no", "jamas pensamos", "no quiero volver", "no deberia",
    "no deberiamos", "basta de", "ya basta", "es mejor",
]

MARCADORES_IRA_DIRECTA = [
    "odio", "rabia", "furia", "maldigo", "maldito", "enemigo",
    "quiero pelear", "voy a pelear", "venganza", "destruir",
    "hate", "rage", "revenge",
]

MARCADORES_AMOR_AFIRMATIVO = [
    "te amo", "te quiero", "para siempre", "por siempre", "contigo",
    "dulce amor", "amor eterno", "for all time", "forever",
    "cuidarte", "protegerte", "a tu lado", "mi vida",
    "tenerte para mi", "tenerte", "eres tu", "mi existir y mi amor",
]

MARCADORES_PERDIDA_AUSENCIA = [
    "perdi", "te perdi", "perder", "perdida", "ausencia", "adios",
    "se ira", "se fue", "se va", "para siempre", "ya no esta",
    "ya no estas", "sin ella", "sin el", "sin ti", "no contar",
    "no buscarte", "falta tu calor", "falta", "prescindir de mi",
    "dejaste aqui", "dejaste", "te olvidaste", "me olvide",
    "lejos", "nunca se vaya", "nunca amanezca", "no amanezca",
    "ya no pienso en ti", "ya no lloro", "destrozado", "triste",
    "cansado", "dolor", "duelo", "lloro", "llorar",
]

MARCADORES_RESIGNACION_SUPERACION_DOLOROSA = [
    "ya me resigne", "me resigne", "ya me acostumbre", "me acostumbre",
    "ya me convenci", "me convenci", "al fin logre", "me ensene a vivir",
    "aprendi a vivir", "vivir con", "ya no pienso", "ya no lloro",
    "deje atras", "dejar atras", "seguir sin", "acostumbrarme",
]

MARCADORES_DOLOR_INTENSO = [
    "dolor", "amargo", "triste", "destrozado", "enloquecer",
    "mi vida se apaga", "se apaga", "no soy nada", "irremediable",
    "lloro", "llorar", "cansado", "pena", "penas", "sufrir",
    "sufria", "duele", "duelen", "irreparable", "inestable",
    "me mates", "matarme", "me muero", "morirme",
]

MARCADORES_FELICIDAD_APARENTE = [
    "aunque mientas", "mientas", "haz feliz", "dile que soy feliz",
    "no le digas", "no le cuentes", "fingir", "finjo", "aparentar",
    "sonrio aunque", "sonrio pero", "brindar por su alegria",
]

MARCADORES_DEPENDENCIA_AFECTIVA = [
    "sin su amor no soy nada", "sin tu amor no soy nada",
    "mi vida se apaga", "ella es la estrella", "alumbra mi ser",
    "no soy nada", "me falta", "necesito", "no puedo vivir",
    "no vivo sin", "deten el tiempo", "haz esta noche perpetua",
    "te quiero acompanar", "quiero acompanar", "no eres mia",
    "no eres mio", "me alejo mi vida", "ojala que te escapes",
]

MARCADORES_VINCULO_IMPOSIBLE = [
    "irreparable", "inestable", "no eres mia", "no eres mio",
    "no es mia", "no es mio", "me alejo", "me alejo mi vida",
    "ojala que te escapes", "si lo haces me mates", "me mates",
    "no me pertenece", "no me perteneces", "no eres para mi",
    "victima", "exilio", "vida yo se que no eres mia",
]

MARCADORES_ILUSION_AFECTIVA = [
    "crei", "creia", "pense", "pensaba", "yo dije", "dije",
    "me ama", "me quiere", "llora por mi", "llorabas por mi",
    "seguro que tu", "estoy seguro", "parecia amor",
]

MARCADORES_DESMENTIDO_AFECTIVO = [
    "no era asi", "no era", "era agua", "agua nada mas",
    "nada mas", "confundi", "confundido", "me equivoque",
    "no me quiere", "ya no me quieres", "adios", "me enganaba",
    "falso", "falsa", "ilusion", "sin importancia",
    "amor era agua con sal", "agua con sal", "gotas del alma",
]

MARCADORES_CONTRASTE_AFECTIVO = [
    "pero", "aunque", "sin embargo", "tu no", "tÃº no", "pero ausente",
    "estas conmigo pero ausente", "estÃ¡s conmigo pero ausente",
    "los que me besan tu no", "el que se entrega tu no",
    "los que me abrazan tu no", "las que acarician tu no",
    "fisico", "fisica", "solo cuerpo", "cuerpo pero",
]

MARCADORES_DESCONEXION_AFECTIVA = [
    "ausente", "distante", "distinta", "ya no sientes", "no sientes",
    "no te sale", "no me quieres", "no me amas", "finges", "fingir",
    "mientele", "miente", "mentir", "prefiero eso a no tenerte",
    "te tienes que ir", "siempre te vas", "con quien te vas",
    "otro hombre", "otra persona", "robo", "robado", "me ha robado",
    "que fue de la de antes", "que fue de antes",
]

MARCADORES_RESULTADO_POSITIVO = [
    "alegria", "feliz", "contento", "contenta", "cosas buenas",
    "bueno", "buenas", "bien", "mejor", "mejores", "esperanza",
    "luz", "primavera", "sonrisa", "gozo", "dicha", "paz",
    "alivio", "sanar", "sano", "cura", "curaste", "renacer",
    "florecer", "brilla", "brillar", "triunfo", "triunfar",
    "logre", "logro", "lograr", "cumplio", "cumplido", "se cumplio",
    "sueÃ±o cumplido", "sueno cumplido", "happy", "joy", "good",
]

MARCADORES_TRANSFORMACION_POSITIVA = [
    "cambiaste", "cambio", "cambiar", "convertiste", "convertir",
    "transformaste", "transformar", "hiciste de", "volviste",
    "me ensenaste", "me ensenaste", "aprendi", "sanaste",
    "curaste", "salvaste", "sacaste", "dejaste atras",
    "paso de", "pasar de",
]

MARCADORES_COSTO_SACRIFICIO = [
    "todo lo dio", "lo dio todo", "di todo", "darlo todo",
    "dejando su vida", "dejo su vida", "dejar su vida",
    "hecha pedazos", "hecho pedazos", "pedazos", "pagar muy caro",
    "pague caro", "cansado", "cansada", "sacrificio", "sacrificar",
    "pruebas", "caidas", "caer", "sufrir", "sufrimiento",
    "dolor", "pena", "perder", "perdida", "sin descanso",
    "ya no sera mi voz", "mi voz ya no sera", "mi canto no sera",
    "me callare", "quedare callado", "vida al pasar",
]

MARCADORES_LOGRO_PROPOSITO = [
    "triunfar", "triunfo", "victoria", "lograr", "logre", "logro",
    "cumplio", "cumplido", "sueÃ±o", "sueno", "meta", "objetivo",
    "llegar mas alto", "llegar alto", "alcanzar", "conseguir",
    "lo consegui", "valio la pena", "merecio la pena",
]


def aplicar_compuertas_semanticas_generales(
        texto: str,
        texto_contexto: str,
        clave: str,
        score_final: float,
        score_lexico: float) -> tuple[float, float]:
    """
    Compuertas generales para letras: separan una emocion expresada como
    tema central de una emocion mencionada como ejemplo, obstaculo o contraste.
    """
    normalizado = normalizar_simple(texto)
    contexto = normalizar_simple(texto_contexto or texto)

    afecto = tiene_marcador(contexto, MARCADORES_AFECTO)
    relacion_danada = tiene_marcador(contexto, MARCADORES_RELACION_DANADA)
    cierre = tiene_marcador(contexto, MARCADORES_CIERRE_DECISION)
    amor_afirmativo = tiene_marcador(contexto, MARCADORES_AMOR_AFIRMATIVO)

    deseo = tiene_marcador(contexto, MARCADORES_DESEO)
    bloqueo = tiene_marcador(contexto, MARCADORES_BLOQUEO_AFECTIVO)
    agotamiento = tiene_marcador(contexto, ["aguantar", "soportar", "ya esta bien", "ya basta", "cansado", "harto", "hasta aqui"])
    agotamiento_actual = tiene_marcador(normalizado, ["aguantar", "soportar", "ya esta bien", "ya basta", "cansado", "harto", "hasta aqui"])

    conflicto = tiene_marcador(contexto, MARCADORES_CONFLICTO)
    reflexion = tiene_marcador(contexto, MARCADORES_REFLEXION)
    ira_directa = tiene_marcador(normalizado, MARCADORES_IRA_DIRECTA)
    perdida_ausencia = tiene_marcador(contexto, MARCADORES_PERDIDA_AUSENCIA)
    perdida_actual = tiene_marcador(normalizado, MARCADORES_PERDIDA_AUSENCIA)
    resignacion = tiene_marcador(contexto, MARCADORES_RESIGNACION_SUPERACION_DOLOROSA)
    resignacion_actual = tiene_marcador(normalizado, MARCADORES_RESIGNACION_SUPERACION_DOLOROSA)
    dolor_intenso = tiene_marcador(contexto, MARCADORES_DOLOR_INTENSO)
    dolor_actual = tiene_marcador(normalizado, MARCADORES_DOLOR_INTENSO)
    felicidad_aparente = tiene_marcador(contexto, MARCADORES_FELICIDAD_APARENTE)
    felicidad_aparente_actual = tiene_marcador(normalizado, MARCADORES_FELICIDAD_APARENTE)
    dependencia = tiene_marcador(contexto, MARCADORES_DEPENDENCIA_AFECTIVA)
    dependencia_actual = tiene_marcador(normalizado, MARCADORES_DEPENDENCIA_AFECTIVA)
    vinculo_imposible = tiene_marcador(contexto, MARCADORES_VINCULO_IMPOSIBLE)
    vinculo_imposible_actual = tiene_marcador(normalizado, MARCADORES_VINCULO_IMPOSIBLE)
    ilusion_afectiva = tiene_marcador(contexto, MARCADORES_ILUSION_AFECTIVA)
    ilusion_actual = tiene_marcador(normalizado, MARCADORES_ILUSION_AFECTIVA)
    desmentido_afectivo = tiene_marcador(contexto, MARCADORES_DESMENTIDO_AFECTIVO)
    desmentido_actual = tiene_marcador(normalizado, MARCADORES_DESMENTIDO_AFECTIVO)
    contraste_afectivo = (
        afecto
        and (
            tiene_marcador(contexto, MARCADORES_CONTRASTE_AFECTIVO)
            or tiene_marcador(contexto, MARCADORES_DESCONEXION_AFECTIVA)
        )
        and tiene_marcador(contexto, MARCADORES_DESCONEXION_AFECTIVA)
    )
    contraste_actual = (
        tiene_marcador(normalizado, MARCADORES_CONTRASTE_AFECTIVO)
        or tiene_marcador(normalizado, MARCADORES_DESCONEXION_AFECTIVA)
    ) and tiene_marcador(normalizado, MARCADORES_DESCONEXION_AFECTIVA)
    neg_actual_pos = primera_posicion(
        normalizado,
        MARCADORES_DOLOR_INTENSO + MARCADORES_PERDIDA_AUSENCIA,
    )
    transform_actual_pos = primera_posicion(normalizado, MARCADORES_TRANSFORMACION_POSITIVA)
    positivo_actual_pos = primera_posicion(
        normalizado,
        MARCADORES_RESULTADO_POSITIVO + MARCADORES_AMOR_AFIRMATIVO,
    )
    trayectoria_positiva = (
        neg_actual_pos is not None
        and positivo_actual_pos is not None
        and (
            (transform_actual_pos is not None and transform_actual_pos >= neg_actual_pos)
            or positivo_actual_pos > neg_actual_pos
        )
        and tiene_marcador(normalizado, MARCADORES_RESULTADO_POSITIVO)
    )
    costo_actual_pos = primera_posicion(normalizado, MARCADORES_COSTO_SACRIFICIO)
    logro_actual_pos = primera_posicion(normalizado, MARCADORES_LOGRO_PROPOSITO)
    logro_costoso = (
        costo_actual_pos is not None
        and logro_actual_pos is not None
        and logro_actual_pos >= costo_actual_pos
    )
    logro_costoso_contexto = (
        tiene_marcador(contexto, MARCADORES_COSTO_SACRIFICIO)
        and tiene_marcador(contexto, MARCADORES_LOGRO_PROPOSITO)
    )

    # Amor en contexto de dano relacional no debe convertirse en "amor";
    # normalmente expresa desamor, traicion, cierre o dignidad herida.
    if (afecto and relacion_danada and not amor_afirmativo) or contraste_afectivo:
        if clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.88 if contraste_actual else 0.86)
            score_final = max(score_final, 0.82 if contraste_actual else 0.80)
        elif clave == "amor":
            score_final *= 0.12 if contraste_actual else 0.18
            score_lexico *= 0.20 if contraste_actual else 0.25
        elif clave == "tristeza" and contraste_afectivo:
            score_lexico = max(score_lexico, 0.78 if contraste_actual else 0.68)
            score_final = max(score_final, 0.74 if contraste_actual else 0.64)
        elif clave == "deseo_anhelo" and contraste_afectivo:
            score_lexico = max(score_lexico, 0.72)
            score_final = max(score_final, 0.68)
        elif clave == "molestia_fastidio" and tiene_marcador(contexto, ["finges", "miente", "no te sale", "odio", "siempre te vas"]):
            score_lexico = max(score_lexico, 0.72)
            score_final = max(score_final, 0.68)
        elif clave in {"aprobacion_validacion", "cuidado_carino", "neutral_contemplativa"}:
            score_final *= 0.35

    # VÃ­nculo imposible o autodestructivo: puede decir "te quiero",
    # pero la frase lo rodea con irreparabilidad, inestabilidad, alejamiento
    # o pertenencia negada. No debe ganar amor.
    if afecto and vinculo_imposible:
        if clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.84 if vinculo_imposible_actual else 0.74)
            score_final = max(score_final, 0.80 if vinculo_imposible_actual else 0.70)
        elif clave == "tristeza":
            score_lexico = max(score_lexico, 0.82 if vinculo_imposible_actual else 0.72)
            score_final = max(score_final, 0.78 if vinculo_imposible_actual else 0.68)
        elif clave == "miedo_angustia" and tiene_marcador(contexto, ["me mates", "matarme", "me muero", "escapes"]):
            score_lexico = max(score_lexico, 0.70)
            score_final = max(score_final, 0.66)
        elif clave == "remordimiento_culpa" and tiene_marcador(contexto, ["me alejo", "perdonemos", "nos debemos"]):
            score_lexico = max(score_lexico, 0.68)
            score_final = max(score_final, 0.64)
        elif clave == "amor":
            score_final *= 0.16 if vinculo_imposible_actual else 0.28
            score_lexico *= 0.25

    # Ilusion afectiva desmentida: el contexto completo puede contar una
    # decepcion, pero la frase local conserva su funcion dentro de la escena.
    # Una frase de ilusion inicial no se reescribe hacia atras; la frase que
    # contiene el desmentido se vuelve realizacion/decepcion.
    if ilusion_afectiva and desmentido_afectivo:
        rechazo_actual = tiene_marcador(normalizado, ["ya no me quieres", "no me quiere", "adios", "nada mejor adios"])
        desmentido_reflexivo_actual = tiene_marcador(
            normalizado,
            [
                "no era asi", "era agua", "agua nada mas", "no era lluvia",
                "nunca crei", "agua con sal", "gotas del alma",
            ],
        )
        ilusion_pura_actual = ilusion_actual and not desmentido_actual and not rechazo_actual

        if ilusion_pura_actual:
            if clave == "alegria":
                score_lexico = max(score_lexico, 0.74)
                score_final = max(score_final, 0.70)
            elif clave == "amor":
                score_lexico = max(score_lexico, 0.70)
                score_final = max(score_final, 0.66)
            elif clave == "decepcion_desamor":
                score_final *= 0.45
                score_lexico *= 0.55
        elif desmentido_actual:
            if clave == "realizacion_darse_cuenta":
                if rechazo_actual:
                    score_lexico = max(score_lexico, 0.78)
                    score_final = max(score_final, 0.74)
                else:
                    score_lexico = max(score_lexico, 0.90 if desmentido_reflexivo_actual else 0.84)
                    score_final = max(score_final, 0.86 if desmentido_reflexivo_actual else 0.80)
            elif clave == "decepcion_desamor":
                score_lexico = max(score_lexico, 0.92 if rechazo_actual else 0.74)
                score_final = max(score_final, 0.88 if rechazo_actual else 0.72)
            elif clave == "tristeza" and rechazo_actual:
                score_lexico = max(score_lexico, 0.74)
                score_final = max(score_final, 0.70)
            elif clave in {"amor", "alegria", "aprobacion_validacion"}:
                score_final *= 0.20
                score_lexico *= 0.35
        else:
            if clave == "decepcion_desamor":
                score_lexico = max(score_lexico, 0.64)
                score_final = max(score_final, 0.60)
            elif clave == "realizacion_darse_cuenta":
                score_lexico = max(score_lexico, 0.62)
                score_final = max(score_final, 0.58)

    if cierre:
        if clave == "aceptacion_desapego":
            score_lexico = max(score_lexico, 0.78)
            score_final = max(score_final, 0.74)
        elif clave == "orgullo_autovaloracion" and relacion_danada:
            score_lexico = max(score_lexico, 0.70)
            score_final = max(score_final, 0.68)
        elif clave == "decepcion_desamor" and relacion_danada:
            score_lexico = max(score_lexico, 0.76)
            score_final = max(score_final, 0.72)
        elif clave in {"amor", "aprobacion_validacion", "neutral_contemplativa"}:
            score_final *= 0.45

    # Palabras positivas dentro de perdida, ausencia o resignacion dolorosa
    # no deben dominar: "amor" puede ser objeto perdido, "feliz" puede ser
    # fingido y "ya no lloro" puede ser duelo procesado.
    if (perdida_ausencia or resignacion or dolor_intenso or dependencia or felicidad_aparente) and not trayectoria_positiva:
        if clave == "amor" and not (amor_afirmativo and not (perdida_ausencia or dolor_intenso or dependencia)):
            score_final *= 0.24
            score_lexico *= 0.35
        elif clave == "alegria" and felicidad_aparente:
            score_final *= 0.18
            score_lexico *= 0.30
        elif clave == "aprobacion_validacion" and (resignacion or perdida_ausencia or dolor_intenso):
            score_final *= 0.22
            score_lexico *= 0.35
        elif clave == "redencion_renacer" and (dolor_actual or perdida_actual or "nunca amanezca" in normalizado or "no amanezca" in normalizado):
            score_final *= 0.22
            score_lexico *= 0.35

    if trayectoria_positiva or logro_costoso:
        if clave == "alegria":
            score_lexico = max(score_lexico, 0.86)
            score_final = max(score_final, 0.78 if logro_costoso else 0.82)
        elif clave == "amor" and (afecto or amor_afirmativo):
            score_lexico = max(score_lexico, 0.84)
            score_final = max(score_final, 0.80)
        elif clave == "redencion_renacer":
            score_lexico = max(score_lexico, 0.82)
            score_final = max(score_final, 0.78)
        elif clave == "optimismo_esperanza":
            score_lexico = max(score_lexico, 0.74)
            score_final = max(score_final, 0.70)
        elif clave == "orgullo_autovaloracion" and logro_costoso:
            score_lexico = max(score_lexico, 0.90)
            score_final = max(score_final, 0.86)
        elif clave == "realizacion_darse_cuenta" and logro_costoso:
            score_lexico = max(score_lexico, 0.82)
            score_final = max(score_final, 0.78)
        elif clave in {"tristeza", "duelo_pena", "decepcion_desamor"}:
            score_final *= 0.35
            score_lexico *= 0.50

    if logro_costoso_contexto and not logro_costoso:
        if clave == "orgullo_autovaloracion":
            score_lexico = max(score_lexico, 0.58)
            score_final = max(score_final, 0.54)
        elif clave == "realizacion_darse_cuenta":
            score_lexico = max(score_lexico, 0.62)
            score_final = max(score_final, 0.58)
        elif clave == "confusion":
            score_final *= 0.35
            score_lexico *= 0.50
        elif clave == "molestia_fastidio":
            score_final *= 0.55
            score_lexico *= 0.65

    perdida_voz_silencio = tiene_marcador(
        normalizado,
        [
            "mi voz ya no sera", "ya no sera mi voz", "mi canto no sera",
            "me callare", "quedare callado", "me quedare callado",
            "no sepa que seguir contando",
        ],
    )
    if perdida_voz_silencio and not logro_costoso:
        if clave == "realizacion_darse_cuenta":
            score_lexico = max(score_lexico, 0.76)
            score_final = max(score_final, 0.72)
        elif clave == "tristeza":
            score_lexico = max(score_lexico, 0.74)
            score_final = max(score_final, 0.70)
        elif clave == "confusion":
            score_final *= 0.40
            score_lexico *= 0.55

    if perdida_ausencia and (afecto or dolor_intenso or dependencia) and not trayectoria_positiva:
        if clave == "tristeza":
            score_lexico = max(score_lexico, 0.84 if (dolor_actual or perdida_actual) else 0.74)
            score_final = max(score_final, 0.80 if (dolor_actual or perdida_actual) else 0.70)
        elif clave == "duelo_pena" and (dolor_intenso or perdida_actual):
            score_lexico = max(score_lexico, 0.76)
            score_final = max(score_final, 0.72)
        elif clave == "decepcion_desamor" and afecto:
            score_lexico = max(score_lexico, 0.76)
            score_final = max(score_final, 0.72)
        elif clave == "deseo_anhelo" and (afecto or dependencia):
            score_lexico = max(score_lexico, 0.72)
            score_final = max(score_final, 0.68)

    if resignacion and not trayectoria_positiva:
        if clave == "aceptacion_desapego":
            score_lexico = max(score_lexico, 0.82 if resignacion_actual else 0.72)
            score_final = max(score_final, 0.78 if resignacion_actual else 0.68)
        elif clave == "melancolia_nostalgia":
            score_lexico = max(score_lexico, 0.74 if resignacion_actual else 0.66)
            score_final = max(score_final, 0.70 if resignacion_actual else 0.62)
        elif clave == "tristeza" and (perdida_ausencia or dolor_intenso):
            score_lexico = max(score_lexico, 0.72)
            score_final = max(score_final, 0.68)

    if felicidad_aparente:
        if clave == "tristeza":
            score_lexico = max(score_lexico, 0.82 if felicidad_aparente_actual else 0.72)
            score_final = max(score_final, 0.78 if felicidad_aparente_actual else 0.68)
        elif clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.74)
            score_final = max(score_final, 0.70)
        elif clave == "deseo_anhelo" and afecto:
            score_lexico = max(score_lexico, 0.70)
            score_final = max(score_final, 0.66)

    if dependencia and (dolor_intenso or perdida_ausencia) and not trayectoria_positiva:
        if clave == "tristeza":
            score_lexico = max(score_lexico, 0.88 if dependencia_actual else 0.78)
            score_final = max(score_final, 0.84 if dependencia_actual else 0.74)
        elif clave == "deseo_anhelo":
            score_lexico = max(score_lexico, 0.76)
            score_final = max(score_final, 0.72)
        elif clave == "miedo_angustia" and ("se apaga" in contexto or "enloquecer" in contexto):
            score_lexico = max(score_lexico, 0.70)
            score_final = max(score_final, 0.66)

    # Deseo + obstaculo afectivo: no es amor pleno, sino deseo contenido,
    # frustracion o rechazo percibido.
    if deseo and bloqueo:
        if clave == "deseo_anhelo":
            score_lexico = max(score_lexico, 0.84)
            score_final = max(score_final, 0.76 if agotamiento_actual else 0.80)
        elif clave == "molestia_fastidio" and agotamiento:
            score_lexico = max(score_lexico, 0.86 if agotamiento_actual else 0.76)
            score_final = max(score_final, 0.84 if agotamiento_actual else 0.72)
        elif clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.62)
            score_final = max(score_final, 0.62)
        elif clave == "miedo_angustia" and "miedo" in contexto and "tengo miedo" not in contexto:
            score_final *= 0.60
        elif clave == "amor" and not amor_afirmativo:
            score_final *= 0.42
            score_lexico *= 0.50

    # Si el conflicto aparece dentro de aprendizaje/advertencia, no es ira
    # dominante. Si aparece sin ese marco y con intensidad, si puede serlo.
    if conflicto and reflexion and not ira_directa:
        if clave == "realizacion_darse_cuenta":
            score_lexico = max(score_lexico, 0.78)
            score_final = max(score_final, 0.74)
        elif clave == "remordimiento_culpa":
            score_lexico = max(score_lexico, 0.66)
            score_final = max(score_final, 0.66)
        elif clave in {"ira", "molestia_fastidio", "desaprobacion_rechazo"}:
            score_final *= 0.35
            score_lexico *= 0.50
    elif conflicto and ira_directa:
        if clave == "ira":
            score_lexico = max(score_lexico, 0.84)
            score_final = max(score_final, 0.80)
        elif clave == "orgullo_autovaloracion" and relacion_danada:
            score_lexico = max(score_lexico, 0.70)
            score_final = max(score_final, 0.68)

    return score_final, score_lexico


def es_relleno_vocal(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    if not normalizado:
        return True
    tokens = normalizado.split()
    if not tokens:
        return True
    relleno = {
        "uh", "uuh", "uhh", "oh", "ooh", "ohh", "ah", "aah", "ahh",
        "eh", "mm", "mmm", "hmm", "na", "la", "woh", "woah", "yeah",
        "hey",
    }
    return all(token in relleno for token in tokens)


def calcular_score_lexico_frase(texto: str, clave: str) -> float:
    base = calcular_score_lexico(texto, clave)
    terminos = LEXICOS_EXTRA_MUSICA.get(clave, [])
    if not terminos:
        return base

    normalizado = normalizar_simple(texto)
    peso = 0.0
    for termino in terminos:
        termino_norm = normalizar_simple(termino)
        if termino_norm and termino_norm in normalizado:
            peso += 1.5 if " " in termino_norm else 1.0
    extra = min(1.0, peso / 3.0)
    return max(base, extra)


def ajustar_score_por_contexto(
        texto: str,
        clave: str,
        score_final: float,
        score_lexico: float,
        texto_contexto: str = "") -> tuple[float, float]:
    """
    Reglas de interpretacion musical para frases donde una lectura literal
    cambia el sentido emocional. Ejemplo: "ya no vivo por vivir" es negativo
    si aparece aislado, pero en contexto de "canto/vivo contento/por fin" es
    proposito vital y redencion.
    """
    normalizado = normalizar_simple(texto)
    normalizado_contexto = normalizar_simple(texto_contexto or texto)

    contexto_proposito = (
        "ya no vivo por vivir" in normalizado
        or "me ensenaste a vivir" in normalizado
        or "ensenaste a vivir" in normalizado
        or "ensenando a vivir" in normalizado
        or "hoy canto y vivo contento" in normalizado
        or "hoy canto y vivo contenta" in normalizado
        or ("por fin" in normalizado and "ya no vivo" in normalizado)
        or ("razon" in normalizado and "vivir" in normalizado)
        or ("motivo" in normalizado and "vivir" in normalizado)
    )
    contexto_positivo = any(
        marca in normalizado
        for marca in [
            "contento", "contenta", "por fin", "me ensenaste a vivir",
            "ensenaste a vivir", "me ensenaste a querer", "ensenaste a querer",
            "me enamore", "enamore", "te quiero tanto", "dulce amor",
            "fui enamorando", "amor", "querer",
            "volveras a ver la luz", "volveras", "esperanza",
        ]
    )
    contexto_doloroso = any(
        marca in normalizado
        for marca in [
            "me dejo atras", "me dejo atras", "no hay tiempo que pueda valer",
            "no quiero perecer", "no puedo perecer", "empece a perecer",
            "volvi a cantar", "no supe alcanzar", "tuve que callar",
            "no vas a escuchar", "mis cantos se ahogan", "alcohol",
        ]
    )

    contexto_transformacion = (
        contexto_proposito
        or "nuevo amanecer" in normalizado
        or "vendra un nuevo amanecer" in normalizado
        or "vendran nuevos dias" in normalizado
        or "volveras a ver la luz" in normalizado
        or "despues del dolor" in normalizado
        or "del dolor" in normalizado and "amanecer" in normalizado
    )

    if clave == "redencion_renacer" and contexto_doloroso and not contexto_proposito:
        score_final *= 0.22
        score_lexico *= 0.35

    if contexto_transformacion and contexto_positivo and not (contexto_doloroso and not contexto_proposito):
        if clave == "redencion_renacer":
            score_lexico = max(score_lexico, 1.0)
            score_final = max(score_final, 0.84)
        elif clave == "amor":
            score_lexico = max(score_lexico, 0.75)
            score_final = max(score_final, 0.72)
        elif clave == "optimismo_esperanza":
            score_lexico = max(score_lexico, 0.75)
            score_final = max(score_final, 0.70)
        elif clave in {"molestia_fastidio", "miedo_angustia", "tristeza"}:
            score_final *= 0.18

    if "dame otro amanecer" in normalizado or "como lo hiciste aquella vez" in normalizado:
        if clave == "deseo_anhelo":
            score_lexico = max(score_lexico, 0.78)
            score_final = max(score_final, 0.74)
        elif clave == "redencion_renacer" and "nuevo amanecer" not in normalizado:
            score_final *= 0.28
            score_lexico *= 0.45

    if "no quiero perecer" in normalizado or "no puedo perecer" in normalizado or "empece a perecer" in normalizado:
        if clave == "miedo_angustia":
            score_lexico = max(score_lexico, 0.86)
            score_final = max(score_final, 0.78)
        elif clave == "duelo_pena":
            score_lexico = max(score_lexico, 0.74)
            score_final = max(score_final, 0.70)
        elif clave == "redencion_renacer":
            score_final *= 0.24
            score_lexico *= 0.35

    if any(marca in normalizado for marca in ["no supe alcanzar", "tuve que callar", "no vas a escuchar"]):
        if clave == "tristeza":
            score_lexico = max(score_lexico, 0.82)
            score_final = max(score_final, 0.76)
        elif clave == "duelo_pena":
            score_lexico = max(score_lexico, 0.76)
            score_final = max(score_final, 0.72)
        elif clave == "redencion_renacer":
            score_final *= 0.20
            score_lexico *= 0.35

    conflicto_mencionado = any(
        marca in normalizado_contexto
        for marca in [
            "enojarnos", "enojo", "enojado", "enojada", "peleando",
            "pelear", "maltratarnos", "maltrato", "decirnos cosas",
        ]
    )
    marco_reflexivo = any(
        marca in normalizado_contexto
        for marca in [
            "no valio la pena", "no valio la pena enojarnos",
            "el amor no es asi", "quiero advertirte", "tomemos cuidado",
            "si seguimos peleando", "se nos va la mano",
            "jamas pensamos", "jamÃ¡s pensamos",
        ]
    )

    if conflicto_mencionado and marco_reflexivo:
        if clave == "realizacion_darse_cuenta":
            score_lexico = max(score_lexico, 0.86)
            score_final = max(score_final, 0.80)
        elif clave == "remordimiento_culpa":
            score_lexico = max(score_lexico, 0.72)
            score_final = max(score_final, 0.70)
        elif clave == "aceptacion_desapego":
            score_lexico = max(score_lexico, 0.58)
            score_final = max(score_final, 0.60)
        elif clave in {"ira", "molestia_fastidio", "desaprobacion_rechazo"}:
            ira_directa = any(
                marca in normalizado
                for marca in ["odio", "rabia", "furia", "maldito", "quiero pelear", "voy a pelear"]
            )
            if not ira_directa:
                score_final *= 0.26
                score_lexico *= 0.45

    marcas_traicion = [
        "fuiste infiel", "noche infiel", "infiel en mi morada",
        "te olvidaste de mi", "amor ajeno", "no eras mi camino",
        "ya no quiero nada contigo", "no seremos ya ni amigos",
        "me convierto en tu enemigo", "ahora tu quieres volver",
        "fabricaste una noche infiel", "olvidandote que de mi eras amada",
    ]
    marcas_cierre = [
        "en silencio me alejare", "no pienso reprocharte nada",
        "ya no quiero nada contigo", "no seremos ya ni amigos",
        "me convierto en tu enemigo", "no eras mi camino",
    ]
    contexto_traicion = any(marca in normalizado_contexto for marca in marcas_traicion)
    contexto_cierre = any(marca in normalizado_contexto for marca in marcas_cierre)
    traicion_actual = any(marca in normalizado for marca in marcas_traicion)
    cierre_actual = any(marca in normalizado for marca in marcas_cierre)
    realizacion_actual = any(marca in normalizado for marca in ["corria el velo", "descubriendo", "no eras mi camino"])
    enemigo_actual = "enemigo" in normalizado

    if contexto_traicion or contexto_cierre:
        if clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.90 if traicion_actual else 0.66)
            score_final = max(score_final, 0.84 if traicion_actual else 0.64)
        elif clave == "aceptacion_desapego" and cierre_actual:
            score_lexico = max(score_lexico, 0.92)
            score_final = max(score_final, 0.84 if enemigo_actual else 0.86)
        elif clave == "realizacion_darse_cuenta" and realizacion_actual:
            score_lexico = max(score_lexico, 0.90)
            score_final = max(score_final, 0.88)
        elif clave == "orgullo_autovaloracion" and (
            enemigo_actual or "no pienso reprocharte" in normalizado or "ya no quiero nada contigo" in normalizado
        ):
            score_lexico = max(score_lexico, 0.86)
            score_final = max(score_final, 0.82)
        elif clave == "ira" and enemigo_actual:
            score_lexico = max(score_lexico, 0.90)
            score_final = max(score_final, 0.88)
        elif clave in {"amor", "aprobacion_validacion", "neutral_contemplativa", "cuidado_carino"}:
            score_final *= 0.18
            score_lexico *= 0.25

    marcas_deseo = [
        "no me dejas que te toque", "roce tu mejilla", "te deseo",
        "llene de caricias", "tus manos", "caricias", "beso",
    ]
    marcas_bloqueo = [
        "retiras al momento", "tanto miedo", "ya esta bien",
        "ninerias", "hasta aqui he podido aguantar",
        "siempre estas diciendo menos", "no me dejas",
    ]
    deseo_bloqueado = (
        any(marca in normalizado_contexto for marca in marcas_deseo)
        and any(marca in normalizado_contexto for marca in marcas_bloqueo)
    )
    agotamiento_actual = any(
        marca in normalizado
        for marca in ["aguantar", "soportar", "ya esta bien", "ya basta", "cansado", "harto", "hasta aqui"]
    )

    if deseo_bloqueado:
        if clave == "deseo_anhelo":
            score_lexico = max(score_lexico, 0.88)
            score_final = max(score_final, 0.76 if agotamiento_actual else 0.82)
        elif clave == "molestia_fastidio":
            score_lexico = max(score_lexico, 0.88 if agotamiento_actual else 0.78)
            score_final = max(score_final, 0.86 if agotamiento_actual else 0.74)
        elif clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.68)
            score_final = max(score_final, 0.66)
        elif clave == "miedo_angustia" and "miedo" in normalizado_contexto:
            score_lexico = max(score_lexico, 0.64)
            score_final = max(score_final, 0.62)
        elif clave == "amor":
            score_final *= 0.38
            score_lexico *= 0.45

    score_final, score_lexico = aplicar_compuertas_semanticas_generales(
        texto,
        texto_contexto,
        clave,
        score_final,
        score_lexico,
    )

    return score_final, score_lexico


def dividir_en_segmentos(letra: str, min_words: int = 4, max_words: int = 24) -> list[dict]:
    """
    Divide por estrofas y versos, agrupando versos demasiado cortos.
    Devuelve segmentos con estrofa e indice local.
    """
    bloques = re.split(r"\n\s*\n+", letra)
    segmentos = []

    for idx_bloque, bloque in enumerate(bloques, start=1):
        lineas = []
        for linea in bloque.splitlines():
            linea = limpiar_linea(linea)
            if not linea or FILLER_RE.match(linea) or es_relleno_vocal(linea):
                continue
            lineas.append(linea)

        buffer = []
        for linea in lineas:
            palabras_buffer = sum(len(x.split()) for x in buffer)
            palabras_linea = len(linea.split())

            if not buffer:
                buffer.append(linea)
                continue

            if palabras_buffer < min_words or palabras_buffer + palabras_linea <= max_words:
                buffer.append(linea)
            else:
                texto = " ".join(buffer).strip()
                if len(texto.split()) >= min_words:
                    segmentos.append({
                        "estrofa": idx_bloque,
                        "texto": texto,
                    })
                buffer = [linea]

        if buffer:
            texto = " ".join(buffer).strip()
            if (
                len(texto.split()) >= min_words
                and not FILLER_RE.match(texto)
                and not es_relleno_vocal(texto)
            ):
                segmentos.append({
                    "estrofa": idx_bloque,
                    "texto": texto,
                })

    for idx, segmento in enumerate(segmentos, start=1):
        segmento["segmento"] = idx
    return segmentos


def construir_contexto_segmento(segmentos: list[dict], indice: int, max_words: int = 70) -> str:
    """
    Une segmento anterior, actual y posterior para resolver frases ambiguas.
    La salida de reportes conserva la frase original; el contexto solo ayuda
    a clasificarla.
    """
    partes = []
    for pos in (indice - 1, indice, indice + 1):
        if 0 <= pos < len(segmentos):
            partes.append(segmentos[pos]["texto"])

    palabras = " ".join(partes).split()
    if len(palabras) <= max_words:
        return " ".join(palabras)

    # Mantener el segmento actual completo y recortar el contexto alrededor.
    actual = segmentos[indice]["texto"].split()
    margen = max(0, (max_words - len(actual)) // 2)
    anterior = segmentos[indice - 1]["texto"].split()[-margen:] if indice > 0 else []
    posterior = segmentos[indice + 1]["texto"].split()[:margen] if indice + 1 < len(segmentos) else []
    return " ".join(anterior + actual + posterior)


def detectar_arco_local(texto_contexto: str) -> set[str]:
    """
    Detecta tendencias emocionales del vecindario de una frase. No sustituye
    al Transformer: solo corrige frases cortas que dependen de lo anterior.
    """
    normalizado = normalizar_simple(texto_contexto)
    arcos = set()

    positivo_amor = any(
        marca in normalizado
        for marca in [
            "te quiero", "te amo", "amor", "dulce amor", "enamore",
            "enamorando", "beso", "labios", "carino", "corazon",
            "heart", "love", "darling", "baby",
        ]
    )
    proposito = any(
        marca in normalizado
        for marca in [
            "ya no vivo por vivir", "ensenaste a vivir", "me ensenaste a vivir",
            "por fin", "contento", "contenta", "razon de vivir",
            "motivo para vivir",
        ]
    )
    esperanza = any(
        marca in normalizado
        for marca in [
            "nuevo amanecer", "volveras a ver la luz", "luz", "amanecer",
            "futuro", "esperanza", "salir adelante", "new morning",
            "see the light", "hope",
        ]
    )
    dolor_impotencia = any(
        marca in normalizado
        for marca in [
            "me dejo atras", "no hay tiempo que pueda valer",
            "no quiero perecer", "no puedo perecer", "empece a perecer",
            "volvi a cantar", "no supe alcanzar", "tuve que callar",
            "no vas a escuchar", "mis cantos se ahogan",
        ]
    )
    traicion_cierre = any(
        marca in normalizado
        for marca in [
            "fuiste infiel", "noche infiel", "infiel en mi morada",
            "te olvidaste de mi", "amor ajeno", "no eras mi camino",
            "ya no quiero nada contigo", "no seremos ya ni amigos",
            "me convierto en tu enemigo", "ahora tu quieres volver",
        ]
    )
    despedida_madura = any(
        marca in normalizado
        for marca in [
            "no sirve el rencor", "decir adios es crecer", "poder decir adios",
            "media vuelta", "me ire con el sol", "dar la media vuelta",
            "let go", "move on",
        ]
    )
    conflicto_reflexivo = (
        any(
            marca in normalizado
            for marca in [
                "enojarnos", "enojo", "peleando", "pelear",
                "maltratarnos", "decirnos cosas",
            ]
        )
        and any(
            marca in normalizado
            for marca in [
                "no valio la pena", "el amor no es asi",
                "quiero advertirte", "tomemos cuidado",
                "si seguimos peleando", "se nos va la mano",
                "jamas pensamos", "jamÃ¡s pensamos",
            ]
        )
    )

    if positivo_amor and not traicion_cierre:
        arcos.add("amor")
    if proposito or (positivo_amor and esperanza and not dolor_impotencia):
        arcos.add("redencion_renacer")
    if esperanza:
        arcos.add("optimismo_esperanza")
    if despedida_madura:
        arcos.add("aceptacion_desapego")
    if conflicto_reflexivo:
        arcos.add("conflicto_reflexivo")
    if traicion_cierre:
        arcos.add("traicion_cierre")
    return arcos


def aplicar_arco_local(clave: str, score_final: float, score_lexico: float, arcos: set[str]) -> tuple[float, float]:
    if not arcos:
        return score_final, score_lexico

    return score_final, score_lexico


def aplicar_arco_local_segmento(
        texto: str,
        clave: str,
        score_final: float,
        score_lexico: float,
        arcos: set[str]) -> tuple[float, float]:
    """
    Usa el contexto como desempate, pero no deja que una emocion del
    vecindario invada frases con sentido propio. Ejemplo: en una estrofa
    positiva, "ya no vivo por vivir" se corrige a redencion; pero una frase
    como "me fui enamorando" debe seguir siendo amor/deseo.
    """
    if not arcos:
        return score_final, score_lexico

    normalizado = normalizar_simple(texto)
    senal_redencion = any(
        marca in normalizado
        for marca in [
            "ya no vivo por vivir", "ensenaste a vivir", "me ensenaste a vivir",
            "por fin", "contento", "contenta", "nuevo amanecer",
            "volveras a ver la luz", "crecer", "luz",
        ]
    )
    senal_amor = any(
        marca in normalizado
        for marca in [
            "te quiero", "te amo", "amor", "dulce amor", "enamore",
            "enamorando", "beso", "labios", "brazos", "corazon",
        ]
    )
    amor_doloroso = (
        senal_amor
        and (
            tiene_marcador(normalizado, MARCADORES_PERDIDA_AUSENCIA)
            or tiene_marcador(normalizado, MARCADORES_DOLOR_INTENSO)
            or tiene_marcador(normalizado, MARCADORES_DEPENDENCIA_AFECTIVA)
            or tiene_marcador(normalizado, MARCADORES_RELACION_DANADA)
            or tiene_marcador(normalizado, MARCADORES_DESCONEXION_AFECTIVA)
        )
    )
    senal_desapego = any(
        marca in normalizado
        for marca in [
            "no sirve el rencor", "decir adios", "media vuelta",
            "me ire con el sol", "te vas porque yo quiero",
        ]
    )
    senal_anhelo_pasado = any(
        marca in normalizado
        for marca in [
            "dame otro amanecer", "como lo hiciste aquella vez",
            "sienteme", "te busque", "hasta amanecer",
        ]
    )
    senal_perecer = any(
        marca in normalizado
        for marca in [
            "no quiero perecer", "no puedo perecer", "empece a perecer",
            "ya empece a perecer", "perecer",
        ]
    )
    senal_impotencia_triste = any(
        marca in normalizado
        for marca in [
            "me dejo atras", "no supe alcanzar", "tuve que callar",
            "no vas a escuchar", "mi canto no vas a escuchar",
            "mis cantos se ahogan", "vida ya me da igual",
        ]
    )
    senal_conflicto = any(
        marca in normalizado
        for marca in [
            "enojarnos", "enojo", "enojado", "peleando", "pelear",
            "maltratarnos", "decirnos cosas",
        ]
    )
    senal_reflexion_conflicto = any(
        marca in normalizado
        for marca in [
            "no valio la pena", "el amor no es asi", "quiero advertirte",
            "tomemos cuidado", "si seguimos peleando", "se nos va la mano",
            "jamas pensamos", "jamÃ¡s pensamos",
        ]
    )
    senal_traicion_cierre = any(
        marca in normalizado
        for marca in [
            "fuiste infiel", "noche infiel", "infiel en mi morada",
            "te olvidaste de mi", "amor ajeno", "no eras mi camino",
            "ya no quiero nada contigo", "no seremos ya ni amigos",
            "me convierto en tu enemigo", "ahora tu quieres volver",
            "no pienso reprocharte", "en silencio me alejare",
        ]
    )

    if clave == "redencion_renacer" and "redencion_renacer" in arcos and senal_redencion:
        score_lexico = max(score_lexico, 0.85)
        score_final = max(score_final, 0.78)
    elif clave == "amor" and "amor" in arcos and senal_amor and not amor_doloroso:
        score_lexico = max(score_lexico, 0.85)
        score_final = max(score_final, 0.78)
    elif clave == "deseo_anhelo" and "amor" in arcos and senal_amor:
        score_lexico = max(score_lexico, 0.70)
        score_final = max(score_final, 0.68)
    elif clave == "aceptacion_desapego" and "aceptacion_desapego" in arcos and senal_desapego:
        score_lexico = max(score_lexico, 0.85)
        score_final = max(score_final, 0.78)

    if clave == "redencion_renacer" and (senal_anhelo_pasado or senal_perecer or senal_impotencia_triste) and not senal_redencion:
        score_final *= 0.20
        score_lexico *= 0.35
    elif clave == "deseo_anhelo" and senal_anhelo_pasado:
        score_lexico = max(score_lexico, 0.78)
        score_final = max(score_final, 0.74)
    elif clave == "miedo_angustia" and senal_perecer:
        score_lexico = max(score_lexico, 0.86)
        score_final = max(score_final, 0.78)
    elif clave == "duelo_pena" and (senal_perecer or senal_impotencia_triste):
        score_lexico = max(score_lexico, 0.78)
        score_final = max(score_final, 0.74)
    elif clave == "tristeza" and senal_impotencia_triste:
        score_lexico = max(score_lexico, 0.82)
        score_final = max(score_final, 0.76)

    if "conflicto_reflexivo" in arcos and (senal_conflicto or senal_reflexion_conflicto):
        if clave == "realizacion_darse_cuenta":
            score_lexico = max(score_lexico, 0.86)
            score_final = max(score_final, 0.80)
        elif clave == "remordimiento_culpa" and senal_conflicto:
            score_lexico = max(score_lexico, 0.72)
            score_final = max(score_final, 0.70)
        elif clave in {"ira", "molestia_fastidio", "desaprobacion_rechazo"}:
            score_final *= 0.30
            score_lexico *= 0.45

    if "traicion_cierre" in arcos and senal_traicion_cierre:
        cierre_actual = any(
            marca in normalizado
            for marca in [
                "en silencio me alejare", "no pienso reprocharte",
                "ya no quiero nada contigo", "no seremos ya ni amigos",
                "me convierto en tu enemigo",
            ]
        )
        realizacion_actual = "no eras mi camino" in normalizado or "corria el velo" in normalizado or "descubriendo" in normalizado
        enemigo_actual = "enemigo" in normalizado
        if clave == "decepcion_desamor":
            score_lexico = max(score_lexico, 0.90)
            score_final = max(score_final, 0.84)
        elif clave == "aceptacion_desapego" and cierre_actual:
            score_lexico = max(score_lexico, 0.92)
            score_final = max(score_final, 0.84 if enemigo_actual else 0.86)
        elif clave == "realizacion_darse_cuenta" and realizacion_actual:
            score_lexico = max(score_lexico, 0.90)
            score_final = max(score_final, 0.88)
        elif clave == "orgullo_autovaloracion" and cierre_actual:
            score_lexico = max(score_lexico, 0.86)
            score_final = max(score_final, 0.82)
        elif clave == "ira" and enemigo_actual:
            score_lexico = max(score_lexico, 0.90)
            score_final = max(score_final, 0.88)
        elif clave in {"amor", "aprobacion_validacion", "neutral_contemplativa", "cuidado_carino"}:
            score_final *= 0.18
            score_lexico *= 0.25

    if "redencion_renacer" in arcos and senal_redencion and clave in {"molestia_fastidio", "miedo_angustia", "tristeza"}:
        score_final *= 0.25
    if "amor" in arcos and senal_amor and clave in {"molestia_fastidio", "miedo_angustia"}:
        score_final *= 0.35
    return score_final, score_lexico


def clasificar_segmento(
        classifier,
        texto: str,
        texto_contexto: str,
        threshold: float,
        top_k: int) -> list[dict]:
    labels = [cat["prompt"] for cat in CATEGORIAS_FRASES]
    prompt_to_cat = {cat["prompt"]: cat for cat in CATEGORIAS_FRASES}

    arcos = detectar_arco_local(texto_contexto)

    resultado = classifier(
        texto_contexto,
        candidate_labels=labels,
        hypothesis_template="{}",
        multi_label=True,
    )

    filas = []
    for prompt, score in zip(resultado["labels"], resultado["scores"]):
        cat = prompt_to_cat[prompt]
        score_transformer = float(score)
        score_lexico_actual = calcular_score_lexico_frase(texto, cat["clave"])
        score_lexico_contexto = calcular_score_lexico_frase(texto_contexto, cat["clave"])
        score_lexico = max(score_lexico_actual, score_lexico_contexto * 0.45)
        score_final = min(1.0, score_transformer + (LEXICO_FRASE_AUXILIAR_PESO * score_lexico))

        if cat["clave"] in {
            "curiosidad_busqueda", "asco_repulsion", "sorpresa_asombro",
            "gratitud", "diversion_ironia", "entusiasmo_emocion",
            "aprobacion_validacion",
        } and score_transformer < 0.35 and score_lexico == 0:
            score_final *= 0.75
        if cat["clave"] == "neutral_contemplativa" and score_transformer < 0.30 and score_lexico == 0:
            score_final *= 0.85
        if score_lexico >= 0.75:
            score_final = min(1.0, score_final + (0.05 * score_lexico))
        score_final, score_lexico = ajustar_score_por_contexto(
            texto,
            cat["clave"],
            score_final,
            score_lexico,
            texto_contexto,
        )
        score_final, score_lexico = aplicar_arco_local_segmento(
            texto,
            cat["clave"],
            score_final,
            score_lexico,
            arcos,
        )

        if score_final >= score_transformer:
            score_final = score_transformer + (0.18 * (score_final - score_transformer))
        else:
            score_final = score_transformer - (0.35 * (score_transformer - score_final))
        score_final = max(0.0, min(1.0, score_final))

        filas.append({
            **cat,
            "score": round(score_final, 6),
            "score_transformer": round(score_transformer, 6),
            "score_lexico": round(score_lexico, 6),
        })

    ranking = sorted(filas, key=lambda x: x["score"], reverse=True)
    seleccion = [fila for fila in ranking if fila["score"] >= threshold]
    if not seleccion:
        seleccion = ranking[:1]
    return seleccion[:top_k]


def etiqueta_lista(etiquetas: list[dict]) -> str:
    return "; ".join(f"{e['clave']}:{e['score']}" for e in etiquetas)


def clasificar_canciones_por_segmentos(
        canciones: list[dict],
        out_dir: str,
        modelo: str,
        threshold: float,
        top_k: int) -> tuple[list[dict], list[dict]]:
    classifier = cargar_zero_shot(modelo)
    filas_segmento = []
    resumenes = []

    for pos, cancion in enumerate(canciones, start=1):
        segmentos = dividir_en_segmentos(cancion["letra"])
        print(
            f"[{pos}/{len(canciones)}] {cancion['artista']} - {cancion['nombre']} "
            f"({len(segmentos)} segmentos)",
            flush=True,
        )

        acumulado = defaultdict(float)
        conteo = Counter()

        for idx_segmento, segmento in enumerate(segmentos):
            contexto_segmento = construir_contexto_segmento(segmentos, idx_segmento)
            etiquetas = clasificar_segmento(
                classifier,
                segmento["texto"],
                contexto_segmento,
                threshold=threshold,
                top_k=top_k,
            )
            principal = etiquetas[0]
            conteo[principal["clave"]] += 1
            for etiqueta in etiquetas:
                acumulado[etiqueta["clave"]] += etiqueta["score"]
                filas_segmento.append({
                    **{k: v for k, v in cancion.items() if k != "letra"},
                    "estrofa": segmento["estrofa"],
                    "segmento": segmento["segmento"],
                    "frase": segmento["texto"],
                    "contexto": contexto_segmento,
                    "categoria_id": etiqueta["id"],
                    "clave": etiqueta["clave"],
                    "label": etiqueta["label"],
                    "grupo": etiqueta["grupo"],
                    "score": etiqueta["score"],
                    "score_transformer": etiqueta["score_transformer"],
                    "score_lexico": etiqueta["score_lexico"],
                    "es_principal_segmento": 1 if etiqueta is principal else 0,
                })

        ranking = sorted(acumulado.items(), key=lambda item: item[1], reverse=True)
        ranking_conteo = conteo.most_common()
        dominante = ranking_conteo[0][0] if ranking_conteo else (ranking[0][0] if ranking else "sin_segmentos")
        acompanantes_lista = []
        for clave, _ in ranking_conteo:
            if clave != dominante and clave not in acompanantes_lista:
                acompanantes_lista.append(clave)
        for clave, _ in ranking:
            if clave != dominante and clave not in acompanantes_lista:
                acompanantes_lista.append(clave)
        acompanantes_lista = acompanantes_lista[:3]
        acompanantes = ", ".join(acompanantes_lista)
        conclusion = construir_conclusion(cancion, dominante, acompanantes, conteo, ranking)
        resumenes.append({
            **{k: v for k, v in cancion.items() if k != "letra"},
            "segmentos": len(segmentos),
            "dominante": dominante,
            "acompanantes": acompanantes,
            "conteo_principales": "; ".join(f"{k}:{v}" for k, v in conteo.most_common()),
            "ranking_score": "; ".join(f"{k}:{round(v, 3)}" for k, v in ranking[:8]),
            "conclusion": conclusion,
        })

    guardar_resultados(filas_segmento, resumenes, out_dir)
    return filas_segmento, resumenes


def construir_conclusion(cancion: dict, dominante: str, acompanantes: str, conteo: Counter, ranking: list[tuple[str, float]]) -> str:
    if not ranking:
        return "No se detectaron segmentos emocionales suficientes."
    partes = [f"Domina {dominante.replace('_', ' ')}"]
    if acompanantes:
        partes.append(f"acompanada por {acompanantes.replace('_', ' ')}")
    if conteo:
        top_conteo = ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in conteo.most_common(3))
        partes.append(f"en las frases principales: {top_conteo}")
    return f"{cancion['nombre']} refleja " + ", ".join(partes) + "."


def guardar_resultados(filas_segmento: list[dict], resumenes: list[dict], out_dir: str):
    campos_segmento = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz", "estrofa", "segmento", "frase",
        "contexto", "categoria_id", "clave", "label", "grupo", "score",
        "score_transformer", "score_lexico", "es_principal_segmento",
    ]
    with open(os.path.join(out_dir, "frases_clasificadas.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos_segmento)
        writer.writeheader()
        for fila in filas_segmento:
            writer.writerow({campo: fila.get(campo, "") for campo in campos_segmento})

    campos_resumen = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "segmentos", "dominante", "acompanantes",
        "conteo_principales", "ranking_score", "conclusion",
    ]
    with open(os.path.join(out_dir, "resumen_canciones.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos_resumen)
        writer.writeheader()
        for fila in resumenes:
            writer.writerow({campo: fila.get(campo, "") for campo in campos_resumen})

    por_categoria = defaultdict(list)
    for fila in filas_segmento:
        por_categoria[fila["clave"]].append(fila)

    categorias_dir = os.path.join(out_dir, "categorias")
    for cat in CATEGORIAS_FRASES:
        ruta = os.path.join(
            categorias_dir,
            f"frase_{cat['id']:02d}_{limpiar_nombre_archivo(cat['clave'])}.txt",
        )
        with open(ruta, "w", encoding="utf-8") as f:
            for idx, fila in enumerate(por_categoria.get(cat["clave"], []), start=1):
                f.write(
                    f"{idx} | {fila['nombre']} - {fila['artista']} | "
                    f"seg={fila['segmento']} | score={fila['score']} | {fila['frase']}\n"
                )

    por_cancion = defaultdict(list)
    for fila in filas_segmento:
        if int(fila.get("es_principal_segmento", 0)) == 1:
            por_cancion[fila["spotify_track_id"]].append(fila)

    resumen_por_id = {fila["spotify_track_id"]: fila for fila in resumenes}
    canciones_dir = os.path.join(out_dir, "por_cancion")
    for idx, resumen in enumerate(resumenes, start=1):
        nombre = limpiar_nombre_archivo(f"{idx:02d}_{resumen['artista']}_{resumen['nombre']}")[:120]
        ruta = os.path.join(canciones_dir, f"{nombre}.txt")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(f"{resumen['nombre']} - {resumen['artista']}\n")
            f.write(f"{resumen['conclusion']}\n\n")
            for fila in sorted(
                por_cancion.get(resumen["spotify_track_id"], []),
                key=lambda x: (int(x["estrofa"]), int(x["segmento"])),
            ):
                f.write(
                    f"[{fila['segmento']}] {fila['clave']} "
                    f"(score={fila['score']}): {fila['frase']}\n"
                )


def guardar_meta(canciones: list[dict], ruido: list[dict], out_dir: str, ancla: datetime, horas: int):
    campos = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz",
    ]
    with open(os.path.join(out_dir, "letras_usadas_meta.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for cancion in canciones:
            writer.writerow({campo: cancion.get(campo, "") for campo in campos})

    campos_ruido = campos + ["motivo_ruido", "longitud"]
    with open(os.path.join(out_dir, "letras_descartadas_ruido.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos_ruido)
        writer.writeheader()
        for cancion in ruido:
            writer.writerow({campo: cancion.get(campo, "") for campo in campos_ruido})

    with open(os.path.join(out_dir, "parametros.txt"), "w", encoding="utf-8") as f:
        f.write(f"ancla_db_utc={ancla.isoformat()}\n")
        f.write(f"ventana_horas={horas}\n")
        f.write(f"inicio_utc={(ancla - timedelta(hours=horas)).isoformat()}\n")
        f.write(f"canciones_usadas={len(canciones)}\n")
        f.write(f"letras_descartadas_ruido={len(ruido)}\n")


def cargar_cancion_desde_txt(ruta_txt: str) -> list[dict]:
    ruta_abs = os.path.abspath(ruta_txt)
    if not os.path.exists(ruta_abs):
        raise FileNotFoundError(f"No se encontro el archivo: {ruta_abs}")

    with open(ruta_abs, "r", encoding="utf-8") as archivo:
        letra = limpiar_letra_cruda(archivo.read())

    nombre = os.path.splitext(os.path.basename(ruta_abs))[0]
    return [{
        "indice": 1,
        "spotify_track_id": f"archivo::{nombre}",
        "nombre": nombre,
        "artista": "archivo_txt",
        "album": "",
        "fecha_reproduccion": "",
        "relevancia_letra": "prueba_txt",
        "patron_escucha": "archivo_independiente",
        "motivo_matriz": "entrada_txt_manual",
        "score_matriz": "",
        "letra": letra,
    }]


def guardar_meta_archivo_txt(canciones: list[dict], out_dir: str, ruta_txt: str):
    campos = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz",
    ]
    with open(os.path.join(out_dir, "letras_usadas_meta.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for cancion in canciones:
            writer.writerow({campo: cancion.get(campo, "") for campo in campos})

    with open(os.path.join(out_dir, "letras_descartadas_ruido.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "spotify_track_id", "nombre", "motivo_ruido", "longitud"])

    with open(os.path.join(out_dir, "parametros.txt"), "w", encoding="utf-8") as f:
        f.write("modo=archivo_txt\n")
        f.write(f"archivo_txt={os.path.abspath(ruta_txt)}\n")
        f.write(f"canciones_usadas={len(canciones)}\n")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasifica frases de letras de ultimas 24h.")
    parser.add_argument("--horas", type=int, default=24)
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--archivo-txt", default=None, help="Procesa solo una letra desde un TXT independiente.")
    parser.add_argument("--modelo", default=MODELO_ZERO_SHOT)
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]):
    args = parse_args(argv)
    out_dir_base = OUT_DIR_TXT if args.archivo_txt and args.out_dir == OUT_DIR else args.out_dir
    out_dir = os.path.abspath(out_dir_base)
    preparar_salida(out_dir)

    if args.archivo_txt:
        canciones = cargar_cancion_desde_txt(args.archivo_txt)
        guardar_meta_archivo_txt(canciones, out_dir, args.archivo_txt)
        print("=== Prueba aislada: frases/estrofas desde TXT ===")
        print(f"Archivo: {os.path.abspath(args.archivo_txt)}")
        print(f"Canciones usadas: {len(canciones)}")
    else:
        canciones, ruido, ancla = cargar_canciones_ultimas_horas(args.horas, args.limite)
        guardar_meta(canciones, ruido, out_dir, ancla, args.horas)
        print("=== Prueba aislada: frases/estrofas ultimas 24h ===")
        print(f"Ancla DB UTC: {ancla.isoformat()}")
        print(f"Ventana: {args.horas}h")
        print(f"Canciones usadas: {len(canciones)}")
        print(f"Descartadas por ruido: {len(ruido)}")
    print(f"Salida: {out_dir}\n")

    filas, resumenes = clasificar_canciones_por_segmentos(
        canciones,
        out_dir=out_dir,
        modelo=args.modelo,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    print("\n=== Resumen ===")
    print(f"Frases clasificadas: {len(filas)}")
    print(f"Canciones resumidas: {len(resumenes)}")
    print(f"Revisa: {os.path.join(out_dir, 'resumen_canciones.csv')}")
    print(f"Revisa: {os.path.join(out_dir, 'por_cancion')}")


if __name__ == "__main__":
    main()

