#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goemotions_transformer_prueba.py
================================
Experimento aislado para clasificar letras crudas con una taxonomia fina
inspirada en GoEmotions.

No modifica main.py, matrices, canciones_*.txt ni moodtracker.db.

Lee moodtracker.db, toma las canciones que ya pasaron los filtros de matriz
(usar_en_matriz = 1), limpia ruido basico de Genius y clasifica las letras
sin lematizar usando zero-shot multilingual NLI.

La clasificacion es multietiqueta:
    una misma cancion puede aparecer en amor, deseo/anhelo y tristeza,
    o en decepcion/desamor y orgullo, etc.

Salidas:
    prueba_transformers_goemotions/
        letras_filtradas_crudas.txt
        letras_filtradas_crudas_meta.csv
        goemotion_00_admiracion_aprecio.txt
        ...
        clasificacion_goemotions_transformer.csv
        top1_goemotions_transformer.csv
        resumen_goemotions_transformer.csv
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
from typing import Iterable


BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "moodtracker.db")
DESCARTADAS_META = os.path.join(BASE_DIR, "canciones_descartadas_meta.txt")
OUT_DIR = os.path.join(BASE_DIR, "prueba_transformers_goemotions")

# Modelo pequeno y multilingue para zero-shot classification.
# Es mas liviano para CPU que mDeBERTa base y funciona en espanol/ingles.
MODELO_ZERO_SHOT = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


GOEMOTIONS_CANCIONES = [
    {
        "id": 0,
        "clave": "admiracion_aprecio",
        "label": "admiracion, aprecio o reconocimiento",
        "prompt": "La letra expresa admiracion, aprecio, reconocimiento o respeto hacia alguien.",
        "grupo": "positiva",
        "descripcion": "valora, admira o reconoce a alguien; amor incondicional o respeto",
    },
    {
        "id": 1,
        "clave": "diversion_ironia",
        "label": "diversion, humor o ironia",
        "prompt": "La letra tiene diversion, humor, juego, burla o ironia.",
        "grupo": "positiva",
        "descripcion": "tono jugueton, burlesco, ligero o ironico",
    },
    {
        "id": 2,
        "clave": "ira",
        "label": "ira, enojo o rabia",
        "prompt": "La letra expresa ira, enojo, rabia o confrontacion intensa.",
        "grupo": "negativa",
        "descripcion": "furia directa, confrontacion o reclamo intenso",
    },
    {
        "id": 3,
        "clave": "molestia_fastidio",
        "label": "molestia, fastidio o irritacion",
        "prompt": "La letra expresa molestia, fastidio, irritacion, hartazgo o cansancio emocional.",
        "grupo": "negativa",
        "descripcion": "enojo leve, cansancio emocional o hartazgo",
    },
    {
        "id": 4,
        "clave": "aprobacion_validacion",
        "label": "aprobacion, apoyo o validacion",
        "prompt": "La letra expresa aprobacion, apoyo, validacion o aceptacion.",
        "grupo": "positiva",
        "descripcion": "apoyo, respaldo, aceptacion o afirmacion de otra persona",
    },
    {
        "id": 5,
        "clave": "cuidado_carino",
        "label": "cuidado, ternura o carino",
        "prompt": "La letra expresa cuidado, ternura, carino, proteccion o compania.",
        "grupo": "amor_apego",
        "descripcion": "proteccion, ternura, cuidado afectivo o compania estable",
    },
    {
        "id": 6,
        "clave": "confusion",
        "label": "confusion o desconcierto",
        "prompt": "La letra expresa confusion, desconcierto, contradiccion o no entender lo que ocurre.",
        "grupo": "respuesta",
        "descripcion": "no entender lo que ocurre, contradiccion o paradoja",
    },
    {
        "id": 7,
        "clave": "curiosidad_busqueda",
        "label": "curiosidad, busqueda o pregunta",
        "prompt": "La letra expresa curiosidad real, busqueda de respuestas, preguntas o exploracion.",
        "grupo": "respuesta",
        "descripcion": "busqueda de respuestas, duda activa o exploracion",
    },
    {
        "id": 8,
        "clave": "deseo_anhelo",
        "label": "deseo, anhelo o nostalgia por alguien",
        "prompt": "La letra expresa deseo, anhelo, nostalgia, necesidad o extranar a alguien.",
        "grupo": "amor_apego",
        "descripcion": "querer a alguien ausente, necesidad, deseo romantico o espera",
    },
    {
        "id": 9,
        "clave": "decepcion_desamor",
        "label": "decepcion, desamor o desencanto",
        "prompt": "La letra expresa decepcion amorosa, desamor, ruptura, desencanto u orgullo herido.",
        "grupo": "negativa",
        "descripcion": "ruptura, orgullo herido, relacion fallida o amor no correspondido",
    },
    {
        "id": 10,
        "clave": "desaprobacion_rechazo",
        "label": "desaprobacion, rechazo o critica",
        "prompt": "La letra expresa desaprobacion, rechazo, critica social o juicio negativo.",
        "grupo": "negativa",
        "descripcion": "rechazo moral, critica social o juicio negativo",
    },
    {
        "id": 11,
        "clave": "asco_repulsion",
        "label": "asco, repulsion o desprecio",
        "prompt": "La letra expresa asco, repulsion, desprecio, corrupcion o rechazo moral intenso.",
        "grupo": "negativa",
        "descripcion": "rechazo intenso, corrupcion, degradacion o desprecio",
    },
    {
        "id": 12,
        "clave": "verguenza_vulnerabilidad",
        "label": "verguenza, pudor o vulnerabilidad",
        "prompt": "La letra expresa verguenza, pudor, vulnerabilidad, secreto o dignidad expuesta.",
        "grupo": "negativa",
        "descripcion": "exposicion emocional, dignidad herida o secreto incomodo",
    },
    {
        "id": 13,
        "clave": "entusiasmo_emocion",
        "label": "entusiasmo, emocion o euforia",
        "prompt": "La letra expresa entusiasmo, euforia, celebracion, energia positiva o exaltacion.",
        "grupo": "positiva",
        "descripcion": "intensidad positiva, exaltacion, celebracion o impulso",
    },
    {
        "id": 14,
        "clave": "miedo_angustia",
        "label": "miedo, temor o angustia",
        "prompt": "La letra expresa miedo, temor, angustia, peligro, amenaza o panico.",
        "grupo": "negativa",
        "descripcion": "amenaza, oscuridad, ansiedad intensa o peligro",
    },
    {
        "id": 15,
        "clave": "gratitud",
        "label": "gratitud o agradecimiento",
        "prompt": "La letra expresa gratitud, agradecimiento o reconocimiento por algo recibido.",
        "grupo": "positiva",
        "descripcion": "dar gracias, valorar lo recibido o reconocer un regalo emocional",
    },
    {
        "id": 16,
        "clave": "duelo_pena",
        "label": "duelo, pena profunda o perdida",
        "prompt": "La letra expresa duelo, pena profunda, perdida, muerte o despedida dolorosa.",
        "grupo": "negativa",
        "descripcion": "perdida, muerte, despedida o dolor profundo",
    },
    {
        "id": 17,
        "clave": "alegria",
        "label": "alegria o felicidad",
        "prompt": "La letra expresa alegria, felicidad, gozo o bienestar.",
        "grupo": "positiva",
        "descripcion": "felicidad general, bienestar o gozo",
    },
    {
        "id": 18,
        "clave": "amor",
        "label": "amor romantico o amor profundo",
        "prompt": "La letra expresa amor romantico, amor profundo, entrega afectiva o amor eterno.",
        "grupo": "amor_apego",
        "descripcion": "declaracion de amor, vinculo romantico o entrega afectiva",
    },
    {
        "id": 19,
        "clave": "nerviosismo_ansiedad",
        "label": "nerviosismo, ansiedad o inquietud",
        "prompt": "La letra expresa nerviosismo, ansiedad, inquietud, tension o incertidumbre.",
        "grupo": "negativa",
        "descripcion": "intranquilidad, tension interna o incertidumbre afectiva",
    },
    {
        "id": 20,
        "clave": "optimismo_esperanza",
        "label": "optimismo, esperanza o fe en el futuro",
        "prompt": "La letra expresa optimismo, esperanza, fe en el futuro, promesa o superacion.",
        "grupo": "positiva",
        "descripcion": "esperanza, reconciliacion, promesa, futuro o superacion",
    },
    {
        "id": 21,
        "clave": "orgullo_autovaloracion",
        "label": "orgullo, dignidad o autovaloracion",
        "prompt": "La letra expresa orgullo, dignidad, autovaloracion, desafio o seguridad personal.",
        "grupo": "positiva",
        "descripcion": "afirmacion personal, dignidad, desafio o seguridad de valor propio",
    },
    {
        "id": 22,
        "clave": "realizacion_darse_cuenta",
        "label": "realizacion, revelacion o darse cuenta",
        "prompt": "La letra expresa realizacion, revelacion, aceptar una verdad o darse cuenta.",
        "grupo": "respuesta",
        "descripcion": "comprender una verdad, aceptar un destino o tomar conciencia",
    },
    {
        "id": 23,
        "clave": "alivio_liberacion",
        "label": "alivio, liberacion o descanso",
        "prompt": "La letra expresa alivio, liberacion, descanso, soltar una carga o dejar atras.",
        "grupo": "positiva",
        "descripcion": "soltar una carga, dejar atras o sentirse libre",
    },
    {
        "id": 24,
        "clave": "remordimiento_culpa",
        "label": "remordimiento, culpa o arrepentimiento",
        "prompt": "La letra expresa remordimiento, culpa, arrepentimiento, error o pedir perdon.",
        "grupo": "negativa",
        "descripcion": "culpa, pedir perdon, reconocer error o arrepentirse",
    },
    {
        "id": 25,
        "clave": "tristeza",
        "label": "tristeza o melancolia",
        "prompt": "La letra expresa tristeza, melancolia, soledad o abatimiento.",
        "grupo": "negativa",
        "descripcion": "melancolia, soledad, tristeza de fondo o abatimiento",
    },
    {
        "id": 26,
        "clave": "sorpresa_asombro",
        "label": "sorpresa o asombro",
        "prompt": "La letra expresa sorpresa, asombro, impacto o un giro inesperado.",
        "grupo": "respuesta",
        "descripcion": "impacto por algo inesperado, asombro o giro emocional",
    },
    {
        "id": 27,
        "clave": "neutral_contemplativa",
        "label": "neutralidad, contemplacion o narracion sin emocion dominante",
        "prompt": "La letra es neutral, contemplativa, atmosferica o narrativa sin emocion dominante.",
        "grupo": "neutral",
        "descripcion": "relato descriptivo, atmosfera contemplativa o emocion poco definida",
    },
]


GENIUS_NOISE = re.compile(
    r"(\d+\s*Contributors?.*?Lyrics|"
    r"\[.*?\]|"
    r"You might also like|"
    r"Embed$)",
    re.IGNORECASE | re.MULTILINE,
)

# El lexico manual queda desactivado como clasificador.
# Las decisiones principales deben venir del transformer; las reglas externas
# solo pueden actuar como ajustes contextuales de bajo peso.
LEXICOS_CANCIONES = {}
LEXICO_AUXILIAR_PESO = 0.06

def reparar_mojibake(texto: str) -> str:
    if not isinstance(texto, str):
        return texto
    if "Ãƒ" not in texto and "Ã‚" not in texto:
        return texto
    for encoding in ("latin1", "cp1252"):
        try:
            reparado = texto.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if reparado.count("Ãƒ") + reparado.count("Ã‚") < texto.count("Ãƒ") + texto.count("Ã‚"):
            return reparado
    return texto


def limpiar_letra_cruda(letra: str) -> str:
    letra = reparar_mojibake(letra or "")
    letra = GENIUS_NOISE.sub(" ", letra)
    letra = re.sub(r"^\s*\d+\s*$", "", letra, flags=re.MULTILINE)
    letra = re.sub(r"\r\n?", "\n", letra)
    letra = re.sub(r"\n{3,}", "\n\n", letra)
    letra = re.sub(r"[ \t]{2,}", " ", letra)
    return letra.strip()


def limpiar_nombre_archivo(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-zA-Z0-9_]+", "_", texto.lower()).strip("_")
    return texto


def normalizar_para_lexico(texto: str) -> str:
    texto = reparar_mojibake(texto or "").lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def calcular_score_lexico(letra: str, clave: str) -> float:
    texto = normalizar_para_lexico(letra)
    terminos = LEXICOS_CANCIONES.get(clave, [])
    if not terminos:
        return 0.0

    coincidencias = 0
    peso = 0.0
    for termino in terminos:
        termino_norm = normalizar_para_lexico(termino)
        if not termino_norm:
            continue
        if termino_norm in texto:
            coincidencias += 1
            peso += 1.5 if " " in termino_norm else 1.0

    if coincidencias == 0:
        return 0.0
    return min(1.0, peso / 4.0)


def leer_descartadas() -> set[str]:
    ids = set()
    if not os.path.exists(DESCARTADAS_META):
        return ids
    with open(DESCARTADAS_META, "r", encoding="utf-8") as archivo:
        next(archivo, None)
        for linea in archivo:
            partes = linea.strip().split("|")
            if len(partes) >= 2 and partes[1].strip():
                ids.add(partes[1].strip())
    return ids


def cargar_letras_filtradas(limite: int | None = None) -> list[dict]:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"No se encontro la base: {DB_PATH}")

    descartadas = leer_descartadas()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT spotify_track_id, nombre, artista, album, fecha_reproduccion,
               letra, relevancia_letra, patron_escucha, usar_en_matriz,
               motivo_matriz, score_matriz
        FROM canciones
        WHERE letra IS NOT NULL
          AND letra != ''
          AND COALESCE(usar_en_matriz, 1) = 1
        ORDER BY fecha_reproduccion DESC
    """)

    canciones = []
    vistos = set()
    for row in cur.fetchall():
        track_id = row["spotify_track_id"]
        if not track_id or track_id in vistos or track_id in descartadas:
            continue

        letra_limpia = limpiar_letra_cruda(row["letra"])
        if len(letra_limpia) < 20:
            continue

        vistos.add(track_id)
        canciones.append({
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
            "letra": letra_limpia,
        })
        if limite and len(canciones) >= limite:
            break

    conn.close()
    return canciones


def preparar_salida(out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def guardar_letras_crudas(canciones: list[dict], out_dir: str):
    ruta_txt = os.path.join(out_dir, "letras_filtradas_crudas.txt")
    ruta_meta = os.path.join(out_dir, "letras_filtradas_crudas_meta.csv")

    with open(ruta_txt, "w", encoding="utf-8") as archivo:
        for cancion in canciones:
            letra = re.sub(r"\s+", " ", cancion["letra"]).strip()
            archivo.write(f"{cancion['indice']} | {letra}\n")

    campos = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz",
    ]
    with open(ruta_meta, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for cancion in canciones:
            writer.writerow({campo: cancion.get(campo, "") for campo in campos})


def cargar_zero_shot(modelo: str):
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Ejecuta:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install transformers torch tqdm\n"
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    print(f"[Zero-shot] Modelo: {modelo}")
    print(f"[Zero-shot] Dispositivo: {'cuda' if device == 0 else 'cpu'}")
    return pipeline(
        "zero-shot-classification",
        model=modelo,
        device=device,
    )


def fragmentar_letra(texto: str, max_chars: int, max_chunks: int) -> list[str]:
    texto = re.sub(r"\s+", " ", texto).strip()
    if len(texto) <= max_chars:
        return [texto]

    oraciones = re.split(r"(?<=[.!?Â¿Â¡])\s+", texto)
    chunks = []
    actual = ""
    for oracion in oraciones:
        if len(actual) + len(oracion) + 1 <= max_chars:
            actual = f"{actual} {oracion}".strip()
        else:
            if actual:
                chunks.append(actual)
            actual = oracion
        if len(chunks) >= max_chunks:
            break
    if actual and len(chunks) < max_chunks:
        chunks.append(actual)
    return chunks[:max_chunks] or [texto[:max_chars]]


def clasificar_letra(
        classifier,
        letra: str,
        threshold: float,
        top_k_min: int,
        max_labels: int,
        max_chars: int,
        max_chunks: int) -> tuple[list[dict], list[dict]]:
    labels = [categoria["prompt"] for categoria in GOEMOTIONS_CANCIONES]
    prompt_to_categoria = {categoria["prompt"]: categoria for categoria in GOEMOTIONS_CANCIONES}
    acumulados = defaultdict(float)
    chunks = fragmentar_letra(letra, max_chars=max_chars, max_chunks=max_chunks)

    for chunk in chunks:
        resultado = classifier(
            chunk,
            candidate_labels=labels,
            hypothesis_template="{}",
            multi_label=True,
        )
        for label, score in zip(resultado["labels"], resultado["scores"]):
            acumulados[label] += float(score)

    filas_score = []
    for prompt, score in acumulados.items():
        categoria = prompt_to_categoria[prompt]
        score_transformer = score / len(chunks)
        score_lexico = calcular_score_lexico(letra, categoria["clave"])
        score_final = min(1.0, score_transformer + (LEXICO_AUXILIAR_PESO * score_lexico))

        # El lexico ya no decide la emocion; solo amortigua etiquetas muy
        # generales cuando el transformer tambien viene con baja confianza.
        if categoria["clave"] in {"curiosidad_busqueda", "asco_repulsion", "sorpresa_asombro"}:
            if score_transformer < 0.35 and score_lexico == 0:
                score_final *= 0.75
        if categoria["clave"] == "neutral_contemplativa":
            if score_transformer < 0.30 and score_lexico == 0:
                score_final *= 0.85
        if score_lexico >= 0.75:
            score_final = min(1.0, score_final + (0.04 * score_lexico))

        filas_score.append((prompt, score_final, score_transformer, score_lexico))

    ranking = sorted(filas_score, key=lambda item: item[1], reverse=True)

    seleccionadas = [
        {
            **prompt_to_categoria[prompt],
            "score": round(score_final, 6),
            "score_transformer": round(score_transformer, 6),
            "score_lexico": round(score_lexico, 6),
            "chunks_usados": len(chunks),
        }
        for prompt, score_final, score_transformer, score_lexico in ranking
        if score_final >= threshold
    ]
    if len(seleccionadas) < top_k_min:
        seleccionadas = [
            {
                **prompt_to_categoria[prompt],
                "score": round(score_final, 6),
                "score_transformer": round(score_transformer, 6),
                "score_lexico": round(score_lexico, 6),
                "chunks_usados": len(chunks),
            }
            for prompt, score_final, score_transformer, score_lexico in ranking[:top_k_min]
        ]
    if max_labels and len(seleccionadas) > max_labels:
        seleccionadas = seleccionadas[:max_labels]

    ranking_completo = [
        {
            **prompt_to_categoria[prompt],
            "score": round(score_final, 6),
            "score_transformer": round(score_transformer, 6),
            "score_lexico": round(score_lexico, 6),
            "chunks_usados": len(chunks),
        }
        for prompt, score_final, score_transformer, score_lexico in ranking
    ]
    return seleccionadas, ranking_completo


def clasificar_canciones(
        canciones: list[dict],
        out_dir: str,
        modelo: str,
        threshold: float,
        top_k_min: int,
        max_labels: int,
        max_chars: int,
        max_chunks: int) -> tuple[list[dict], list[dict]]:
    classifier = cargar_zero_shot(modelo)
    clasificaciones = []
    top1 = []

    total = len(canciones)
    for pos, cancion in enumerate(canciones, start=1):
        print(f"[{pos}/{total}] {cancion['artista']} - {cancion['nombre']}", flush=True)
        seleccionadas, ranking = clasificar_letra(
            classifier,
            cancion["letra"],
            threshold=threshold,
            top_k_min=top_k_min,
            max_labels=max_labels,
            max_chars=max_chars,
            max_chunks=max_chunks,
        )
        mejor = ranking[0]
        top1.append({**cancion, **mejor})
        for etiqueta in seleccionadas:
            clasificaciones.append({**cancion, **etiqueta})

    guardar_clasificacion(clasificaciones, top1, out_dir)
    return clasificaciones, top1


def ruta_categoria(out_dir: str, categoria: dict) -> str:
    return os.path.join(
        out_dir,
        f"goemotion_{categoria['id']:02d}_{limpiar_nombre_archivo(categoria['clave'])}.txt",
    )


def guardar_clasificacion(clasificaciones: list[dict], top1: list[dict], out_dir: str):
    campos = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz", "id", "clave", "label",
        "grupo", "score", "score_transformer", "score_lexico", "chunks_usados",
    ]
    ruta_csv = os.path.join(out_dir, "clasificacion_goemotions_transformer.csv")
    with open(ruta_csv, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in clasificaciones:
            writer.writerow({campo: fila.get(campo, "") for campo in campos})

    ruta_top1 = os.path.join(out_dir, "top1_goemotions_transformer.csv")
    with open(ruta_top1, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in top1:
            writer.writerow({campo: fila.get(campo, "") for campo in campos})

    por_categoria = defaultdict(list)
    for fila in clasificaciones:
        por_categoria[fila["id"]].append(fila)

    for categoria in GOEMOTIONS_CANCIONES:
        with open(ruta_categoria(out_dir, categoria), "w", encoding="utf-8") as archivo:
            for idx, fila in enumerate(por_categoria.get(categoria["id"], []), start=1):
                letra = re.sub(r"\s+", " ", fila["letra"]).strip()
                archivo.write(
                    f"{idx} | {fila['nombre']} - {fila['artista']} | "
                    f"score={fila['score']} | {letra}\n"
                )

    conteo_multi = Counter(fila["id"] for fila in clasificaciones)
    conteo_top1 = Counter(fila["id"] for fila in top1)
    resumen_path = os.path.join(out_dir, "resumen_goemotions_transformer.csv")
    with open(resumen_path, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.writer(archivo)
        writer.writerow([
            "id", "clave", "label", "grupo",
            "cantidad_multietiqueta", "cantidad_top1",
        ])
        for categoria in GOEMOTIONS_CANCIONES:
            writer.writerow([
                categoria["id"],
                categoria["clave"],
                categoria["label"],
                categoria["grupo"],
                conteo_multi.get(categoria["id"], 0),
                conteo_top1.get(categoria["id"], 0),
            ])


def imprimir_resumen(clasificaciones: list[dict], top1: list[dict], out_dir: str):
    conteo_multi = Counter(fila["id"] for fila in clasificaciones)
    conteo_top1 = Counter(fila["id"] for fila in top1)
    print("\n=== Resumen GoEmotions adaptado ===")
    print(f"Canciones clasificadas: {len(top1)}")
    print(f"Asignaciones multietiqueta: {len(clasificaciones)}")
    for categoria in GOEMOTIONS_CANCIONES:
        print(
            f"{categoria['id']:02d} - {categoria['clave']}: "
            f"multi={conteo_multi.get(categoria['id'], 0)} | "
            f"top1={conteo_top1.get(categoria['id'], 0)}"
        )
    print(f"\nCarpeta generada: {out_dir}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasificacion fina multietiqueta de letras crudas."
    )
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--solo-extraer", action="store_true")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--modelo", default=MODELO_ZERO_SHOT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Score minimo para incluir una etiqueta. Default: 0.45.",
    )
    parser.add_argument(
        "--top-k-min",
        type=int,
        default=3,
        help="Minimo de etiquetas por cancion aunque no superen threshold. Default: 3.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=5,
        help="Maximo de etiquetas por cancion. Default: 5.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Tamano maximo de cada fragmento de letra. Default: 1200.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=3,
        help="Maximo de fragmentos por letra. Default: 3.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)

    print("=== Prueba aislada: GoEmotions adaptado + zero-shot ===")
    print("No modifica main.py, canciones_*.txt, matrices ni la base.\n")

    canciones = cargar_letras_filtradas(limite=args.limite)
    preparar_salida(out_dir)
    guardar_letras_crudas(canciones, out_dir)
    print(f"Letras filtradas extraidas: {len(canciones)}")
    print(f"Salida cruda: {os.path.join(out_dir, 'letras_filtradas_crudas.txt')}")

    if args.solo_extraer:
        print("\nModo --solo-extraer activo. No se cargo zero-shot.")
        return

    clasificaciones, top1 = clasificar_canciones(
        canciones,
        out_dir=out_dir,
        modelo=args.modelo,
        threshold=args.threshold,
        top_k_min=args.top_k_min,
        max_labels=args.max_labels,
        max_chars=args.max_chars,
        max_chunks=args.max_chunks,
    )
    imprimir_resumen(clasificaciones, top1, out_dir)


if __name__ == "__main__":
    main()

