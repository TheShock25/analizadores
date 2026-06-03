#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ekman_transformer_prueba.py
===========================
Experimento aislado para clasificar letras crudas con un Transformer.

Este archivo NO se conecta con main.py ni modifica el pipeline del proyecto.
Solo lee moodtracker.db, toma las canciones que ya pasaron los filtros para
matriz (usar_en_matriz = 1), limpia ruido basico de Genius y clasifica cada
letra sin lematizar en las 6 emociones basicas de Ekman:

    0 alegria
    1 tristeza
    2 miedo
    3 ira
    4 asco
    5 sorpresa

Salidas:
    prueba_transformers_ekman/
        letras_filtradas_crudas.txt
        letras_filtradas_crudas_meta.csv
        ekman_0_alegria.txt
        ekman_1_tristeza.txt
        ekman_2_miedo.txt
        ekman_3_ira.txt
        ekman_4_asco.txt
        ekman_5_sorpresa.txt
        clasificacion_ekman_transformer.csv
        resumen_ekman_transformer.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sqlite3
import sys
import textwrap
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable


BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "moodtracker.db")
OUT_DIR = os.path.join(BASE_DIR, "prueba_transformers_ekman")
DESCARTADAS_META = os.path.join(BASE_DIR, "canciones_descartadas_meta.txt")

MODELO_EMOCIONES = "pysentimiento/robertuito-emotion-analysis"

EKMAN = {
    0: "alegria",
    1: "tristeza",
    2: "miedo",
    3: "ira",
    4: "asco",
    5: "sorpresa",
}

ALIAS_EKMAN = {
    "alegria": 0,
    "alegría": 0,
    "joy": 0,
    "happiness": 0,
    "tristeza": 1,
    "sadness": 1,
    "miedo": 2,
    "fear": 2,
    "ira": 3,
    "anger": 3,
    "angry": 3,
    "asco": 4,
    "disgust": 4,
    "sorpresa": 5,
    "surprise": 5,
}

GENIUS_NOISE = re.compile(
    r"(\d+\s*Contributors?.*?Lyrics|"
    r"\[.*?\]|"
    r"You might also like|"
    r"Embed$)",
    re.IGNORECASE | re.MULTILINE,
)


def _normalizar_label(label: str) -> str:
    label = str(label).strip().lower()
    label = unicodedata.normalize("NFKD", label)
    label = label.encode("ASCII", "ignore").decode("utf-8")
    return label


def reparar_mojibake(texto: str) -> str:
    """
    Repara texto UTF-8 que quedo interpretado como latin-1/cp1252.
    No modifica la base de datos; solo corrige la salida experimental.
    """
    if not isinstance(texto, str):
        return texto
    if "Ã" not in texto and "Â" not in texto:
        return texto
    for encoding in ("latin1", "cp1252"):
        try:
            reparado = texto.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if reparado.count("Ã") + reparado.count("Â") < texto.count("Ã") + texto.count("Â"):
            return reparado
    return texto


def limpiar_letra_cruda(letra: str) -> str:
    """Quita ruido de Genius, conserva letra legible y no lematiza."""
    letra = reparar_mojibake(letra or "")
    letra = GENIUS_NOISE.sub(" ", letra or "")
    letra = re.sub(r"^\s*\d+\s*$", "", letra, flags=re.MULTILINE)
    letra = re.sub(r"\r\n?", "\n", letra)
    letra = re.sub(r"\n{3,}", "\n\n", letra)
    letra = re.sub(r"[ \t]{2,}", " ", letra)
    return letra.strip()


def _leer_descartadas() -> set[str]:
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
    """
    Lee letras crudas de moodtracker.db.

    Criterios:
      - letra no vacia
      - COALESCE(usar_en_matriz, 1) = 1
      - sin duplicar spotify_track_id
      - excluye canciones ya marcadas como descartadas por ruido
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"No se encontro la base: {DB_PATH}")

    descartadas = _leer_descartadas()
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

    with open(ruta_txt, "w", encoding="utf-8") as txt:
        for cancion in canciones:
            letra_una_linea = re.sub(r"\s+", " ", cancion["letra"]).strip()
            txt.write(f"{cancion['indice']} | {letra_una_linea}\n")

    campos = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz",
    ]
    with open(ruta_meta, "w", encoding="utf-8", newline="") as meta:
        writer = csv.DictWriter(meta, fieldnames=campos)
        writer.writeheader()
        for cancion in canciones:
            writer.writerow({campo: cancion.get(campo, "") for campo in campos})


def _cargar_transformer(modelo_nombre: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Instala primero:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install torch transformers\n"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Transformer] Modelo: {modelo_nombre}")
    print(f"[Transformer] Dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    model = AutoModelForSequenceClassification.from_pretrained(modelo_nombre)
    model.to(device)
    model.eval()

    id2label = {
        int(idx): str(label)
        for idx, label in model.config.id2label.items()
    }
    print(f"[Transformer] Etiquetas del modelo: {id2label}")
    return torch, tokenizer, model, device, id2label


def _mapear_a_ekman(label: str) -> int | None:
    return ALIAS_EKMAN.get(_normalizar_label(label))


def _max_length_modelo(tokenizer, model) -> int:
    valores = []
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and tokenizer_max < 100_000:
        valores.append(tokenizer_max)
    model_max = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_max, int):
        # RoBERTa reserva posiciones para tokens especiales.
        valores.append(max(8, model_max - 2))
    return min(valores) if valores else 128


def _crear_chunks_tokenizados(texto: str, tokenizer, model, max_chunks: int):
    max_length = _max_length_modelo(tokenizer, model)
    token_ids = tokenizer.encode(texto, add_special_tokens=False, verbose=False)
    if not token_ids:
        token_ids = tokenizer.encode(" ", add_special_tokens=False)

    especiales = tokenizer.num_special_tokens_to_add(pair=False)
    chunk_size = max(8, max_length - especiales)
    chunks = [
        token_ids[inicio:inicio + chunk_size]
        for inicio in range(0, len(token_ids), chunk_size)
    ]
    if max_chunks and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    tokenizados = []
    for chunk in chunks:
        texto_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        tokenizados.append(
            tokenizer(
                texto_chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
        )
    return tokenizados


def predecir_ekman(
        texto: str,
        torch,
        tokenizer,
        model,
        device,
        id2label: dict[int, str],
        max_chunks: int) -> dict:
    """
    Predice una de las 6 emociones de Ekman.

    Si el modelo devuelve etiquetas extra como neutral/others, se ignoran para
    elegir la emocion final, pero se conservan en el CSV como etiqueta_modelo.
    """
    chunks = _crear_chunks_tokenizados(texto, tokenizer, model, max_chunks=max_chunks)

    probs_acumuladas = None
    with torch.no_grad():
        for inputs in chunks:
            inputs = {clave: valor.to(device) for clave, valor in inputs.items()}
            outputs = model(**inputs)
            probs_chunk = torch.softmax(outputs.logits, dim=1).detach().cpu()[0]
            probs_acumuladas = (
                probs_chunk if probs_acumuladas is None
                else probs_acumuladas + probs_chunk
            )

    probs = (probs_acumuladas / len(chunks)).tolist()

    modelo_idx = max(range(len(probs)), key=lambda idx: probs[idx])
    modelo_label = id2label.get(modelo_idx, str(modelo_idx))
    modelo_confianza = float(probs[modelo_idx])

    candidatos_ekman = []
    for idx, prob in enumerate(probs):
        label = id2label.get(idx, str(idx))
        ekman_id = _mapear_a_ekman(label)
        if ekman_id is not None:
            candidatos_ekman.append((ekman_id, label, float(prob)))

    if candidatos_ekman:
        ekman_id, ekman_label_modelo, ekman_confianza = max(
            candidatos_ekman,
            key=lambda item: item[2],
        )
    else:
        ekman_id = _mapear_a_ekman(modelo_label)
        if ekman_id is None:
            ekman_id = 5
        ekman_label_modelo = modelo_label
        ekman_confianza = modelo_confianza

    return {
        "emocion_id": ekman_id,
        "emocion": EKMAN[ekman_id],
        "confianza_ekman": round(ekman_confianza, 6),
        "etiqueta_modelo": modelo_label,
        "confianza_modelo": round(modelo_confianza, 6),
        "etiqueta_ekman_modelo": ekman_label_modelo,
        "chunks_usados": len(chunks),
    }


def clasificar_canciones(
        canciones: list[dict],
        out_dir: str,
        modelo_nombre: str,
        max_chunks: int) -> list[dict]:
    torch, tokenizer, model, device, id2label = _cargar_transformer(modelo_nombre)

    resultados = []
    total = len(canciones)
    for pos, cancion in enumerate(canciones, start=1):
        print(
            f"[{pos}/{total}] {cancion['artista']} - {cancion['nombre']}",
            flush=True,
        )
        pred = predecir_ekman(
            cancion["letra"],
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            id2label=id2label,
            max_chunks=max_chunks,
        )
        resultados.append({**cancion, **pred})

    guardar_clasificacion(resultados, out_dir)
    return resultados


def _nombre_archivo_emocion(emocion_id: int) -> str:
    return f"ekman_{emocion_id}_{EKMAN[emocion_id]}.txt"


def guardar_clasificacion(resultados: list[dict], out_dir: str):
    campos = [
        "indice", "spotify_track_id", "nombre", "artista", "album",
        "fecha_reproduccion", "relevancia_letra", "patron_escucha",
        "motivo_matriz", "score_matriz", "emocion_id", "emocion",
        "confianza_ekman", "etiqueta_modelo", "confianza_modelo",
        "etiqueta_ekman_modelo", "chunks_usados",
    ]

    ruta_csv = os.path.join(out_dir, "clasificacion_ekman_transformer.csv")
    with open(ruta_csv, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in resultados:
            writer.writerow({campo: fila.get(campo, "") for campo in campos})

    por_emocion = defaultdict(list)
    for fila in resultados:
        por_emocion[fila["emocion_id"]].append(fila)

    for emocion_id, emocion in EKMAN.items():
        ruta = os.path.join(out_dir, _nombre_archivo_emocion(emocion_id))
        with open(ruta, "w", encoding="utf-8") as archivo:
            for idx, fila in enumerate(por_emocion.get(emocion_id, []), start=1):
                letra_una_linea = re.sub(r"\s+", " ", fila["letra"]).strip()
                archivo.write(
                    f"{idx} | {fila['nombre']} - {fila['artista']} | "
                    f"confianza={fila['confianza_ekman']} | {letra_una_linea}\n"
                )

    conteo = Counter(fila["emocion_id"] for fila in resultados)
    resumen_path = os.path.join(out_dir, "resumen_ekman_transformer.csv")
    with open(resumen_path, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["emocion_id", "emocion", "cantidad"])
        for emocion_id, emocion in EKMAN.items():
            writer.writerow([emocion_id, emocion, conteo.get(emocion_id, 0)])


def imprimir_resumen(resultados: list[dict], out_dir: str):
    conteo = Counter(fila["emocion_id"] for fila in resultados)
    print("\n=== Resumen Ekman Transformer ===")
    print(f"Total clasificado: {len(resultados)}")
    for emocion_id, emocion in EKMAN.items():
        print(f"{emocion_id} - {emocion}: {conteo.get(emocion_id, 0)}")
    print(f"\nCarpeta generada: {out_dir}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica letras crudas filtradas con Transformer Ekman."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Opcional: clasifica solo las primeras N letras para prueba rapida.",
    )
    parser.add_argument(
        "--solo-extraer",
        action="store_true",
        help="Solo genera letras crudas/meta, sin cargar Transformer.",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help="Carpeta de salida. Por defecto: prueba_transformers_ekman",
    )
    parser.add_argument(
        "--modelo",
        default=MODELO_EMOCIONES,
        help=f"Modelo Hugging Face a usar. Por defecto: {MODELO_EMOCIONES}",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=8,
        help="Maximo de fragmentos por letra para promediar probabilidades. Default: 8.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)

    print("=== Prueba aislada: letras crudas + Transformer Ekman ===")
    print("No modifica main.py, canciones_*.txt, matrices ni la base de datos.\n")

    canciones = cargar_letras_filtradas(limite=args.limite)
    preparar_salida(out_dir)
    guardar_letras_crudas(canciones, out_dir)

    print(f"Letras filtradas extraidas: {len(canciones)}")
    print(f"Salida cruda: {os.path.join(out_dir, 'letras_filtradas_crudas.txt')}")

    if args.solo_extraer:
        print("\nModo --solo-extraer activo. No se cargo el Transformer.")
        return

    if not canciones:
        print("No hay letras para clasificar.")
        return

    resultados = clasificar_canciones(canciones, out_dir, args.modelo, args.max_chunks)
    imprimir_resumen(resultados, out_dir)


if __name__ == "__main__":
    main()
