#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goemotions_jerarquico_prueba.py
================================
Experimento aislado para clasificar una letra con arquitectura jerarquica:

    frase -> estrofa -> cancion

No modifica main.py, moodtracker.db, matrices ni las salidas anteriores.

Entradas:
    prueba_cancion_24h.txt por defecto, o --archivo-txt.

Salidas:
    prueba_transformers_jerarquico_txt/
        frases_clasificadas.csv
        estrofas_resumen.csv
        cancion_resumen.csv
        reporte_jerarquico.txt
        parametros.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from typing import Iterable

from goemotions_frases_24h_prueba import (
    BASE_DIR,
    CATEGORIAS_FRASES,
    MARCADORES_AFECTO,
    MARCADORES_AMOR_AFIRMATIVO,
    MARCADORES_BLOQUEO_AFECTIVO,
    MARCADORES_CONFLICTO,
    MARCADORES_DESCONEXION_AFECTIVA,
    MARCADORES_DESEO,
    MARCADORES_DOLOR_INTENSO,
    MARCADORES_PERDIDA_AUSENCIA,
    MARCADORES_REFLEXION,
    MARCADORES_RELACION_DANADA,
    clasificar_segmento,
    es_relleno_vocal,
    limpiar_linea,
    normalizar_simple,
    tiene_marcador,
)
from goemotions_transformer_prueba import (
    MODELO_ZERO_SHOT,
    cargar_zero_shot,
    limpiar_letra_cruda,
)


OUT_DIR = os.path.join(BASE_DIR, "prueba_transformers_jerarquico_txt")
DEFAULT_TXT = os.path.join(BASE_DIR, "prueba_cancion_24h.txt")


def reparar_texto_mojibake(texto: str) -> str:
    if not isinstance(texto, str):
        return texto
    if "Ã" in texto:
        for encoding in ("latin1", "cp1252"):
            try:
                reparado = texto.encode(encoding).decode("utf-8")
                if reparado.count("Ã") < texto.count("Ã"):
                    return reparado
            except UnicodeError:
                continue
    if "Ã" not in texto and "Â" not in texto:
        return texto
    for encoding in ("latin1", "cp1252"):
        try:
            reparado = texto.encode(encoding).decode("utf-8")
            if reparado.count("Ã") + reparado.count("Â") < texto.count("Ã") + texto.count("Â"):
                return reparado
        except UnicodeError:
            continue
    return texto


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasificacion jerarquica frase->estrofa->cancion para una letra TXT."
    )
    parser.add_argument("--archivo-txt", default=DEFAULT_TXT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--modelo", default=MODELO_ZERO_SHOT)
    parser.add_argument(
        "--frases-por-estrofa",
        type=int,
        default=4,
        help="Agrupa la letra cada N frases si el TXT no trae estrofas. Recomendado: 4 o 5.",
    )
    parser.add_argument("--top-k-frase", type=int, default=5)
    parser.add_argument("--top-k-estrofa", type=int, default=7)
    parser.add_argument(
        "--peso-estrofa-completa",
        type=float,
        default=0.40,
        help="Peso del analisis directo de la estrofa completa.",
    )
    parser.add_argument(
        "--peso-promedio-frases",
        type=float,
        default=0.40,
        help="Peso del promedio de frases corregidas dentro de la estrofa.",
    )
    parser.add_argument(
        "--peso-contexto-estrofas",
        type=float,
        default=0.20,
        help="Peso del contexto estrofa anterior + actual + siguiente.",
    )
    return parser.parse_args(list(argv))


def preparar_salida(out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def leer_lineas_letra(path_txt: str) -> list[str]:
    path_abs = os.path.abspath(path_txt)
    if not os.path.exists(path_abs):
        raise FileNotFoundError(f"No se encontro el archivo: {path_abs}")

    with open(path_abs, "r", encoding="utf-8") as archivo:
        texto = limpiar_letra_cruda(archivo.read())
    texto = reparar_texto_mojibake(texto)

    lineas = []
    for linea in texto.splitlines():
        limpia = limpiar_linea(reparar_texto_mojibake(linea))
        if limpia:
            lineas.append(limpia)
    return lineas


def agrupar_en_estrofas(lineas: list[str], frases_por_estrofa: int) -> list[dict]:
    if frases_por_estrofa < 1:
        raise ValueError("--frases-por-estrofa debe ser mayor que cero")

    estrofas = []
    for inicio in range(0, len(lineas), frases_por_estrofa):
        frases = lineas[inicio:inicio + frases_por_estrofa]
        estrofas.append({
            "estrofa": len(estrofas) + 1,
            "frases": frases,
            "texto": " ".join(frases),
        })
    return estrofas


MARCADORES_CONTINUACION = [
    "y", "por", "porque", "mientras", "cuando", "donde",
    "que", "sin", "con", "pero", "aunque", "para",
]

CONECTORES_CONTRASTE = [
    "pero", "aunque", "sin embargo", "aun asi", "aun así",
    "en cambio", "mas no", "pero aqui", "pero aquí",
]


def frase_pide_contexto_cruzado(frase: str) -> bool:
    normalizada = normalizar_simple(frase)
    tokens = normalizada.split()
    if not tokens:
        return False
    return tokens[0] in MARCADORES_CONTINUACION or tokens[-1] in MARCADORES_CONTINUACION


def dividir_por_contraste(texto: str) -> tuple[str, str] | None:
    normalizado = normalizar_simple(texto)
    mejor = None
    for conector in CONECTORES_CONTRASTE:
        conector_norm = normalizar_simple(conector)
        patron = f" {conector_norm} "
        pos = normalizado.rfind(patron)
        if pos >= 0:
            mejor = (pos, conector_norm)
            break
    if not mejor:
        return None
    pos, conector = mejor
    antes = normalizado[:pos].strip()
    despues = normalizado[pos + len(conector) + 2:].strip()
    if not antes or not despues:
        return None
    return antes, despues


def hay_contraste_negado(texto: str) -> bool:
    partes = dividir_por_contraste(texto)
    if not partes:
        return False
    antes, despues = partes
    antes_negativo = hay_contexto_negativo_relacional(antes) or tiene_marcador(
        antes,
        ["no hay", "sin", "nunca", "dolor", "herida", "frio", "oscuridad", "soledad"],
    )
    despues_niega = tiene_marcador(
        despues,
        ["no es asi", "no es igual", "aqui no", "aqui es distinto", "es diferente", "no pasa"],
    )
    return antes_negativo and despues_niega


def es_tarareo_o_vocalizacion(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    if not normalizado:
        return True
    if es_relleno_vocal(texto):
        return True
    tokens = normalizado.split()
    if not tokens:
        return True
    vocales = {"ah", "oh", "uh", "eh", "mm", "mmm", "na", "la", "yeah", "hey", "ay"}
    tokens_vocales = sum(
        1 for token in tokens
        if token in vocales or (len(token) > 3 and len(set(token)) <= 2)
    )
    return len(tokens) <= 10 and (tokens_vocales / len(tokens)) >= 0.75


def canon_repeticion(texto: str) -> str:
    normalizado = normalizar_simple(texto)
    tokens = [
        token for token in normalizado.split()
        if token not in {"no", "nah", "yeah", "hey", "oh", "ah", "uh", "eh", "come", "on"}
    ]
    return " ".join(tokens)


def es_nombre_o_coro_corto(texto: str) -> bool:
    tokens_norm = normalizar_simple(texto).split()
    if not tokens_norm or len(tokens_norm) > 5:
        return False
    if tiene_verbo_probable(texto):
        return False

    tokens_raw = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+", texto)
    if not tokens_raw:
        return False
    capitalizados = sum(1 for token in tokens_raw if token[:1].isupper() and len(token) > 1)
    cortos_filler = sum(1 for token in tokens_norm if token in {"no", "nah", "yeah", "hey", "oh", "ah", "uh", "eh", "come", "on"})
    return capitalizados >= max(1, len(tokens_raw) - cortos_filler - 1) or cortos_filler / len(tokens_norm) >= 0.50


def es_coro_repetitivo_contextual(frase: str, frases_estrofa: list[str]) -> bool:
    canon = canon_repeticion(frase)
    if not canon:
        return True
    normalizado = normalizar_simple(frase)
    tokens = normalizado.split()
    if len(tokens) <= 6:
        conteo_token = Counter(token for token in tokens if len(token) > 1)
        if conteo_token and max(conteo_token.values()) >= 2:
            return True

    canones = [canon_repeticion(item) for item in frases_estrofa]
    repetida = sum(1 for item in canones if item and item == canon) >= 2
    return repetida or es_nombre_o_coro_corto(frase)


def motivo_tematico_con_carga(texto: str) -> bool:
    if es_tarareo_o_vocalizacion(texto):
        return False
    normalizado = normalizar_simple(texto)
    if re.search(r"\b(inside|interior|profund\w*|dentro)\b", normalizado):
        return True
    if (
        detectar_paradoja_placer_danio(texto) != "no_paradoja"
        or detectar_conflicto_interno(texto)
        or detectar_compulsion_impulso(texto)
        or detectar_advertencia_afectiva([texto])
        or detectar_autoconciencia_problematica([texto])
        or detectar_no_compromiso_afectivo([texto])
    ):
        return True
    if detectar_modo_protesta_social(texto) != "no_protesta":
        return True
    if detectar_pregunta_retorica(texto) or detectar_imperativo_liberacion(texto):
        return True
    modo_devastacion = detectar_modo_devastacion_interna(texto)
    if modo_devastacion != "no_devastacion":
        return True
    return detectar_orientacion_corporal(texto) in {
        "sensual_ambiguo", "sensual_positivo", "sensual_conflictivo", "sensual_oscuro", "corporal_negativo",
    }


def tiene_verbo_probable(texto: str) -> bool:
    tokens = normalizar_simple(texto).split()
    if not tokens:
        return False
    auxiliares = {
        "soy", "eres", "es", "somos", "son", "estoy", "estas", "esta",
        "estamos", "estan", "fui", "fue", "era", "eras", "seras",
        "tengo", "tienes", "tiene", "hice", "haces", "hace", "quiero",
        "quieres", "quiere", "amo", "amas", "ama", "puedo", "puedes",
        "puede", "debo", "debes", "debe", "voy", "vas", "va",
    }
    if any(token in auxiliares for token in tokens):
        return True
    return any(
        re.search(r"(ar|er|ir|ando|iendo|ado|ido|aba|ia|are|ere|ire|aste|iste|amos|emos|imos|an|en)$", token)
        for token in tokens
        if len(token) > 4
    )


def tipo_y_peso_frase(texto: str) -> tuple[str, float]:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    if es_tarareo_o_vocalizacion(texto):
        return "tarareo", 0.0
    if not tokens:
        return "sin_contenido", 0.0
    if len(tokens) <= 3:
        return "soporte_corta", 0.30
    if len(tokens) <= 5 and (not tiene_verbo_probable(texto) or tokens[0] in MARCADORES_CONTINUACION):
        return "soporte", 0.45
    if tokens[0] in MARCADORES_CONTINUACION or tokens[-1] in MARCADORES_CONTINUACION:
        return "fragmento_incompleto", 0.60
    return "nucleo", 1.0


def texto_con_contenido(frases: list[str]) -> str:
    utiles = [
        frase for frase in frases
        if tipo_y_peso_frase(frase)[1] > 0
        and (not es_nombre_o_coro_corto(frase) or motivo_tematico_con_carga(frase))
    ]
    return " ".join(utiles)


def contexto_frase(estrofas: list[dict], idx_estrofa: int, idx_frase: int) -> str:
    estrofa = estrofas[idx_estrofa]
    frases = estrofa["frases"]
    partes = []

    if idx_frase == 0:
        if idx_estrofa > 0 and frase_pide_contexto_cruzado(frases[idx_frase]):
            partes.append(estrofas[idx_estrofa - 1]["frases"][-1])
        posiciones = [idx_frase, idx_frase + 1, idx_frase + 2]
    elif idx_frase == len(frases) - 1:
        posiciones = [idx_frase - 1, idx_frase]
    else:
        posiciones = [idx_frase - 1, idx_frase, idx_frase + 1]

    for pos in posiciones:
        if 0 <= pos < len(frases):
            partes.append(frases[pos])

    if (
        idx_frase == len(frases) - 1
        and idx_estrofa + 1 < len(estrofas)
        and frase_pide_contexto_cruzado(frases[idx_frase])
    ):
        partes.append(estrofas[idx_estrofa + 1]["frases"][0])

    return " ".join(partes)


def contexto_estrofa(estrofas: list[dict], idx_estrofa: int) -> str:
    partes = []
    for pos in (idx_estrofa - 1, idx_estrofa, idx_estrofa + 1):
        if 0 <= pos < len(estrofas):
            partes.append(estrofas[pos]["texto"])
    return " ".join(partes)


def contexto_estrofa_contenido(estrofas: list[dict], idx_estrofa: int) -> str:
    partes = []
    for pos in (idx_estrofa - 1, idx_estrofa, idx_estrofa + 1):
        if 0 <= pos < len(estrofas):
            partes.append(texto_con_contenido(estrofas[pos]["frases"]))
    return " ".join(parte for parte in partes if parte.strip())


def vector_desde_etiquetas(etiquetas: list[dict], normalizar: bool = True) -> dict[str, float]:
    vector = defaultdict(float)
    for etiqueta in etiquetas:
        vector[etiqueta["clave"]] += float(etiqueta["score"])

    if normalizar:
        total = sum(vector.values())
        if total > 0:
            for clave in list(vector.keys()):
                vector[clave] = vector[clave] / total
    return dict(vector)


def combinar_vectores(vectores: list[dict[str, float]], pesos: list[float] | None = None) -> dict[str, float]:
    combinado = defaultdict(float)
    if not vectores:
        return {}
    if pesos is None:
        pesos = [1.0] * len(vectores)

    total_peso = sum(pesos) or 1.0
    for vector, peso in zip(vectores, pesos):
        for clave, valor in vector.items():
            combinado[clave] += valor * peso

    for clave in list(combinado.keys()):
        combinado[clave] = combinado[clave] / total_peso
    return dict(combinado)


def normalizar_vector(vector: dict[str, float]) -> dict[str, float]:
    total = sum(vector.values())
    if total <= 0:
        return vector
    return {clave: valor / total for clave, valor in vector.items()}


def top_vector(vector: dict[str, float], n: int = 5) -> list[tuple[str, float]]:
    return sorted(vector.items(), key=lambda item: item[1], reverse=True)[:n]


def vector_a_texto(vector: dict[str, float], n: int = 5) -> str:
    return "; ".join(f"{clave}:{valor * 100:.2f}%" for clave, valor in top_vector(vector, n))


def clave_a_label() -> dict[str, str]:
    return {cat["clave"]: cat["label"] for cat in CATEGORIAS_FRASES}


def grupo_emocional(clave: str) -> str:
    if clave in {
        "amor", "alegria", "admiracion_aprecio", "cuidado_carino",
        "gratitud", "aprobacion_validacion", "entusiasmo_emocion",
        "optimismo_esperanza", "alivio_liberacion",
    }:
        return "positivo"
    if clave in {
        "decepcion_desamor", "tristeza", "duelo_pena", "molestia_fastidio",
        "ira", "miedo_angustia", "remordimiento_culpa", "verguenza_vulnerabilidad",
        "asco_repulsion", "desaprobacion_rechazo",
    }:
        return "negativo"
    return "ambiguo"


def hay_contradiccion_frases(vectores: list[dict[str, float]], pesos: list[float]) -> bool:
    grupos = set()
    for vector, peso in zip(vectores, pesos):
        if not vector or peso <= 0:
            continue
        clave = top_vector(vector, 1)[0][0]
        grupo = grupo_emocional(clave)
        if grupo != "ambiguo":
            grupos.add(grupo)
    return "positivo" in grupos and "negativo" in grupos


POSITIVAS_DIRECTAS = {
    "amor", "alegria", "admiracion_aprecio", "gratitud",
    "cuidado_carino", "aprobacion_validacion", "entusiasmo_emocion",
    "optimismo_esperanza", "redencion_renacer", "alivio_liberacion",
}

NEGATIVAS_CONTEXTO = {
    "decepcion_desamor", "molestia_fastidio", "desaprobacion_rechazo",
    "tristeza", "duelo_pena", "miedo_angustia", "remordimiento_culpa",
    "confusion",
}

ROMANCE_APEGO = {
    "amor", "deseo_anhelo", "cuidado_carino", "gratitud",
    "admiracion_aprecio", "aprobacion_validacion",
}

NEGATIVAS_DOLOR = {
    "decepcion_desamor", "tristeza", "duelo_pena", "miedo_angustia",
    "molestia_fastidio", "ira", "remordimiento_culpa",
    "desaprobacion_rechazo", "asco_repulsion",
}

TRANSICION_RESOLUCION = {
    "realizacion_darse_cuenta", "alivio_liberacion",
    "optimismo_esperanza", "aceptacion_desapego", "redencion_renacer",
}

PETICION_VULNERABLE = {
    "cuidado_carino", "aprobacion_validacion", "deseo_anhelo",
    "verguenza_vulnerabilidad", "amor",
}

SENSUAL_MODULADORES = {
    "deseo_anhelo", "amor", "entusiasmo_emocion",
    "nerviosismo_ansiedad", "admiracion_aprecio",
    "cuidado_carino", "diversion_ironia",
}

COMODINES_NO_SENSUALES = {
    "realizacion_darse_cuenta", "gratitud",
    "aprobacion_validacion", "molestia_fastidio",
    "aceptacion_desapego",
}

COMODINES_POSITIVOS = {
    "aprobacion_validacion", "gratitud", "admiracion_aprecio",
    "amor", "alegria", "cuidado_carino",
}

DEVASTACION_MODULADORES = {
    "miedo_angustia", "nerviosismo_ansiedad", "remordimiento_culpa",
    "verguenza_vulnerabilidad", "tristeza", "duelo_pena",
    "desaprobacion_rechazo", "molestia_fastidio", "decepcion_desamor",
}

RESILIENCIA_MODULADORES = {
    "redencion_renacer", "optimismo_esperanza", "realizacion_darse_cuenta",
    "aceptacion_desapego", "nerviosismo_ansiedad", "remordimiento_culpa",
    "cuidado_carino", "deseo_anhelo",
}

DOLOR_AMOROSO = {
    "decepcion_desamor", "tristeza", "duelo_pena",
    "molestia_fastidio", "remordimiento_culpa", "nerviosismo_ansiedad",
}

PROTESTA_MODULADORES = {
    "desaprobacion_rechazo", "molestia_fastidio",
    "realizacion_darse_cuenta", "redencion_renacer",
    "alivio_liberacion", "curiosidad_busqueda",
    "nerviosismo_ansiedad", "orgullo_autovaloracion",
}

PARADOJA_MODULADORES = {
    "confusion", "nerviosismo_ansiedad", "remordimiento_culpa",
    "molestia_fastidio", "verguenza_vulnerabilidad",
    "desaprobacion_rechazo", "deseo_anhelo",
}

MARCADORES_CRITICA_RELACIONAL = [
    "mal", "confundi", "confundiste", "confundir", "rival",
    "derechos", "derecho", "acab", "termin", "contra", "opuesto",
    "culpa", "reproche", "discusion", "pelea", "problema",
    "fallo", "fallaste", "dano", "danaste", "herida", "heriste",
]

MARCADORES_EMOCION_POSITIVA_LEXICAL = [
    "amor", "amores", "beso", "besos", "cuerpo", "anhelo",
    "corazon", "ojos", "caricia", "caricias", "querer",
    "quiero", "deseo", "esperanza", "ilusion",
]

MARCADORES_INVERSION_EMOCIONAL = [
    "disminuyendo", "disminuir", "disminuye", "apagando", "apagar",
    "apagado", "muriendo", "muerto", "perdido", "perdi",
    "amor roto", "amor rota", "alma rota", "corazon roto",
    "relacion rota", "acabado", "acabo", "termino", "distante", "ausente",
    "sin entrega", "sin amor", "sin querer", "sin esperanza",
    "sin ilusion", "nada", "no era asi", "ya no",
]

MARCADORES_METAFORA_NEGATIVA = [
    "hielo", "frio", "fria", "congelo", "congelar", "congelado",
    "piedra", "desierto", "vacio", "vacia", "silencio", "distancia",
    "oscuridad", "sombra", "herida", "ceniza", "alma piedra",
]

MARCADORES_SOSPECHA_AMOROSA = [
    "otros amores", "otro amor", "otra mujer", "otro hombre",
    "pude ver", "vi en tus ojos", "mientras yo callaba",
    "callaba", "te vi", "te descubri",
]

MARCADORES_AUTOAFIRMACION_DESAPEGO = [
    "ya me canse", "me canse", "cansado", "cansada", "cansancio",
    "ya me canse de callar", "no eres lo unico", "no seras lo unico",
    "de ahora en adelante", "se paga", "te dejo", "me alejo",
]

MARCADORES_AMOR_CORRESPONDIDO = [
    "me enamore", "enamore", "te conoci", "gran amor",
    "para los dos", "me querias", "me quieres", "querias igual",
    "quieres igual", "te amo", "me amas", "nos amamos",
    "todo mi carino", "mi carino", "carino es para ti",
    "significas para mi", "eres para mi", "soy para ti",
]

MARCADORES_IMAGEN_CALIDA_ROMANTICA = [
    "brilla", "brillan", "brillaba", "brillaban", "primavera",
    "sonrisa", "luz", "ternura", "dulzura", "feliz",
    "alegria", "contento", "contenta",
]

MARCADORES_HALAGO_ROMANTICO = [
    "primorosa", "primoroso", "preciosa", "precioso", "hermosa",
    "hermoso", "linda", "lindo", "bella", "bello", "adorada",
    "adorado", "querida", "querido", "mi amor", "mi vida",
    "mi existir", "mi corazon",
]

MARCADORES_POSESION_AFECTIVA = [
    "mia", "mio", "para mi", "para ti", "para los dos", "eres tu",
    "reina mia", "mi amor", "mi vida", "mi existir", "mi carino",
    "todo mi carino", "tenerte para mi", "para mi nomas",
]

MARCADORES_METAFORA_CORPORAL_ROMANTICA = [
    "se me sale el corazon", "me late el corazon", "me robas el aliento",
    "me roba el aliento", "me quitas el aliento", "me tiemblan las piernas",
    "me vuelves loco", "me vuelves loca", "se acelera mi corazon",
]

MARCADORES_FRASE_SOPORTE_ROMANTICA = [
    "si me miras", "cuando te veo", "cuando me miras", "desde aquel dia",
    "desde aquel instante", "en tus ojos", "para mi", "para ti",
    "para los dos", "igual", "eres tu", "reina mia",
]

MARCADORES_REPROCHE_IMPERATIVO = [
    "juzgame", "condename", "odiame", "niegame", "no me ignores",
    "ignores", "ignorar", "creeme", "salvame", "defiendeme",
    "quiéreme", "quiereme", "culpable", "responsable",
]

MARCADORES_CAMBIO_NEGATIVO = [
    "antes", "cambiaste", "cambiar", "cambio", "distinto", "distinta",
    "diferente", "ya no eres", "ya no estas", "como antes",
]

PALABRAS_REPETICION_EMOCIONAL = {
    "amor", "querer", "quiereme", "odiame", "niegame", "salvame",
    "creeme", "perdon", "dolor", "adios",
}


def tiene_conflicto_real(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    if not tiene_marcador(normalizado, MARCADORES_CONFLICTO):
        return False

    hits = [marcador for marcador in MARCADORES_CONFLICTO if marcador in normalizado]
    tokens = set(normalizado.split())
    hits_reales = []
    for marcador in hits:
        if marcador == "ira" and marcador not in tokens:
            continue
        hits_reales.append(marcador)
    return bool(hits_reales)


def hay_contexto_negativo_relacional(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return (
        tiene_marcador(normalizado, MARCADORES_CRITICA_RELACIONAL)
        or tiene_marcador(normalizado, MARCADORES_RELACION_DANADA)
        or tiene_conflicto_real(normalizado)
        or tiene_marcador(normalizado, MARCADORES_DESCONEXION_AFECTIVA)
        or tiene_marcador(normalizado, MARCADORES_BLOQUEO_AFECTIVO)
        or tiene_marcador(normalizado, MARCADORES_PERDIDA_AUSENCIA)
        or tiene_marcador(normalizado, MARCADORES_DOLOR_INTENSO)
        or tiene_marcador(normalizado, MARCADORES_REPROCHE_IMPERATIVO)
        or hay_inversion_emocional(normalizado)
        or hay_metafora_negativa(normalizado)
        or hay_sospecha_amorosa(normalizado)
        or hay_autoafirmacion_desapego(normalizado)
    )


def contar_marcadores(texto: str, marcadores: list[str]) -> int:
    normalizado = normalizar_simple(texto)
    return sum(1 for marcador in marcadores if marcador in normalizado)


def intensidad_positiva_correspondida(texto: str) -> int:
    normalizado = normalizar_simple(texto)
    intensidad = contar_marcadores(normalizado, MARCADORES_AMOR_CORRESPONDIDO)
    intensidad += contar_marcadores(normalizado, MARCADORES_IMAGEN_CALIDA_ROMANTICA)
    intensidad += contar_marcadores(normalizado, MARCADORES_HALAGO_ROMANTICO)
    intensidad += contar_marcadores(normalizado, MARCADORES_POSESION_AFECTIVA)
    intensidad += contar_marcadores(normalizado, MARCADORES_METAFORA_CORPORAL_ROMANTICA)
    if "amor" in normalizado and tiene_marcador(normalizado, ["dos", "igual", "corazon", "enamore"]):
        intensidad += 1
    if tiene_marcador(normalizado, ["querias", "quieres", "me amas"]) and not tiene_marcador(normalizado, ["ya no", "no me"]):
        intensidad += 1
    return intensidad


def intensidad_negativa_dura(texto: str) -> int:
    normalizado = normalizar_simple(texto)
    intensidad = contar_marcadores(normalizado, MARCADORES_INVERSION_EMOCIONAL)
    intensidad += contar_marcadores(normalizado, MARCADORES_METAFORA_NEGATIVA)
    intensidad += contar_marcadores(normalizado, MARCADORES_SOSPECHA_AMOROSA)
    intensidad += contar_marcadores(normalizado, MARCADORES_REPROCHE_IMPERATIVO)
    intensidad += contar_marcadores(normalizado, MARCADORES_AUTOAFIRMACION_DESAPEGO)
    intensidad += contar_marcadores(normalizado, MARCADORES_CRITICA_RELACIONAL)
    return intensidad


def hay_contexto_negativo_duro(texto: str) -> bool:
    return intensidad_negativa_dura(texto) >= 2


def hay_contexto_positivo_correspondido(texto: str) -> bool:
    positivo = intensidad_positiva_correspondida(texto)
    negativo = intensidad_negativa_dura(texto)
    return positivo >= 2 and positivo >= negativo + 1


def hay_amor_mencionado_no_sentido(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    menciona_afecto = tiene_marcador(normalizado, MARCADORES_AFECTO)
    amor_afirmativo = tiene_marcador(normalizado, MARCADORES_AMOR_AFIRMATIVO)
    critica = hay_contexto_negativo_relacional(normalizado) or tiene_marcador(normalizado, MARCADORES_REFLEXION)
    positivo_protegido = hay_contexto_positivo_correspondido(normalizado) and not hay_contexto_negativo_duro(normalizado)
    return menciona_afecto and critica and not amor_afirmativo and not positivo_protegido


def hay_inversion_emocional(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return (
        tiene_marcador(normalizado, MARCADORES_EMOCION_POSITIVA_LEXICAL)
        and tiene_marcador(normalizado, MARCADORES_INVERSION_EMOCIONAL)
    )


def hay_metafora_negativa(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return tiene_marcador(normalizado, MARCADORES_METAFORA_NEGATIVA)


def hay_sospecha_amorosa(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return tiene_marcador(normalizado, MARCADORES_SOSPECHA_AMOROSA)


def hay_autoafirmacion_desapego(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return tiene_marcador(normalizado, MARCADORES_AUTOAFIRMACION_DESAPEGO)


def hay_dependencia_amorosa_sin_ruptura(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    afecto = tiene_marcador(normalizado, MARCADORES_AFECTO)
    dependencia = tiene_marcador(normalizado, ["sin", "no se vivir", "no puedo vivir", "me muero", "me falta"])
    ruptura = (
        tiene_marcador(normalizado, MARCADORES_RELACION_DANADA)
        or tiene_marcador(normalizado, MARCADORES_PERDIDA_AUSENCIA)
        or tiene_marcador(normalizado, MARCADORES_DESCONEXION_AFECTIVA)
        or tiene_marcador(normalizado, ["ya no", "adios", "abandono", "rechazo", "te vas", "me dejas"])
    )
    return afecto and dependencia and not ruptura


def hay_halago_romantico(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return (
        tiene_marcador(normalizado, MARCADORES_HALAGO_ROMANTICO)
        or tiene_marcador(normalizado, MARCADORES_POSESION_AFECTIVA)
    )


def hay_metafora_corporal_romantica(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return tiene_marcador(normalizado, MARCADORES_METAFORA_CORPORAL_ROMANTICA)


def es_frase_soporte_romantica(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    if len(tokens) > 5:
        return False
    return tiene_marcador(normalizado, MARCADORES_FRASE_SOPORTE_ROMANTICA)


def es_repeticion_emocional_generica(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    tokens = [token for token in normalizado.split() if len(token) > 2]
    if not tokens or len(tokens) > 8:
        return False

    conteo = Counter(tokens)
    repetidos = [token for token, cantidad in conteo.items() if cantidad >= 2]
    if not repetidos:
        return False

    unicos = set(tokens)
    return bool(unicos & PALABRAS_REPETICION_EMOCIONAL) and len(unicos) <= 3


def normalizar_pesos(*pesos: float) -> list[float]:
    limpios = [max(0.0, peso) for peso in pesos]
    total = sum(limpios)
    if total <= 0:
        return [1.0 / len(limpios)] * len(limpios)
    return [peso / total for peso in limpios]


def sumar_grupo(vector: dict[str, float], claves: set[str]) -> float:
    return sum(vector.get(clave, 0.0) for clave in claves)


def vector_por_indices(
        vectores: list[dict[str, float]],
        pesos: list[float],
        indices: list[int]) -> dict[str, float]:
    seleccion = []
    pesos_sel = []
    for indice in indices:
        if 0 <= indice < len(vectores) and vectores[indice] and pesos[indice] > 0:
            seleccion.append(vectores[indice])
            pesos_sel.append(pesos[indice])
    return combinar_vectores(seleccion, pesos_sel) if seleccion else {}


def indices_utiles(pesos: list[float]) -> list[int]:
    return [indice for indice, peso in enumerate(pesos) if peso > 0]


def vector_inicio_y_cierre(
        vectores: list[dict[str, float]],
        pesos: list[float]) -> tuple[dict[str, float], dict[str, float]]:
    utiles = indices_utiles(pesos)
    if not utiles:
        return {}, {}

    tam_cierre = 2 if len(utiles) >= 4 else 1
    cierre_idx = utiles[-tam_cierre:]
    inicio_idx = utiles[:-tam_cierre] or utiles[:1]

    pesos_cierre = list(pesos)
    if len(cierre_idx) >= 2:
        # La ultima linea suele cerrar el sentido de la estrofa.
        pesos_cierre[cierre_idx[-1]] *= 1.35
        pesos_cierre[cierre_idx[0]] *= 1.15

    return (
        vector_por_indices(vectores, pesos, inicio_idx),
        vector_por_indices(vectores, pesos_cierre, cierre_idx),
    )


def detectar_resolucion_positiva(
        vectores: list[dict[str, float]],
        pesos: list[float]) -> bool:
    inicio, cierre = vector_inicio_y_cierre(vectores, pesos)
    if not cierre:
        return False

    cierre_positivo = sumar_grupo(cierre, POSITIVAS_DIRECTAS | ROMANCE_APEGO | TRANSICION_RESOLUCION)
    cierre_negativo = sumar_grupo(cierre, NEGATIVAS_DOLOR)
    inicio_negativo = sumar_grupo(inicio, NEGATIVAS_DOLOR)
    inicio_positivo = sumar_grupo(inicio, POSITIVAS_DIRECTAS | ROMANCE_APEGO)

    if cierre_positivo >= 0.34 and cierre_positivo > cierre_negativo * 1.20:
        return True
    return inicio_negativo > inicio_positivo * 1.10 and cierre_positivo > cierre_negativo * 1.05


def cierre_invalido_por_conflicto(frases: list[str], pesos: list[float]) -> bool:
    utiles = indices_utiles(pesos)
    if not utiles:
        return True
    cierre = " ".join(frases[indice] for indice in utiles[-2:])
    cierre_repetitivo = all(es_coro_repetitivo_contextual(frases[indice], frases) for indice in utiles[-2:])
    return (
        cierre_repetitivo
        or detectar_limite_rechazo(cierre)
        or detectar_modo_conflicto_relacional(cierre) != "no_conflictivo"
        or detectar_pregunta_retorica(cierre)
        or detectar_imperativo_liberacion(cierre)
        or detectar_modo_protesta_social(cierre) != "no_protesta"
    )


def detectar_ambientacion(
        tipos_frases: list[str],
        pesos: list[float],
        vectores: list[dict[str, float]]) -> bool:
    utiles = [(tipo, peso, vector) for tipo, peso, vector in zip(tipos_frases, pesos, vectores) if peso > 0 and vector]
    if not utiles:
        return False
    soportes = sum(1 for tipo, _, _ in utiles if tipo.startswith("soporte") or tipo == "fragmento_incompleto")
    nucleos = len(utiles) - soportes
    carga_negativa = sum(sumar_grupo(vector, NEGATIVAS_DOLOR) * peso for _, peso, vector in utiles)
    carga_positiva = sum(sumar_grupo(vector, POSITIVAS_DIRECTAS | ROMANCE_APEGO) * peso for _, peso, vector in utiles)
    return soportes > nucleos and carga_negativa <= max(0.45, carga_positiva * 1.35)


def detectar_peticion_afectiva(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    if not normalizado:
        return False
    # Patron sintactico general: peticion dirigida al yo afectivo, no frase de cancion concreta.
    return bool(re.search(
        r"\b(quiero|quisiera|necesito|pido|dejame|dejame|haz|hazme|tratame|trata)\b"
        r".{0,45}\b(me|mi|conmigo)\b",
        normalizado,
    ))


def ajustar_por_peticion_afectiva(vector: dict[str, float], texto: str, nivel: str) -> dict[str, float]:
    if not vector or not detectar_peticion_afectiva(texto):
        return vector

    negativo = sumar_grupo(vector, {"ira", "desaprobacion_rechazo", "asco_repulsion"})
    vulnerable = sumar_grupo(vector, PETICION_VULNERABLE | {"molestia_fastidio"})
    if negativo > vulnerable * 1.35:
        return vector

    ajustado = dict(vector)
    for clave in {"ira", "desaprobacion_rechazo", "asco_repulsion"}:
        if clave in ajustado:
            ajustado[clave] *= 0.55
    if "molestia_fastidio" in ajustado:
        ajustado["molestia_fastidio"] *= 0.72 if nivel.startswith("estrofa") else 0.82

    for clave, piso in {
        "cuidado_carino": 0.13,
        "aprobacion_validacion": 0.10,
        "verguenza_vulnerabilidad": 0.10,
        "deseo_anhelo": 0.09,
    }.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def ajustar_por_cierre_semantico(
        vector: dict[str, float],
        vectores_frases: list[dict[str, float]],
        pesos_frases: list[float],
        tipos_frases: list[str]) -> dict[str, float]:
    if not vector or not vectores_frases:
        return vector

    inicio, cierre = vector_inicio_y_cierre(vectores_frases, pesos_frases)
    if not cierre:
        return vector

    resolucion_positiva = detectar_resolucion_positiva(vectores_frases, pesos_frases)
    ambientacion = detectar_ambientacion(tipos_frases, pesos_frases, vectores_frases)
    cierre_romance = sumar_grupo(cierre, ROMANCE_APEGO | POSITIVAS_DIRECTAS)
    cierre_transicion = sumar_grupo(cierre, TRANSICION_RESOLUCION)
    cierre_negativo = sumar_grupo(cierre, NEGATIVAS_DOLOR)
    inicio_negativo = sumar_grupo(inicio, NEGATIVAS_DOLOR)
    vector_negativo = sumar_grupo(vector, NEGATIVAS_DOLOR)
    vector_positivo = sumar_grupo(vector, POSITIVAS_DIRECTAS | ROMANCE_APEGO | TRANSICION_RESOLUCION)

    if not (resolucion_positiva or ambientacion):
        return vector

    cierre_util = cierre_romance + cierre_transicion
    if cierre_util <= cierre_negativo and vector_negativo > vector_positivo * 1.45:
        return vector

    peso_cierre = 0.26 if resolucion_positiva else 0.18
    if ambientacion:
        peso_cierre += 0.07
    ajustado = combinar_vectores([vector, cierre], [1.0 - peso_cierre, peso_cierre])

    if cierre_util > cierre_negativo or (ambientacion and vector_negativo <= 0.45):
        factor = 0.66 if inicio_negativo > 0.18 else 0.78
        for clave in NEGATIVAS_DOLOR:
            if clave in ajustado:
                ajustado[clave] *= factor

    for clave in ROMANCE_APEGO | TRANSICION_RESOLUCION | {"alegria", "entusiasmo_emocion"}:
        if clave in cierre:
            ajustado[clave] = max(ajustado.get(clave, 0.0), cierre[clave] * 0.55)

    return normalizar_vector(ajustado)


def ajustar_por_contexto_romantico_global(
        vector: dict[str, float],
        vector_contexto: dict[str, float]) -> dict[str, float]:
    if not vector or not vector_contexto:
        return vector

    contexto_romance = sumar_grupo(vector_contexto, ROMANCE_APEGO | POSITIVAS_DIRECTAS)
    contexto_negativo = sumar_grupo(vector_contexto, NEGATIVAS_DOLOR)
    vector_negativo = sumar_grupo(vector, NEGATIVAS_DOLOR)
    vector_romance = sumar_grupo(vector, ROMANCE_APEGO | POSITIVAS_DIRECTAS)

    if contexto_romance <= contexto_negativo * 1.12:
        return vector
    if vector_negativo > 0.50 and vector_negativo > vector_romance * 1.75:
        return vector

    ajustado = combinar_vectores([vector, vector_contexto], [0.86, 0.14])
    for clave in {"molestia_fastidio", "decepcion_desamor", "tristeza", "duelo_pena"}:
        if clave in ajustado and vector.get(clave, 0.0) < 0.30:
            ajustado[clave] *= 0.82
    return normalizar_vector(ajustado)


def contar_patron(texto: str, patron: str) -> int:
    return len(re.findall(patron, texto, flags=re.IGNORECASE))


def rasgos_corporales_sensuales(texto: str) -> dict[str, float]:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    escala = max(1.0, len(tokens) ** 0.5)

    # Campos semanticos generales, no frases de canciones: cuerpo, contacto,
    # intimidad, escena nocturna, juego/fantasia y posible dano/rechazo.
    patrones = {
        "corporal": r"\b(cuerp\w*|piel\w*|boca\w*|labio\w*|mano\w*|brazo\w*|pierna\w*|pecho\w*|cintura\w*|espalda\w*|body|skin|mouth|lip\w*|hand\w*|arm\w*|leg\w*|chest|back)\b",
        "contacto": r"\b(bes\w*|toc\w*|acarici\w*|abraz\w*|roz\w*|sent\w*|entreg\w*|tom\w*|mord\w*|devor\w*|kiss\w*|touch\w*|hold\w*|embrac\w*)\b",
        "intimidad": r"\b(intim\w*|desnud\w*|calor\w*|fuego\w*|sed\w*|dese\w*|pasi\w*|tentaci\w*|placer\w*|naked|heat|fire|desire|passion|pleasure|lovin\w*|lust)\b",
        "escena": r"\b(noche\w*|oscur\w*|sombra\w*|refugi\w*|luz|luces|vela\w*|cama\w*|habitaci\w*|dorm\w*|amanec\w*|night|tonight|dark\w*|shadow\w*|room|bed|sleep\w*)\b",
        "juego": r"\b(jueg\w*|rol(?:es)?|imagin\w*|fantas\w*|provoc\w*|seduc\w*|extrem\w*)\b",
        "invitacion": r"\b(ven|vamos|dame|dejame|permite\w*|quiero|quieres|acerc\w*|quedate|quedate)\b",
        "dano_rechazo": r"\b(forz\w*|oblig\w*|amenaz\w*|rechaz\w*|viol\w*|lastim\w*|sufr\w*|miedo|auxilio|vete|no quiero|no puedo|force\w*|threat\w*|hurt\w*|trap\w*|control\w*)\b",
    }
    return {
        nombre: contar_patron(normalizado, patron) / escala
        for nombre, patron in patrones.items()
    }


def evidencia_realizacion(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return bool(re.search(r"\b(entend\w*|comprend\w*|descubr\w*|revel\w*|verdad|acept\w*|cuenta)\b", normalizado))


def evidencia_gratitud(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return bool(re.search(r"\b(graci\w*|agradec\w*|bendici\w*|regalo\w*)\b", normalizado))


def evidencia_aprobacion(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    return bool(re.search(r"\b(apoy\w*|valid\w*|aprueb\w*|respal\w*|acept\w*|correct\w*)\b", normalizado))


def rasgos_paradoja_placer_danio(texto: str) -> dict[str, float]:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    escala = max(1.0, len(tokens) ** 0.5)
    patrones = {
        "placer_atraccion": r"\b(gust\w*|placer\w*|bien|atra\w*|dese\w*|quer\w*|quier\w*|necesit\w*|disfrut\w*|benefici\w*|like\w*|pleasur\w*|good|want\w*|need\w*|crav\w*|enjoy\w*)\b",
        "danio_consecuencia": r"\b(mal|dañ\w*|dan\w*|herid\w*|duele|dolor|pelig\w*|problem\w*|complic\w*|caro|perjudic\w*|bad|wrong|hurt\w*|harm\w*|danger\w*|problem\w*|complicat\w*)\b",
        "culpa_moral": r"\b(culpa\w*|arrepent\w*|perdon\w*|pecad\w*|prohib\w*|ilegal\w*|inmoral\w*|verguenz\w*|vergonz\w*|guilt\w*|regret\w*|sin|forbidden|illegal|immoral|shame\w*)\b",
        "impulso_compulsion": r"\b(evitar\w*|impedir\w*|control\w*|repit\w*|vuelv\w*|otra\s+vez|aunque|no\s+pued\w*|can't|cannot|can\s+not|can't\s+stop|cant\s+stop|can't\s+help|cant\s+help|again)\b",
        "duda_dilema": r"(\?|¿|\b(no\s+se|no\s+s[eé]|por\s+que|porque|que\s+hacer|what\s+to\s+do|why|don't\s+know|dont\s+know)\b)",
        "yo_reflexivo": r"\b(yo|me|mi|mio|mia|conmigo|myself|me|my|i)\b",
    }
    return {nombre: contar_patron(normalizado, patron) / escala for nombre, patron in patrones.items()}


def detectar_compulsion_impulso(estrofa: str, contexto: str = "") -> bool:
    texto = normalizar_simple(f"{estrofa} {contexto}")
    if not texto:
        return False
    rasgos = rasgos_paradoja_placer_danio(texto)
    estructura_no_control = bool(re.search(
        r"\b(no\s+pued\w*|no\s+logr\w*|no\s+consig\w*|can't|cant|cannot)\b.{0,45}\b(evitar\w*|impedir\w*|parar\w*|detener\w*|control\w*|stop|help|control)\b",
        texto,
    ))
    retorno_consciente = bool(re.search(
        r"\b(aunque|pero|sin\s+embargo|y\s+vuelv\w*|otra\s+vez|again)\b.{0,60}\b(mal|dañ\w*|dan\w*|culpa\w*|arrepent\w*|bad|wrong|hurt\w*|guilt\w*|regret\w*)\b",
        texto,
    ))
    return estructura_no_control or retorno_consciente or (
        rasgos["impulso_compulsion"] >= 0.35
        and rasgos["placer_atraccion"] > 0
        and (rasgos["danio_consecuencia"] + rasgos["culpa_moral"] > 0)
    )


def detectar_conflicto_interno(estrofa: str, contexto: str = "") -> bool:
    texto = normalizar_simple(f"{estrofa} {contexto}")
    if not texto:
        return False
    rasgos = rasgos_paradoja_placer_danio(texto)
    primera_persona = rasgos["yo_reflexivo"] > 0
    paradoja = rasgos["placer_atraccion"] > 0 and (rasgos["danio_consecuencia"] + rasgos["culpa_moral"] > 0)
    duda = rasgos["duda_dilema"] > 0
    compulsion = detectar_compulsion_impulso(estrofa, contexto)
    agente_externo_claro = bool(re.search(r"\b(tu|tus|usted|ustedes|ellos|ellas|he|she|they|you)\b", texto))
    limite_externo = detectar_limite_rechazo(estrofa, contexto) and agente_externo_claro
    return primera_persona and (paradoja or duda or compulsion) and not limite_externo


def detectar_paradoja_placer_danio(estrofa: str, contexto_cancion: str = "") -> str:
    texto = f"{estrofa} {contexto_cancion}".strip()
    normalizado = normalizar_simple(texto)
    if not normalizado or es_tarareo_o_vocalizacion(texto):
        return "no_paradoja"

    rasgos = rasgos_paradoja_placer_danio(normalizado)
    placer = rasgos["placer_atraccion"]
    danio = rasgos["danio_consecuencia"]
    moral = rasgos["culpa_moral"]
    impulso = rasgos["impulso_compulsion"]
    duda = rasgos["duda_dilema"]
    conflicto_interno = detectar_conflicto_interno(estrofa, contexto_cancion)
    compulsion = detectar_compulsion_impulso(estrofa, contexto_cancion)

    if compulsion and (danio + moral + duda > 0):
        return "compulsion_impulso"
    if placer <= 0:
        return "no_paradoja"
    if moral >= 0.20 and danio > 0:
        return "culpa_por_placer" if conflicto_interno else "contradiccion_moral"
    if moral >= 0.20:
        return "contradiccion_moral"
    if danio >= 0.20 and duda > 0:
        return "placer_danino"
    if danio >= 0.20:
        return "deseo_problematico" if conflicto_interno else "placer_danino"
    return "no_paradoja"


def rasgos_protesta_social(texto: str) -> dict[str, float]:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    escala = max(1.0, len(tokens) ** 0.5)

    # Patrones generales de funcion narrativa, no frases de canciones: poder,
    # control, resistencia, liberacion, pregunta desafiante y oposicion grupal.
    patrones = {
        "poder_sistema": r"\b(sistem\w*|poder\w*|autoridad\w*|ley(?:es)?|regla\w*|orden\w*|dinero|mercad\w*|deuda\w*|control\w*|power|system|rule\w*|law|money|market|debt|authority)\b",
        "control_conformidad": r"\b(obedec\w*|call\w*|silenci\w*|cadena\w*|atad\w*|encerr\w*|domina\w*|manda\w*|conform\w*|obey\w*|silent|chain\w*|bound|locked|control\w*|command\w*)\b",
        "resistencia": r"\b(resist\w*|rebel\w*|luch\w*|romp\w*|levant\w*|despiert\w*|desafi\w*|rechaz\w*|fight\w*|rise|stand|rebel\w*|refus\w*|break)\b",
        "liberacion": r"\b(liber\w*|libre\w*|suelta\w*|soltar\w*|salir|escapar|romp\w*.{0,18}cadena\w*|free\w*|release\w*|unchain\w*|escape\w*|let\s+out)\b",
        "pregunta_desafio": r"(\?|¿|\b(por que|porque|para que|quien|quienes|donde|cuando|what|why|who|where|which|whose)\b.{0,55}\b(tu|tus|usted|ustedes|ellos|nosotros|we|you|they|side|lado|bando)\b)",
        "grupo_oposicion": r"\b(nosotros|ustedes|ellos|pueblo|gente|todos|nadie|masa\w*|we|us|they|them|people|everybody|nobody)\b.{0,60}\b(contra|versus|frente|lado|bando|against|versus|side)\b",
        "inconformidad": r"\b(injust\w*|mentir\w*|fals\w*|corrup\w*|opresi\w*|abuso\w*|explot\w*|inequal\w*|unfair|lie\w*|false|corrupt\w*|oppress\w*|abuse\w*)\b",
        "juicio_social": r"\b(critic\w*|quej\w*|culp\w*|avergonz\w*|humill\w*|problem\w*|señal\w*|senal\w*|complain\w*|blame\w*|shame\w*|humiliat\w*|crucif\w*|attention)\b",
    }
    return {nombre: contar_patron(normalizado, patron) / escala for nombre, patron in patrones.items()}


def detectar_pregunta_retorica(frase: str, contexto: str = "") -> bool:
    texto = normalizar_simple(f"{frase} {contexto}")
    if not texto:
        return False
    tiene_pregunta = "?" in frase or "¿" in frase or bool(
        re.search(r"\b(por que|porque|para que|quien|quienes|donde|cuando|what|why|who|where|which)\b", texto)
    )
    if not tiene_pregunta:
        return False
    rasgos = rasgos_protesta_social(texto)
    direccion_colectiva = bool(re.search(r"\b(tu|usted\w*|ellos|nosotros|pueblo|gente|we|you|they|people|side|lado|bando)\b", texto))
    no_es_info_personal = not bool(re.search(r"\b(nombre|edad|hora|fecha|address|name|age|time|date)\b", texto))
    return no_es_info_personal and (direccion_colectiva or rasgos["poder_sistema"] + rasgos["control_conformidad"] + rasgos["resistencia"] > 0.25)


def detectar_imperativo_liberacion(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    if not normalizado:
        return False
    imperativo = bool(re.search(
        r"\b(liber\w*|suelta\w*|solt\w*|romp\w*|dej\w*.{0,18}salir|dejen\w*|escap\w*|levant\w*|despiert\w*|free\w*|release\w*|unchain\w*|break\w*|let\s+out|wake\w*|rise)\b",
        normalizado,
    ))
    rasgos = rasgos_protesta_social(normalizado)
    return imperativo and (rasgos["liberacion"] > 0 or rasgos["control_conformidad"] + rasgos["resistencia"] > 0.15)


def detectar_oposicion_conformidad_resistencia(estrofa: str, contexto: str = "") -> bool:
    texto = f"{estrofa} {contexto}".strip()
    rasgos = rasgos_protesta_social(texto)
    oposicion = rasgos["control_conformidad"] + rasgos["poder_sistema"] + rasgos["grupo_oposicion"]
    resistencia = rasgos["resistencia"] + rasgos["liberacion"] + rasgos["inconformidad"]
    return oposicion >= 0.30 and resistencia >= 0.20


def detectar_critica_sistema(estrofa: str, contexto: str = "") -> bool:
    texto = f"{estrofa} {contexto}".strip()
    rasgos = rasgos_protesta_social(texto)
    return rasgos["poder_sistema"] >= 0.25 and (
        rasgos["inconformidad"] + rasgos["control_conformidad"] + rasgos["resistencia"] + rasgos["pregunta_desafio"]
    ) >= 0.25


def detectar_modo_protesta_social(estrofa: str, contexto_cancion: str = "") -> str:
    texto = f"{estrofa} {contexto_cancion}".strip()
    if not normalizar_simple(texto):
        return "no_protesta"
    if es_tarareo_o_vocalizacion(texto):
        return "no_protesta"

    rasgos = rasgos_protesta_social(texto)
    pregunta = detectar_pregunta_retorica(estrofa, contexto_cancion)
    liberacion = detectar_imperativo_liberacion(texto) or rasgos["liberacion"] >= 0.40
    critica = detectar_critica_sistema(estrofa, contexto_cancion)
    oposicion = detectar_oposicion_conformidad_resistencia(estrofa, contexto_cancion)
    total = sum(rasgos.values())
    juicio_social = rasgos["juicio_social"] + rasgos["grupo_oposicion"]

    if liberacion and (rasgos["control_conformidad"] + rasgos["resistencia"] + rasgos["inconformidad"] >= 0.20):
        return "llamado_liberacion"
    if pregunta and (
        rasgos["pregunta_desafio"]
        + rasgos["poder_sistema"]
        + rasgos["control_conformidad"]
        + rasgos["grupo_oposicion"]
        >= 0.18
    ):
        return "desafio_postura"
    if critica:
        return "critica_sistema"
    if oposicion:
        return "inconformidad_social"
    if juicio_social >= 0.40 and (rasgos["inconformidad"] + rasgos["pregunta_desafio"] + rasgos["resistencia"] >= 0.10):
        return "inconformidad_social"
    if juicio_social >= 0.55:
        return "protesta_social"
    if total >= 0.95 and (rasgos["resistencia"] > 0 or rasgos["inconformidad"] > 0 or pregunta):
        return "protesta_social"
    return "no_protesta"


def detectar_limite_rechazo(frase: str, contexto: str = "") -> bool:
    texto = normalizar_simple(f"{frase} {contexto}")
    if not texto:
        return False

    patrones = [
        r"\b(no|nunca|jamas|jamás)\b.{0,35}\b(quier\w*|pued\w*|deb\w*|vuelv\w*|qued\w*|caer\w*|ced\w*|seduc\w*|toc\w*)\b",
        r"\b(won'?t|will not|can'?t|cannot|don'?t|do not|shouldn'?t|mustn'?t|never)\b.{0,40}\b(stay|fall|return|come|go|seduc\w*|touch|belong|be)\b",
        r"\b(deja(?:me|me)?|dejame|alejate|vete|sueltame|para|basta)\b",
        r"\b(let\s+me|leave\s+me|let\s+me\s+be|go\s+away|stop)\b",
    ]
    return any(re.search(patron, texto) for patron in patrones)


def detectar_interes_instrumental(estrofa: str, contexto: str = "") -> bool:
    texto = normalizar_simple(f"{estrofa} {contexto}")
    if not texto:
        return False
    intercambio = bool(re.search(r"\b(a cambio|por dinero|por fama|benefici\w*|interes\w*|convenien\w*|reward|fame|money|status|deal|price)\b", texto))
    deseo_atencion = bool(re.search(r"\b(desear|deseo|querer|quiero|want|desire|attention|atencion|mirada|seduc)\w*\b", texto))
    estatus = bool(re.search(r"\b(fama|famos\w*|estrella|publico|escenario|estatus|popular|show|star|stage|crowd)\b", texto))
    return intercambio or (deseo_atencion and estatus)


def detectar_dialogo_y_turnos(texto: str) -> dict[str, int | bool]:
    normalizado = normalizar_simple(texto)
    marcas = len(re.findall(r"['\"“”«»]", texto))
    reporte = len(re.findall(r"\b(dijo|dije|dices|digo|dice|respondi\w*|pregunt\w*|said|told|ask(?:ed)?|answer(?:ed)?)\b", normalizado))
    primera = len(re.findall(r"\b(yo|me|mi|mio|i|me|my|mine)\b", normalizado))
    segunda = len(re.findall(r"\b(tu|te|ti|tuyo|you|your|yours)\b", normalizado))
    return {
        "hay_dialogo": bool(marcas >= 2 or reporte > 0),
        "marcas": marcas,
        "reporte": reporte,
        "turnos_probables": int(primera > 0) + int(segunda > 0) + int(reporte > 0),
    }


def rasgos_devastacion(texto: str) -> dict[str, float]:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    escala = max(1.0, len(tokens) ** 0.5)
    patrones = {
        "catastrofe": r"\b(destru\w*|devast\w*|desol\w*|ruin\w*|ceniz\w*|crater\w*|abism\w*|catastrof\w*|armagedd\w*|disaster|wreck\w*|ash\w*|void|abyss|underworld)\b",
        "violencia_muerte": r"\b(muert\w*|matar\w*|morir\w*|guerra\w*|arma\w*|bala\w*|sangr\w*|explos\w*|weapon\w*|bullet\w*|war|death|dead|kill\w*)\b",
        "fragmentacion": r"\b(rot\w*|romp\w*|quebrad\w*|fragment\w*|herid\w*|defil\w*|contamin\w*|toxic\w*|broken|break\w*|shatter\w*|wound\w*|scar\w*|defile\w*)\b",
        "culpa": r"\b(culpa|culpable|arrepent\w*|arruin\w*|hicimos|rompimos|fallamos|mess|wrong|fault|guilt|guilty|ruined|broke|made)\b",
        "abandono": r"\b(abandon\w*|huerfan\w*|huérfan\w*|infancia|ninez|niñez|nino|niño|child\w*|orphan\w*|alone|sol\w*)\b",
        "escape_pasado": r"\b(escap\w*|huir|pasado|recuerdo\w*|trauma\w*|traumatic\w*|past|memory|memories|run\s+away|escape)\b",
        "identidad": r"\b(soy|estoy|me\s+siento|me\s+vuelvo|i\s+am|i'm|im|i\s+feel|i\s+become)\b",
        "peligro_identidad": r"\b(pelig\w*|salvaj\w*|nuclear|radioactiv\w*|explosiv\w*|toxic\w*|weapon\w*|danger\w*|wild|unstable|broken|damaged)\b",
        "esperanza": r"\b(esper\w*|milagr\w*|hope|hoping|miracle|salvar\w*|save)\b",
    }
    return {
        nombre: contar_patron(normalizado, patron) / escala
        for nombre, patron in patrones.items()
    }


def detectar_identidad_metaforica_negativa(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    if not re.search(r"\b(soy|estoy|me\s+siento|me\s+vuelvo|i\s+am|i'm|im|i\s+feel|i\s+become)\b", normalizado):
        return False
    rasgos = rasgos_devastacion(normalizado)
    carga_peligrosa = (
        rasgos["catastrofe"] + rasgos["violencia_muerte"] +
        rasgos["fragmentacion"] + rasgos["peligro_identidad"]
    )
    return carga_peligrosa >= 0.30


def detectar_culpa_responsabilidad(estrofa: str, contexto: str = "") -> bool:
    texto = normalizar_simple(f"{estrofa} {contexto}")
    rasgos = rasgos_devastacion(texto)
    primera_plural = bool(re.search(r"\b(nosotros|nosotras|nos|we|our|ours)\b", texto))
    primera_singular = bool(re.search(r"\b(yo|me|mi|i|my|mine)\b", texto))
    consecuencia = rasgos["catastrofe"] + rasgos["fragmentacion"] + rasgos["culpa"]
    return (primera_plural or primera_singular) and consecuencia >= 0.35


def detectar_abandono_vulnerabilidad(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    rasgos = rasgos_devastacion(normalizado)
    vulnerabilidad = bool(re.search(r"\b(nino|niño|child\w*|infancia|ninez|niñez|orphan\w*|huerfan\w*)\b", normalizado))
    return rasgos["abandono"] >= 0.30 and (vulnerabilidad or rasgos["escape_pasado"] > 0)


def detectar_esperanza_desesperada(texto: str, contexto: str = "") -> bool:
    combinado = f"{texto} {contexto}".strip()
    rasgos = rasgos_devastacion(combinado)
    carga_devastacion = (
        rasgos["catastrofe"] + rasgos["violencia_muerte"] +
        rasgos["fragmentacion"] + rasgos["abandono"]
    )
    return rasgos["esperanza"] > 0 and carga_devastacion >= 0.30


def detectar_modo_devastacion_interna(estrofa: str, contexto_cancion: str = "") -> str:
    texto = f"{estrofa} {contexto_cancion}".strip()
    if not normalizar_simple(texto):
        return "no_devastacion"
    rasgos = rasgos_devastacion(texto)
    escenario = rasgos["catastrofe"] + rasgos["violencia_muerte"]
    interno = rasgos["fragmentacion"] + rasgos["escape_pasado"]

    if detectar_abandono_vulnerabilidad(texto):
        return "abandono_vulnerabilidad"
    if detectar_identidad_metaforica_negativa(estrofa) or (
        rasgos["identidad"] > 0 and rasgos["peligro_identidad"] > 0
    ):
        return "identidad_peligrosa"
    if detectar_culpa_responsabilidad(estrofa, contexto_cancion):
        return "culpa_trauma"
    if detectar_esperanza_desesperada(estrofa, contexto_cancion):
        return "devastacion_externa"
    if interno >= 0.65 and rasgos["identidad"] > 0:
        return "autodestruccion_interna"
    if escenario >= 0.75:
        return "devastacion_externa"
    if interno + escenario >= 0.90:
        return "autodestruccion_interna"
    return "no_devastacion"


def rasgos_resiliencia_amorosa(texto: str) -> dict[str, float]:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    escala = max(1.0, len(tokens) ** 0.5)
    patrones = {
        "amor": r"\b(amor\w*|quer\w*|carin\w*|corazon|corazón|love|heart|care|trust)\b",
        "dolor": r"\b(dolor\w*|duele|trist\w*|lagrim\w*|llor\w*|rot\w*|romp\w*|perd\w*|miedo|pain\w*|hurt\w*|tear\w*|cry|broken|broke\w*|break\w*|lose|lost|fear)\b",
        "dificultad": r"\b(dificil|difícil|duro|cuesta|luch\w*|batall\w*|hard|fight\w*|struggle|chance|risk|win|lose)\b",
        "reparacion": r"\b(repar\w*|arregl\w*|san\w*|cur\w*|reconstru\w*|cuid\w*|confi\w*|mend\w*|heal\w*|repair\w*|rebuild\w*|trust\w*|care)\b",
        "intento": r"\b(intent\w*|trat\w*|probar\w*|seguir\w*|resist\w*|volver\w*|try|trying|tried|keep|again|back)\b",
        "aprendizaje": r"\b(aprend\w*|comprend\w*|entend\w*|verdad|learn\w*|understand\w*|realiz\w*)\b",
        "futuro": r"\b(futuro|mañana|manana|adelante|promes\w*|esper\w*|tomorrow|future|hope|promise)\b",
        "cierre_final": r"\b(adios|adiós|termin\w*|nunca|jam[aá]s|final|olvid\w*|goodbye|never|end|over|return)\b",
    }
    return {
        nombre: contar_patron(normalizado, patron) / escala
        for nombre, patron in patrones.items()
    }


def detectar_modo_resiliencia_amorosa(estrofa: str, contexto_cancion: str = "") -> str:
    texto = f"{estrofa} {contexto_cancion}".strip()
    if not normalizar_simple(texto):
        return "no_resiliencia"
    rasgos = rasgos_resiliencia_amorosa(texto)
    dolor = rasgos["dolor"] + rasgos["dificultad"]
    reparacion = rasgos["reparacion"] + rasgos["intento"]
    reflexion = rasgos["aprendizaje"]
    futuro = rasgos["futuro"]
    amor = rasgos["amor"]
    cierre = rasgos["cierre_final"]

    if amor <= 0 and dolor < 0.35:
        return "no_resiliencia"
    if cierre >= 0.30 and reparacion < 0.25 and futuro < 0.20 and dolor >= 0.20:
        return "fracaso_sin_resolucion"
    if dolor >= 0.25 and reparacion >= 0.35:
        return "reparacion_afectiva"
    if dolor >= 0.20 and reflexion >= 0.25:
        return "aprendizaje_amoroso"
    if dolor >= 0.20 and futuro >= 0.20:
        return "esperanza_dificil"
    if dolor >= 0.35 and (reparacion > 0 or reflexion > 0 or futuro > 0):
        return "dolor_con_resiliencia"
    return "no_resiliencia"


def detectar_modo_conflicto_relacional(estrofa: str, contexto_cancion: str = "") -> str:
    texto = f"{estrofa} {contexto_cancion}".strip()
    normalizado = normalizar_simple(texto)
    if not normalizado:
        return "no_conflictivo"
    if detectar_conflicto_interno(estrofa, contexto_cancion):
        return "no_conflictivo"

    limite = detectar_limite_rechazo(estrofa, contexto_cancion)
    instrumental = detectar_interes_instrumental(estrofa, contexto_cancion)
    dialogo = detectar_dialogo_y_turnos(texto)
    rasgos = rasgos_corporales_sensuales(texto)
    sensual = detectar_orientacion_corporal(texto) in {"sensual_positivo", "sensual_ambiguo", "sensual_conflictivo", "sensual_oscuro"}
    tercero = bool(re.search(r"\b(otro|otra|alguien|pareja|espos\w*|novi\w*|someone|somebody|another|lover|partner|wife|husband|girlfriend|boyfriend)\b", normalizado))
    pareja_externa = bool(re.search(r"\b(my|mi|your|tu|su)\s+(baby|love|lover|partner|wife|husband|girlfriend|boyfriend|amor|pareja|novi\w*|espos\w*)\b", normalizado))
    triangulo_implicito = bool(re.search(r"\b(he|she|him|her|el|ella)\b.{0,35}\b(with me|conmigo|sleep\w*|esper\w*|waiting|home|casa)\b", normalizado))
    culpa_engano = bool(re.search(r"\b(culpa|traici\w*|infiel|enga\w*|mentir\w*|secreto|forbidden|guilt|cheat\w*|lie|secret)\b", normalizado))
    presion = bool(re.search(r"\b(insist\w*|presion\w*|oblig\w*|seduc\w*|tentaci\w*|manipul\w*|force|pressure|seduc\w*|tempt\w*|trap\w*|control\w*)\b", normalizado))

    if limite and sensual:
        return "seduccion_conflictiva"
    if limite:
        return "rechazo_limite"
    if instrumental:
        return "manipulacion_tentacion"
    if sensual and (tercero or pareja_externa or triangulo_implicito or culpa_engano or presion):
        return "seduccion_conflictiva"
    if presion or pareja_externa or triangulo_implicito:
        return "conflicto_relacional"
    if (dialogo["hay_dialogo"] or dialogo["turnos_probables"] >= 2) and (
        tercero or pareja_externa or triangulo_implicito or culpa_engano or presion
    ):
        return "conflicto_relacional"
    if rasgos["dano_rechazo"] > 0.0 and (sensual or dialogo["hay_dialogo"]):
        return "conflicto_relacional"
    return "no_conflictivo"


def detectar_orientacion_corporal(
        texto: str,
        vector: dict[str, float] | None = None,
        vector_contexto: dict[str, float] | None = None) -> str:
    rasgos = rasgos_corporales_sensuales(texto)
    vector = vector or {}
    vector_contexto = vector_contexto or {}

    cuerpo_contacto = rasgos["corporal"] + rasgos["contacto"]
    marco_sensual = (
        cuerpo_contacto * 1.35
        + rasgos["intimidad"] * 1.30
        + rasgos["juego"] * 1.10
        + rasgos["invitacion"] * 0.85
        + rasgos["escena"] * 0.45
    )
    rechazo_dano = rasgos["dano_rechazo"]
    senal_transformer = sumar_grupo(vector, SENSUAL_MODULADORES)
    senal_contexto = sumar_grupo(vector_contexto, SENSUAL_MODULADORES)
    negativa = sumar_grupo(vector, NEGATIVAS_DOLOR) + sumar_grupo(vector_contexto, NEGATIVAS_DOLOR) * 0.45

    intimidad_clara = rasgos["intimidad"] >= 0.45
    cuerpo_en_escena = rasgos["corporal"] >= 0.35 and rasgos["escena"] > 0
    interaccion_interpersonal = bool(re.search(
        r"\b(tu|tus|usted|contigo|conmigo|pareja|amor|you|your|with\s+me|with\s+you|lover|body|skin|kiss\w*|touch\w*)\b",
        normalizar_simple(texto),
    ))
    corporal_o_interpersonal = cuerpo_contacto >= 0.35 or rasgos["contacto"] > 0 or interaccion_interpersonal

    if (marco_sensual < 0.75 and not (intimidad_clara or cuerpo_en_escena)) or (
        rasgos["intimidad"] > 0 and not corporal_o_interpersonal and detectar_paradoja_placer_danio(texto) != "no_paradoja"
    ):
        return "no_sensual"

    if rechazo_dano >= 0.55 and negativa > max(0.24, senal_transformer + senal_contexto):
        return "corporal_negativo"

    conflicto = detectar_limite_rechazo(texto) or detectar_interes_instrumental(texto)
    if conflicto and marco_sensual >= 0.75 and corporal_o_interpersonal:
        return "sensual_conflictivo" if rechazo_dano < 0.55 else "sensual_oscuro"

    if marco_sensual >= 1.65 and corporal_o_interpersonal and (senal_transformer + senal_contexto >= 0.12 or rasgos["invitacion"] > 0):
        return "sensual_positivo"

    if (marco_sensual >= 0.90 and corporal_o_interpersonal) or cuerpo_en_escena:
        return "sensual_ambiguo"

    return "no_sensual"


def detectar_modo_sensual(estrofa: str, contexto_cancion: str = "") -> bool:
    texto = f"{estrofa} {contexto_cancion}".strip()
    return detectar_orientacion_corporal(texto) in {
        "sensual_positivo", "sensual_ambiguo", "sensual_conflictivo", "sensual_oscuro",
    }


def combinar_orientaciones(*orientaciones: str) -> str:
    limpias = [orientacion for orientacion in orientaciones if orientacion]
    if "corporal_negativo" in limpias:
        return "corporal_negativo"
    if "sensual_oscuro" in limpias:
        return "sensual_oscuro"
    if "sensual_conflictivo" in limpias:
        return "sensual_conflictivo"
    if "sensual_positivo" in limpias:
        return "sensual_positivo"
    if "sensual_ambiguo" in limpias:
        return "sensual_ambiguo"
    return "no_sensual"


def detectar_atmosfera_nocturna_o_sensual(
        texto: str,
        vector: dict[str, float] | None = None,
        vector_contexto: dict[str, float] | None = None) -> bool:
    rasgos = rasgos_corporales_sensuales(texto)
    orientacion = detectar_orientacion_corporal(texto, vector, vector_contexto)
    return orientacion in {"sensual_positivo", "sensual_ambiguo"} and rasgos["escena"] > 0


def ajustar_por_orientacion_corporal(
        vector: dict[str, float],
        texto_actual: str,
        contexto: str,
        vector_contexto: dict[str, float] | None,
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    texto_eval = f"{texto_actual} {contexto}".strip()
    orientacion_actual = detectar_orientacion_corporal(texto_actual, vector, vector_contexto)
    orientacion_contexto = detectar_orientacion_corporal(texto_eval, vector, vector_contexto)
    orientacion = combinar_orientaciones(orientacion_actual, orientacion_contexto)
    if orientacion == "no_sensual":
        return vector
    if orientacion == "corporal_negativo":
        return vector

    ajustado = dict(vector)
    evidencia_real = evidencia_realizacion(texto_eval)
    evidencia_grat = evidencia_gratitud(texto_eval)
    evidencia_aprob = evidencia_aprobacion(texto_eval)
    atmosfera = detectar_atmosfera_nocturna_o_sensual(texto_eval, vector, vector_contexto)

    if not evidencia_real and "realizacion_darse_cuenta" in ajustado:
        ajustado["realizacion_darse_cuenta"] *= 0.48 if orientacion == "sensual_positivo" else 0.62
    if not evidencia_grat and "gratitud" in ajustado:
        ajustado["gratitud"] *= 0.46 if orientacion == "sensual_positivo" else 0.64
    if not evidencia_aprob and "aprobacion_validacion" in ajustado:
        ajustado["aprobacion_validacion"] *= 0.58 if orientacion == "sensual_positivo" else 0.72

    if "molestia_fastidio" in ajustado and not hay_contexto_negativo_duro(texto_eval):
        ajustado["molestia_fastidio"] *= 0.55 if orientacion == "sensual_positivo" else 0.72
    if atmosfera:
        for clave in {"tristeza", "duelo_pena", "miedo_angustia"}:
            if clave in ajustado and ajustado[clave] < 0.22:
                ajustado[clave] *= 0.72

    pisos = {
        "deseo_anhelo": 0.16,
        "entusiasmo_emocion": 0.10,
        "amor": 0.10,
        "nerviosismo_ansiedad": 0.08,
        "admiracion_aprecio": 0.07,
    }
    if orientacion == "sensual_ambiguo":
        pisos = {
            "deseo_anhelo": 0.12,
            "nerviosismo_ansiedad": 0.09,
            "entusiasmo_emocion": 0.07,
            "amor": 0.07,
        }
    if nivel.startswith("estrofa"):
        pisos = {clave: valor * 1.15 for clave, valor in pisos.items()}

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    return normalizar_vector(ajustado)


def ajustar_por_conflicto_narrativo(
        vector: dict[str, float],
        texto_actual: str,
        contexto: str,
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    modo = detectar_modo_conflicto_relacional(texto_actual, contexto)
    if modo == "no_conflictivo":
        return vector

    ajustado = dict(vector)
    evidencia_grat = evidencia_gratitud(f"{texto_actual} {contexto}")
    evidencia_aprob = evidencia_aprobacion(f"{texto_actual} {contexto}")
    evidencia_real = evidencia_realizacion(f"{texto_actual} {contexto}")

    for clave in {"aprobacion_validacion", "gratitud", "admiracion_aprecio", "amor"}:
        if clave not in ajustado:
            continue
        if clave == "gratitud" and evidencia_grat:
            continue
        if clave == "aprobacion_validacion" and evidencia_aprob:
            continue
        ajustado[clave] *= 0.30 if nivel.startswith("estrofa") else 0.45

    if not evidencia_real and "realizacion_darse_cuenta" in ajustado:
        ajustado["realizacion_darse_cuenta"] *= 0.58 if nivel.startswith("estrofa") else 0.70

    if modo == "rechazo_limite":
        pisos = {
            "aceptacion_desapego": 0.16,
            "desaprobacion_rechazo": 0.15,
            "molestia_fastidio": 0.13,
            "nerviosismo_ansiedad": 0.10,
        }
    elif modo == "manipulacion_tentacion":
        pisos = {
            "decepcion_desamor": 0.14,
            "desaprobacion_rechazo": 0.14,
            "verguenza_vulnerabilidad": 0.12,
            "nerviosismo_ansiedad": 0.10,
            "molestia_fastidio": 0.10,
        }
    elif modo == "seduccion_conflictiva":
        pisos = {
            "nerviosismo_ansiedad": 0.14,
            "molestia_fastidio": 0.12,
            "decepcion_desamor": 0.12,
            "verguenza_vulnerabilidad": 0.11,
            "deseo_anhelo": 0.09,
        }
    else:
        pisos = {
            "nerviosismo_ansiedad": 0.12,
            "molestia_fastidio": 0.11,
            "decepcion_desamor": 0.10,
            "remordimiento_culpa": 0.09,
        }

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    return normalizar_vector(ajustado)


def ajustar_por_devastacion_interna(
        vector: dict[str, float],
        texto_actual: str,
        contexto: str,
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    modo = detectar_modo_devastacion_interna(texto_actual, contexto)
    if modo == "no_devastacion":
        return vector

    texto_eval = f"{texto_actual} {contexto}".strip()
    ajustado = dict(vector)
    evidencia_grat = evidencia_gratitud(texto_eval)
    evidencia_aprob = evidencia_aprobacion(texto_eval)

    for clave in COMODINES_POSITIVOS:
        if clave not in ajustado:
            continue
        if clave == "gratitud" and evidencia_grat:
            continue
        if clave == "aprobacion_validacion" and evidencia_aprob:
            continue
        ajustado[clave] *= 0.20 if nivel.startswith("estrofa") else 0.30

    if "realizacion_darse_cuenta" in ajustado and not evidencia_realizacion(texto_eval):
        ajustado["realizacion_darse_cuenta"] *= 0.45 if nivel.startswith("estrofa") else 0.60

    if modo == "devastacion_externa":
        pisos = {
            "miedo_angustia": 0.15,
            "tristeza": 0.12,
            "duelo_pena": 0.11,
            "desaprobacion_rechazo": 0.10,
            "nerviosismo_ansiedad": 0.10,
        }
    elif modo == "autodestruccion_interna":
        pisos = {
            "miedo_angustia": 0.15,
            "nerviosismo_ansiedad": 0.13,
            "tristeza": 0.12,
            "verguenza_vulnerabilidad": 0.11,
            "molestia_fastidio": 0.09,
        }
    elif modo == "culpa_trauma":
        pisos = {
            "remordimiento_culpa": 0.16,
            "tristeza": 0.12,
            "desaprobacion_rechazo": 0.10,
            "verguenza_vulnerabilidad": 0.10,
            "miedo_angustia": 0.09,
        }
    elif modo == "identidad_peligrosa":
        pisos = {
            "miedo_angustia": 0.20,
            "nerviosismo_ansiedad": 0.15,
            "desaprobacion_rechazo": 0.13,
            "verguenza_vulnerabilidad": 0.12,
            "molestia_fastidio": 0.11,
        }
    else:
        pisos = {
            "verguenza_vulnerabilidad": 0.14,
            "tristeza": 0.13,
            "duelo_pena": 0.11,
            "miedo_angustia": 0.10,
            "remordimiento_culpa": 0.09,
        }

    if detectar_esperanza_desesperada(texto_actual, contexto):
        pisos["optimismo_esperanza"] = 0.07
        pisos["miedo_angustia"] = max(pisos.get("miedo_angustia", 0.0), 0.11)
        pisos["tristeza"] = max(pisos.get("tristeza", 0.0), 0.10)

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    return normalizar_vector(ajustado)


def ajustar_por_resiliencia_amorosa(
        vector: dict[str, float],
        texto_actual: str,
        contexto: str,
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    modo = detectar_modo_resiliencia_amorosa(texto_actual, contexto)
    if modo in {"no_resiliencia", "fracaso_sin_resolucion"}:
        return vector

    ajustado = dict(vector)
    top = top_vector(ajustado, 3)
    margen = (top[0][1] - top[1][1]) if len(top) > 1 else 1.0
    dolor = sumar_grupo(ajustado, DOLOR_AMOROSO)
    resiliencia = sumar_grupo(ajustado, RESILIENCIA_MODULADORES)

    if "decepcion_desamor" in ajustado and (margen <= 0.035 or resiliencia >= dolor * 0.45):
        ajustado["decepcion_desamor"] *= 0.82
    if "molestia_fastidio" in ajustado:
        ajustado["molestia_fastidio"] *= 0.72 if nivel.startswith("estrofa") else 0.82

    pisos = {
        "redencion_renacer": 0.105,
        "nerviosismo_ansiedad": 0.085,
        "realizacion_darse_cuenta": 0.075,
        "remordimiento_culpa": 0.070,
        "aceptacion_desapego": 0.065,
    }
    if modo == "reparacion_afectiva":
        pisos.update({
            "redencion_renacer": 0.130,
            "cuidado_carino": 0.080,
            "optimismo_esperanza": 0.075,
        })
    elif modo == "aprendizaje_amoroso":
        pisos.update({
            "realizacion_darse_cuenta": 0.115,
            "redencion_renacer": 0.105,
        })
    elif modo == "esperanza_dificil":
        pisos.update({
            "optimismo_esperanza": 0.100,
            "redencion_renacer": 0.110,
            "nerviosismo_ansiedad": 0.095,
        })

    if nivel.startswith("estrofa"):
        pisos = {clave: valor * 1.12 for clave, valor in pisos.items()}

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    return normalizar_vector(ajustado)


def ajustar_por_protesta_social(
        vector: dict[str, float],
        texto_actual: str,
        contexto: str,
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    modo = detectar_modo_protesta_social(texto_actual, contexto)
    if modo == "no_protesta":
        return vector

    texto_eval = f"{texto_actual} {contexto}".strip()
    ajustado = dict(vector)
    evidencia_grat = evidencia_gratitud(texto_eval)
    evidencia_aprob = evidencia_aprobacion(texto_eval)
    evidencia_admiracion = bool(re.search(r"\b(admir\w*|apreci\w*|respet\w*|valor\w*|honor\w*)\b", normalizar_simple(texto_eval)))
    pregunta_retorica = detectar_pregunta_retorica(texto_actual, contexto)
    imperativo_liberacion = detectar_imperativo_liberacion(texto_eval)

    for clave in {"aprobacion_validacion", "gratitud", "admiracion_aprecio", "amor", "alegria"}:
        if clave not in ajustado:
            continue
        if clave == "gratitud" and evidencia_grat:
            continue
        if clave == "aprobacion_validacion" and evidencia_aprob and not pregunta_retorica:
            continue
        if clave == "admiracion_aprecio" and evidencia_admiracion and not pregunta_retorica:
            continue
        ajustado[clave] *= 0.34 if nivel.startswith("estrofa") else 0.48

    if pregunta_retorica and "aprobacion_validacion" in ajustado:
        ajustado["aprobacion_validacion"] *= 0.45
    if not evidencia_realizacion(texto_eval) and "realizacion_darse_cuenta" in ajustado:
        # En protesta la realizacion puede ser secundaria, pero no cajon de sastre.
        ajustado["realizacion_darse_cuenta"] *= 0.86

    if modo == "llamado_liberacion":
        pisos = {
            "alivio_liberacion": 0.145,
            "redencion_renacer": 0.115,
            "desaprobacion_rechazo": 0.105,
            "molestia_fastidio": 0.090,
        }
        if imperativo_liberacion:
            pisos["curiosidad_busqueda"] = 0.070
    elif modo == "desafio_postura":
        pisos = {
            "curiosidad_busqueda": 0.140,
            "desaprobacion_rechazo": 0.115,
            "realizacion_darse_cuenta": 0.095,
            "molestia_fastidio": 0.085,
        }
    elif modo == "critica_sistema":
        pisos = {
            "desaprobacion_rechazo": 0.150,
            "molestia_fastidio": 0.120,
            "realizacion_darse_cuenta": 0.090,
            "nerviosismo_ansiedad": 0.070,
        }
    elif modo == "inconformidad_social":
        pisos = {
            "molestia_fastidio": 0.125,
            "desaprobacion_rechazo": 0.115,
            "redencion_renacer": 0.085,
            "orgullo_autovaloracion": 0.075,
        }
    else:
        pisos = {
            "desaprobacion_rechazo": 0.115,
            "molestia_fastidio": 0.105,
            "realizacion_darse_cuenta": 0.080,
            "curiosidad_busqueda": 0.070,
        }

    if nivel.startswith("estrofa"):
        pisos = {clave: valor * 1.12 for clave, valor in pisos.items()}

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    return normalizar_vector(ajustado)


def ajustar_por_paradoja_placer_danio(
        vector: dict[str, float],
        texto_actual: str,
        contexto: str,
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    modo = detectar_paradoja_placer_danio(texto_actual, contexto)
    conflicto_interno = detectar_conflicto_interno(texto_actual, contexto)
    compulsion = detectar_compulsion_impulso(texto_actual, contexto)
    if modo == "no_paradoja" and not conflicto_interno and not compulsion:
        return vector

    texto_eval = f"{texto_actual} {contexto}".strip()
    ajustado = dict(vector)
    evidencia_grat = evidencia_gratitud(texto_eval)
    evidencia_aprob = evidencia_aprobacion(texto_eval)
    cierre_liberacion = detectar_modo_resiliencia_amorosa(texto_actual, contexto) not in {
        "no_resiliencia", "fracaso_sin_resolucion"
    } or detectar_modo_protesta_social(texto_actual, contexto) == "llamado_liberacion"

    for clave in {"gratitud", "aprobacion_validacion", "sorpresa_asombro", "admiracion_aprecio"}:
        if clave not in ajustado:
            continue
        if clave == "gratitud" and evidencia_grat:
            continue
        if clave == "aprobacion_validacion" and evidencia_aprob:
            continue
        ajustado[clave] *= 0.30 if nivel.startswith("estrofa") else 0.42

    if "aceptacion_desapego" in ajustado and not cierre_liberacion:
        ajustado["aceptacion_desapego"] *= 0.38 if nivel.startswith("estrofa") else 0.50
    if "desaprobacion_rechazo" in ajustado and conflicto_interno:
        ajustado["desaprobacion_rechazo"] *= 0.72
    if "amor" in ajustado and modo in {"contradiccion_moral", "culpa_por_placer", "placer_danino", "compulsion_impulso"}:
        ajustado["amor"] *= 0.66
    if "deseo_anhelo" in ajustado and modo == "no_paradoja":
        ajustado["deseo_anhelo"] *= 0.88

    pisos = {
        "confusion": 0.130,
        "nerviosismo_ansiedad": 0.120,
        "remordimiento_culpa": 0.105,
        "molestia_fastidio": 0.090,
        "verguenza_vulnerabilidad": 0.080,
    }
    if modo == "placer_danino":
        pisos.update({
            "deseo_anhelo": 0.080,
            "desaprobacion_rechazo": 0.060,
        })
    elif modo == "contradiccion_moral":
        pisos.update({
            "remordimiento_culpa": 0.135,
            "verguenza_vulnerabilidad": 0.105,
            "desaprobacion_rechazo": 0.070,
        })
    elif modo == "deseo_problematico":
        pisos.update({
            "deseo_anhelo": 0.105,
            "nerviosismo_ansiedad": 0.135,
            "confusion": 0.120,
        })
    elif modo == "compulsion_impulso" or compulsion:
        pisos.update({
            "nerviosismo_ansiedad": 0.150,
            "confusion": 0.140,
            "remordimiento_culpa": 0.125,
            "molestia_fastidio": 0.105,
        })
    elif modo == "culpa_por_placer":
        pisos.update({
            "remordimiento_culpa": 0.155,
            "verguenza_vulnerabilidad": 0.120,
            "nerviosismo_ansiedad": 0.115,
        })

    if nivel.startswith("estrofa"):
        pisos = {clave: valor * 1.12 for clave, valor in pisos.items()}

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    return normalizar_vector(ajustado)


def calcular_peso_estrofa_resumen(
        pesos_frases: list[float],
        tipos_frases: list[str],
        modo_conflicto: str,
        modo_devastacion: str = "no_devastacion",
        modo_protesta: str = "no_protesta",
        modo_paradoja: str = "no_paradoja",
        funciones: list[str] | None = None,
        calidad: dict[str, float | str] | None = None) -> float:
    peso = max(sum(pesos_frases), 0.001)
    funciones = funciones or []
    calidad = calidad or {"score": 1.0, "nivel": "alta"}
    if not tipos_frases:
        return peso

    bajas = sum(1 for tipo in tipos_frases if tipo in {"tarareo", "sin_contenido", "coro_repetitivo", "soporte_corta"})
    proporcion_baja = bajas / len(tipos_frases)
    if proporcion_baja >= 0.65:
        peso *= 0.45
    elif proporcion_baja >= 0.45:
        peso *= 0.68

    if modo_conflicto != "no_conflictivo":
        peso *= 1.15
    if modo_devastacion != "no_devastacion":
        peso *= 1.18
    if modo_protesta != "no_protesta":
        peso *= 1.14
    if modo_paradoja != "no_paradoja":
        peso *= 1.16
    if any(f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica", "contradiccion_afectiva"}):
        peso *= 1.22
    if any(f in funciones for f in {"enumeracion_relacional", "alarde_ego", "jerga_fragmentada", "baja_densidad_semantica"}):
        peso *= 0.62
    if any(f in funciones for f in {"tarareo_ruido", "coro_repetitivo"}) and calidad.get("score", 1.0) < 0.45:
        peso *= 0.25
    if calidad.get("nivel") == "baja" and not any(
        f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica", "contradiccion_afectiva"}
    ):
        peso *= 0.55
    return max(peso, 0.001)


def ajustar_vector_cancion_por_modo_narrativo(
        vector: dict[str, float],
        items_estrofa: list[dict]) -> dict[str, float]:
    if not vector or not items_estrofa:
        return vector

    peso_total = sum(item.get("frases", 0.0) for item in items_estrofa) or 1.0
    peso_conflicto = sum(
        item.get("frases", 0.0)
        for item in items_estrofa
        if item.get("modo_conflicto") != "no_conflictivo"
    )
    if peso_conflicto / peso_total < 0.22:
        return vector

    ajustado = dict(vector)
    for clave in {"aprobacion_validacion", "gratitud", "admiracion_aprecio", "amor"}:
        if clave in ajustado:
            ajustado[clave] *= 0.72
    if "realizacion_darse_cuenta" in ajustado:
        ajustado["realizacion_darse_cuenta"] *= 0.82

    for clave, piso in {
        "nerviosismo_ansiedad": 0.10,
        "molestia_fastidio": 0.09,
        "decepcion_desamor": 0.08,
        "desaprobacion_rechazo": 0.075,
        "remordimiento_culpa": 0.07,
        "verguenza_vulnerabilidad": 0.065,
    }.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def ajustar_vector_cancion_por_modo_devastacion(
        vector: dict[str, float],
        items_estrofa: list[dict]) -> dict[str, float]:
    if not vector or not items_estrofa:
        return vector

    peso_total = sum(item.get("frases", 0.0) for item in items_estrofa) or 1.0
    peso_devastacion = sum(
        item.get("frases", 0.0)
        for item in items_estrofa
        if item.get("modo_devastacion") != "no_devastacion"
    )
    if peso_devastacion / peso_total < 0.20:
        return vector

    ajustado = dict(vector)
    for clave in COMODINES_POSITIVOS:
        if clave in ajustado:
            ajustado[clave] *= 0.58
    if "realizacion_darse_cuenta" in ajustado:
        ajustado["realizacion_darse_cuenta"] *= 0.76

    for clave, piso in {
        "miedo_angustia": 0.095,
        "nerviosismo_ansiedad": 0.09,
        "tristeza": 0.085,
        "remordimiento_culpa": 0.08,
        "verguenza_vulnerabilidad": 0.075,
        "desaprobacion_rechazo": 0.07,
    }.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def ajustar_vector_cancion_por_modo_resiliencia(
        vector: dict[str, float],
        items_estrofa: list[dict]) -> dict[str, float]:
    if not vector or not items_estrofa:
        return vector

    peso_total = sum(item.get("frases", 0.0) for item in items_estrofa) or 1.0
    peso_resiliencia = sum(
        item.get("frases", 0.0)
        for item in items_estrofa
        if item.get("modo_resiliencia") not in {None, "no_resiliencia", "fracaso_sin_resolucion"}
    )
    if peso_resiliencia / peso_total < 0.20:
        return vector

    ajustado = dict(vector)
    top = top_vector(ajustado, 2)
    margen = (top[0][1] - top[1][1]) if len(top) > 1 else 1.0
    if margen <= 0.04:
        for clave in {"decepcion_desamor", "molestia_fastidio"}:
            if clave in ajustado:
                ajustado[clave] *= 0.84

    for clave, piso in {
        "redencion_renacer": 0.085,
        "nerviosismo_ansiedad": 0.085,
        "remordimiento_culpa": 0.075,
        "realizacion_darse_cuenta": 0.070,
        "optimismo_esperanza": 0.060,
    }.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def ajustar_vector_cancion_por_modo_protesta(
        vector: dict[str, float],
        items_estrofa: list[dict]) -> dict[str, float]:
    if not vector or not items_estrofa:
        return vector

    peso_total = sum(item.get("frases", 0.0) for item in items_estrofa) or 1.0
    peso_protesta = sum(
        item.get("frases", 0.0)
        for item in items_estrofa
        if item.get("modo_protesta") not in {None, "no_protesta"}
    )
    proporcion = peso_protesta / peso_total
    if proporcion < 0.18:
        return vector

    ajustado = dict(vector)
    for clave in {"aprobacion_validacion", "gratitud", "admiracion_aprecio", "amor", "alegria"}:
        if clave in ajustado:
            ajustado[clave] *= 0.68 if proporcion < 0.35 else 0.54

    if "realizacion_darse_cuenta" in ajustado and ajustado["realizacion_darse_cuenta"] > 0.28:
        ajustado["realizacion_darse_cuenta"] *= 0.90

    pisos = {
        "desaprobacion_rechazo": 0.085,
        "molestia_fastidio": 0.080,
        "realizacion_darse_cuenta": 0.070,
        "curiosidad_busqueda": 0.055,
    }
    if proporcion >= 0.35:
        pisos.update({
            "alivio_liberacion": 0.065,
            "redencion_renacer": 0.060,
            "orgullo_autovaloracion": 0.050,
        })

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def ajustar_vector_cancion_por_modo_paradoja(
        vector: dict[str, float],
        items_estrofa: list[dict]) -> dict[str, float]:
    if not vector or not items_estrofa:
        return vector

    peso_total = sum(item.get("frases", 0.0) for item in items_estrofa) or 1.0
    peso_paradoja = sum(
        item.get("frases", 0.0)
        for item in items_estrofa
        if item.get("modo_paradoja") not in {None, "no_paradoja"}
    )
    proporcion = peso_paradoja / peso_total
    if proporcion < 0.18:
        return vector

    ajustado = dict(vector)
    for clave in {"aceptacion_desapego", "aprobacion_validacion", "gratitud", "sorpresa_asombro", "admiracion_aprecio", "amor"}:
        if clave in ajustado:
            ajustado[clave] *= 0.66 if proporcion < 0.35 else 0.52
    if "desaprobacion_rechazo" in ajustado and proporcion >= 0.25:
        ajustado["desaprobacion_rechazo"] *= 0.84

    pisos = {
        "confusion": 0.090,
        "nerviosismo_ansiedad": 0.085,
        "remordimiento_culpa": 0.080,
        "molestia_fastidio": 0.070,
        "verguenza_vulnerabilidad": 0.060,
    }
    if proporcion >= 0.35:
        pisos.update({
            "confusion": 0.110,
            "nerviosismo_ansiedad": 0.105,
            "remordimiento_culpa": 0.095,
            "deseo_anhelo": 0.055,
        })

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def ajustar_vector_cancion_por_funcion_narrativa(
        vector: dict[str, float],
        items_estrofa: list[dict]) -> dict[str, float]:
    if not vector or not items_estrofa:
        return vector

    modo = modo_narrativo_dominante(items_estrofa)
    if modo == "no_definido":
        return vector

    ajustado = dict(vector)
    if modo in {"no_compromiso", "deseo_sin_apego", "alarde_ego", "hedonismo_afectivo"}:
        for clave in {"gratitud", "admiracion_aprecio", "amor", "cuidado_carino", "alegria", "aprobacion_validacion"}:
            if clave in ajustado:
                ajustado[clave] *= 0.62
        pisos = {
            "aceptacion_desapego": 0.075,
            "deseo_anhelo": 0.070,
            "diversion_ironia": 0.060,
        }
        if modo == "alarde_ego":
            pisos["orgullo_autovaloracion"] = 0.065
        if modo == "no_compromiso":
            pisos["confusion"] = 0.060
            pisos["nerviosismo_ansiedad"] = 0.055
    elif modo == "advertencia_afectiva":
        for clave in {"gratitud", "admiracion_aprecio", "amor", "alegria"}:
            if clave in ajustado:
                ajustado[clave] *= 0.58
        pisos = {
            "decepcion_desamor": 0.075,
            "remordimiento_culpa": 0.070,
            "nerviosismo_ansiedad": 0.070,
            "confusion": 0.060,
        }
    elif modo == "autoconciencia_problematica":
        for clave in {"gratitud", "admiracion_aprecio", "alegria", "orgullo_autovaloracion"}:
            if clave in ajustado:
                ajustado[clave] *= 0.58
        pisos = {
            "confusion": 0.080,
            "remordimiento_culpa": 0.075,
            "nerviosismo_ansiedad": 0.070,
            "verguenza_vulnerabilidad": 0.065,
        }
    else:
        pisos = {}

    for clave, piso in pisos.items():
        ajustado[clave] = max(ajustado.get(clave, 0.0), piso)
    return normalizar_vector(ajustado)


def evaluar_estado_semantico(
        vector: dict[str, float],
        modo_resiliencia: str = "no_resiliencia",
        modo_paradoja: str = "no_paradoja",
        funciones: list[str] | None = None,
        calidad: dict[str, float | str] | None = None) -> tuple[str, str]:
    funciones = funciones or []
    calidad = calidad or {"score": 1.0, "nivel": "alta"}
    if not vector:
        return "sin_confianza", ""
    top = top_vector(vector, 3)
    if len(top) < 2:
        return "definido", ""

    margen = top[0][1] - top[1][1]
    if "tarareo_ruido" in funciones:
        return "sin_confianza", "tarareo, ad-lib o ruido vocal sin densidad emocional"
    if "autoconciencia_problematica" in funciones:
        return (
            "mixto_baja_confianza" if margen <= 0.055 else "definido",
            "funcion narrativa: reconocimiento de conflicto personal o incapacidad afectiva",
        )
    if "advertencia_afectiva" in funciones:
        return (
            "mixto_baja_confianza" if margen <= 0.055 else "definido",
            "funcion narrativa: advertencia de dano, no compromiso o consecuencia afectiva",
        )
    if "no_compromiso" in funciones and margen <= 0.065:
        return "baja_confianza_emocional", "no compromiso afectivo con emocion directa poco dominante"
    if "contradiccion_afectiva" in funciones:
        return "mixto_baja_confianza", "funcion narrativa: contradiccion afectiva o moral"
    if "enumeracion_relacional" in funciones:
        return "baja_confianza_emocional", "funcion narrativa: lista o acumulacion relacional"
    if "alarde_ego" in funciones and margen <= 0.055:
        return "baja_confianza_emocional", "alarde o personaje con emocion directa poco definida"
    if calidad.get("nivel") == "baja" and not any(
        f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica", "contradiccion_afectiva"}
    ):
        return "baja_confianza_textual", "jerga, fragmentacion o baja densidad semantica"
    if modo_paradoja != "no_paradoja":
        return "conflicto_interno_placer_danino", lectura_paradoja(modo_paradoja)
    if modo_resiliencia not in {"no_resiliencia", "fracaso_sin_resolucion"}:
        if margen <= 0.045 or top[0][0] in {"decepcion_desamor", "molestia_fastidio"}:
            return "mixto_baja_confianza", lectura_resiliencia(modo_resiliencia)
    if margen <= 0.030:
        return "mixto_baja_confianza", "mezcla emocional sin dominante fuerte"
    return "definido", ""


def lectura_resiliencia(modo_resiliencia: str) -> str:
    return {
        "dolor_con_resiliencia": "dolor amoroso con resistencia emocional",
        "reparacion_afectiva": "dolor amoroso con intento de reparacion",
        "aprendizaje_amoroso": "dolor amoroso procesado como aprendizaje",
        "esperanza_dificil": "dolor amoroso con esperanza dificil",
        "fracaso_sin_resolucion": "desamor final o fracaso sin reparacion",
    }.get(modo_resiliencia, "")


def lectura_paradoja(modo_paradoja: str) -> str:
    return {
        "placer_danino": "atraccion hacia algo que produce dano, culpa o complicacion",
        "contradiccion_moral": "tension entre deseo, juicio moral y consecuencia",
        "deseo_problematico": "deseo o impulso vivido como problema interno",
        "compulsion_impulso": "impulso repetido con dificultad para detenerse",
        "culpa_por_placer": "placer mezclado con culpa, verguenza o arrepentimiento",
    }.get(modo_paradoja, "")


PRIORIDAD_FUNCION_NARRATIVA = {
    "autoconciencia_problematica": 1,
    "advertencia_afectiva": 2,
    "no_compromiso": 3,
    "confesion_vulnerable": 4,
    "contradiccion_afectiva": 5,
    "deseo_sin_apego": 6,
    "alarde_ego": 7,
    "enumeracion_relacional": 8,
    "jerga_fragmentada": 9,
    "baja_densidad_semantica": 10,
    "coro_repetitivo": 11,
    "tarareo_ruido": 12,
    "emocion_directa": 20,
}


def funcion_principal(funciones: list[str]) -> str:
    if not funciones:
        return "emocion_directa"
    return min(funciones, key=lambda funcion: PRIORIDAD_FUNCION_NARRATIVA.get(funcion, 99))


def modo_paradoja_dominante(items_estrofa: list[dict]) -> str:
    if not items_estrofa:
        return "no_paradoja"
    pesos = defaultdict(float)
    total = 0.0
    for item in items_estrofa:
        peso = item.get("frases", 0.0)
        total += peso
        modo = item.get("modo_paradoja", "no_paradoja")
        if modo != "no_paradoja":
            pesos[modo] += peso
    if not pesos or total <= 0:
        return "no_paradoja"
    modo, peso = max(pesos.items(), key=lambda item: item[1])
    return modo if peso / total >= 0.18 else "no_paradoja"


def modo_narrativo_dominante(items_estrofa: list[dict]) -> str:
    return agregar_funciones_narrativas_cancion(items_estrofa).get("modo_dominante", "no_definido")


def calcular_peso_posicion_narrativa(indice_estrofa: int, total_estrofas: int, funciones: list[str]) -> float:
    if total_estrofas <= 1:
        return 1.0
    posicion = (indice_estrofa - 1) / max(1, total_estrofas - 1)
    revelacion = any(
        funcion in funciones
        for funcion in {"autoconciencia_problematica", "advertencia_afectiva", "confesion_vulnerable", "contradiccion_afectiva"}
    )
    cierre_no_compromiso = "no_compromiso" in funciones or "deseo_sin_apego" in funciones
    superficie = any(funcion in funciones for funcion in {"enumeracion_relacional", "alarde_ego", "jerga_fragmentada"})

    if posicion >= 0.66 and revelacion:
        return 1.75
    if posicion >= 0.66 and cierre_no_compromiso:
        return 1.35
    if posicion <= 0.35 and superficie and not revelacion:
        return 0.72
    return 1.0


def peso_narrativo_funcion(funcion: str) -> float:
    return {
        "autoconciencia_problematica": 2.40,
        "advertencia_afectiva": 2.15,
        "confesion_vulnerable": 2.00,
        "contradiccion_afectiva": 1.85,
        "no_compromiso": 1.65,
        "deseo_sin_apego": 1.35,
        "alarde_ego": 0.85,
        "enumeracion_relacional": 0.70,
        "jerga_fragmentada": 0.38,
        "baja_densidad_semantica": 0.30,
        "coro_repetitivo": 0.22,
        "tarareo_ruido": 0.05,
        "emocion_directa": 0.55,
    }.get(funcion, 0.50)


def modo_desde_funcion(funcion: str) -> str | None:
    return {
        "autoconciencia_problematica": "autoconciencia_problematica",
        "advertencia_afectiva": "advertencia_afectiva",
        "confesion_vulnerable": "autoconciencia_problematica",
        "contradiccion_afectiva": "conflicto_interno",
        "no_compromiso": "no_compromiso",
        "deseo_sin_apego": "deseo_sin_apego",
        "alarde_ego": "alarde_ego",
        "enumeracion_relacional": "hedonismo_afectivo",
    }.get(funcion)


def detectar_hedonismo_afectivo_cancion(pesos_modo: dict[str, float]) -> bool:
    superficie = pesos_modo.get("alarde_ego", 0.0) + pesos_modo.get("hedonismo_afectivo", 0.0)
    desapego = pesos_modo.get("no_compromiso", 0.0) + pesos_modo.get("deseo_sin_apego", 0.0)
    revelacion = pesos_modo.get("advertencia_afectiva", 0.0) + pesos_modo.get("autoconciencia_problematica", 0.0)
    return superficie > 0 and desapego > 0 and (superficie + desapego + revelacion) >= 1.0


def agregar_funciones_narrativas_cancion(items_estrofa: list[dict]) -> dict:
    if not items_estrofa:
        return {
            "modo_dominante": "no_definido",
            "submodos": "",
            "lectura_sugerida": "",
            "perfil": "",
        }

    total_estrofas = len(items_estrofa)
    pesos_modo = defaultdict(float)
    pesos_funcion = defaultdict(float)

    for posicion, item in enumerate(items_estrofa, start=1):
        funciones = item.get("funciones_narrativas", [])
        if isinstance(funciones, str):
            funciones = [f.strip() for f in funciones.split("/") if f.strip()]
        if not funciones:
            continue

        calidad = item.get("calidad_textual", {})
        if not isinstance(calidad, dict):
            calidad = {"score": 1.0, "nivel": "alta"}
        peso_base = max(item.get("frases", 0.0), 0.15)
        peso_emocional = item.get("peso_emocional", "normal")
        factor_emocional = {
            "alto": 1.35,
            "normal": 1.0,
            "bajo": 0.55,
            "nulo": 0.15,
        }.get(peso_emocional, 1.0)
        factor_calidad = 0.75 if calidad.get("nivel") == "baja" else 1.0
        if any(f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica", "contradiccion_afectiva"}):
            factor_calidad = max(factor_calidad, 1.0)
        factor_posicion = calcular_peso_posicion_narrativa(posicion, total_estrofas, funciones)

        for funcion in funciones:
            modo = modo_desde_funcion(funcion)
            if not modo:
                continue
            peso = peso_base * factor_emocional * factor_calidad * factor_posicion * peso_narrativo_funcion(funcion)
            pesos_funcion[funcion] += peso
            pesos_modo[modo] += peso

    if detectar_hedonismo_afectivo_cancion(pesos_modo):
        pesos_modo["hedonismo_afectivo"] += (
            pesos_modo.get("alarde_ego", 0.0)
            + pesos_modo.get("deseo_sin_apego", 0.0)
            + pesos_modo.get("no_compromiso", 0.0)
        ) * 0.35

    if not pesos_modo:
        return {
            "modo_dominante": "no_definido",
            "submodos": "",
            "lectura_sugerida": "",
            "perfil": "",
        }

    ordenados = sorted(pesos_modo.items(), key=lambda item: item[1], reverse=True)
    modo_dominante = ordenados[0][0]
    nucleo_no_compromiso = (
        pesos_modo.get("no_compromiso", 0.0)
        + pesos_modo.get("hedonismo_afectivo", 0.0)
        + pesos_modo.get("deseo_sin_apego", 0.0)
    )
    nucleo_revelacion = (
        pesos_modo.get("advertencia_afectiva", 0.0)
        + pesos_modo.get("autoconciencia_problematica", 0.0)
        + pesos_modo.get("conflicto_interno", 0.0)
    )
    if (
        pesos_modo.get("no_compromiso", 0.0) > 0
        and pesos_modo.get("hedonismo_afectivo", 0.0) > 0
        and nucleo_no_compromiso >= ordenados[0][1] * 0.90
    ):
        modo_dominante = "no_compromiso + hedonismo_afectivo"
        if pesos_modo.get("deseo_sin_apego", 0.0) > ordenados[0][1] * 0.18:
            modo_dominante += " + deseo_sin_apego"
    elif nucleo_revelacion >= ordenados[0][1] * 1.15 and pesos_modo.get("autoconciencia_problematica", 0.0) > 0:
        modo_dominante = "autoconciencia_problematica"

    dominantes_partes = {parte.strip() for parte in modo_dominante.split("+")}
    submodos = [
        modo for modo, peso in ordenados
        if peso > 0 and modo not in dominantes_partes
    ][:5]
    perfil = "; ".join(f"{modo}:{peso:.2f}" for modo, peso in ordenados[:6])
    lectura = lectura_narrativa_sugerida(modo_dominante, [modo for modo, _ in ordenados[:6]])
    return {
        "modo_dominante": modo_dominante,
        "submodos": " / ".join(submodos),
        "lectura_sugerida": lectura,
        "perfil": perfil,
    }


def lectura_narrativa_sugerida(modo_dominante: str, modos: list[str]) -> str:
    conjunto = set(modos)
    if modo_dominante.startswith("no_compromiso + hedonismo_afectivo"):
        if "autoconciencia_problematica" in conjunto or "advertencia_afectiva" in conjunto:
            return (
                "La cancion combina relaciones pasajeras, fiesta, deseo sin apego y alarde "
                "con una capa final de advertencia o autocritica afectiva."
            )
        return "Predomina un modo de no compromiso, hedonismo afectivo y deseo sin apego."
    if "autoconciencia_problematica" in conjunto and {"no_compromiso", "hedonismo_afectivo"} & conjunto:
        return (
            "La cancion combina alarde, deseo y relaciones pasajeras con una capa de "
            "autocritica o incapacidad afectiva hacia el cierre."
        )
    if modo_dominante == "hedonismo_afectivo":
        return "Predomina un modo de fiesta, deseo sin apego y acumulacion de relaciones pasajeras."
    if modo_dominante == "no_compromiso":
        return "Predomina la evasion del compromiso afectivo, con deseo o cercania sin estabilidad emocional."
    if modo_dominante == "advertencia_afectiva":
        return "Predomina una advertencia de dano, limite afectivo o riesgo emocional para la otra persona."
    if modo_dominante == "autoconciencia_problematica":
        return "Predomina una confesion de conflicto personal, desconfianza o incapacidad afectiva."
    if modo_dominante == "alarde_ego":
        return "Predomina la construccion de personaje, estatus o alarde con baja vulnerabilidad directa."
    if modo_dominante == "conflicto_interno":
        return "Predomina una contradiccion interna entre deseo, consecuencia y autopercepcion."
    return ""


def es_adlib_o_fragmento_fonetico(texto: str) -> bool:
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    if not tokens:
        return True
    if len(tokens) <= 4 and not tiene_verbo_probable(texto):
        vocales_ruido = sum(
            1 for token in tokens
            if re.fullmatch(r"(ah+|oh+|uh+|eh+|ey+|hey+|ya+|yeah+|nah+|rr+a+|brr+|sk+r+|w+o+w+)", token)
            or re.fullmatch(r"([a-z])\1{2,}", token)
        )
        if vocales_ruido / len(tokens) >= 0.50:
            return True
    if len(tokens) >= 2 and all(len(token) <= 3 for token in tokens):
        repetidos = Counter(tokens)
        return max(repetidos.values()) >= 2 or sum(1 for token in tokens if re.fullmatch(r"[a-z]*([a-z])\1[a-z]*", token)) >= 1
    return False


def calcular_calidad_textual_estrofa(frases: list[str]) -> dict[str, float | str]:
    texto = " ".join(frases)
    tokens = normalizar_simple(texto).split()
    total = max(1, len(tokens))
    frases_utiles = [frase for frase in frases if not es_tarareo_o_vocalizacion(frase)]
    cortas_sin_verbo = sum(1 for frase in frases_utiles if len(normalizar_simple(frase).split()) <= 5 and not tiene_verbo_probable(frase))
    adlibs = sum(1 for frase in frases if es_adlib_o_fragmento_fonetico(frase) or es_tarareo_o_vocalizacion(frase))
    repetidos = total - len(set(tokens))
    tokens_muy_cortos = sum(1 for token in tokens if len(token) <= 2)
    capitalizados = len(re.findall(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ']{2,}\b", texto))
    separadores_lista = len(re.findall(r"[,;/]", texto))
    verbos = sum(1 for frase in frases_utiles if tiene_verbo_probable(frase))

    baja = 0.0
    baja += min(0.30, (adlibs / max(1, len(frases))) * 0.55)
    baja += min(0.25, (cortas_sin_verbo / max(1, len(frases_utiles))) * 0.35) if frases_utiles else 0.25
    baja += min(0.20, (tokens_muy_cortos / total) * 0.45)
    baja += min(0.18, (repetidos / total) * 0.25)
    baja += 0.12 if verbos == 0 and total > 4 else 0.0
    if capitalizados >= 3 and separadores_lista + capitalizados >= 4:
        baja += 0.10

    score = max(0.0, min(1.0, 1.0 - baja))
    if score < 0.45:
        nivel = "baja"
    elif score < 0.68:
        nivel = "media"
    else:
        nivel = "alta"
    return {
        "score": round(score, 4),
        "nivel": nivel,
        "adlibs": adlibs,
        "cortas_sin_verbo": cortas_sin_verbo,
        "capitalizados": capitalizados,
        "separadores_lista": separadores_lista,
    }


def detectar_enumeracion_relacional(frases: list[str]) -> bool:
    texto = " ".join(frases)
    normalizado = normalizar_simple(texto)
    tokens = normalizado.split()
    if len(tokens) < 6:
        return False
    capitalizados = len(re.findall(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ']{2,}\b", texto))
    separadores = len(re.findall(r"[,;/]", texto))
    conectores_lista = len(re.findall(r"\b(y|e|o|or|and|with|con)\b", normalizado))
    verbos = sum(1 for frase in frases if tiene_verbo_probable(frase))
    posesivos = len(re.findall(r"\b(mi|mis|my|mine|mio|mia|tengo|tiene|con|with)\b", normalizado))
    muchas_entidades = capitalizados >= 3 or separadores >= 3 or conectores_lista >= 4
    poco_desarrollo = verbos <= max(1, len(frases) // 2)
    acumulacion = (capitalizados + separadores + conectores_lista + posesivos) / max(1, len(tokens) ** 0.5)
    return muchas_entidades and poco_desarrollo and acumulacion >= 1.20


def detectar_alarde_ego(frases: list[str], contexto: str = "") -> bool:
    texto = normalizar_simple(f"{' '.join(frases)} {contexto}")
    if not texto:
        return False
    primera_persona = len(re.findall(r"\b(yo|me|mi|mis|mio|mia|soy|tengo|voy|hago|my|me|i|im|i'm|mine)\b", texto))
    acumulacion = len(re.findall(r"\b(mucho\w*|todo\w*|nadie|siempre|nunca|mas|mejor|many|all|every|never|always|more|best)\b", texto))
    estatus_exceso = len(re.findall(r"\b(fam\w*|poder\w*|rico\w*|luj\w*|fiest\w*|club|vip|money|cash|rich|famous|party|status|power)\b", texto))
    vulnerabilidad = len(re.findall(r"\b(duele|culpa|perdon|triste|miedo|llor\w*|arrepent\w*|hurt|sorry|sad|afraid|guilt|regret)\b", texto))
    calidad = calcular_calidad_textual_estrofa(frases)
    return primera_persona >= 2 and (acumulacion + estatus_exceso >= 2) and vulnerabilidad == 0 and calidad["score"] <= 0.78


def detectar_no_compromiso_afectivo(frases: list[str], contexto: str = "") -> bool:
    texto = normalizar_simple(f"{' '.join(frases)} {contexto}")
    if not texto:
        return False
    desapego = bool(re.search(
        r"\b(no|nunca|jamas|jam[aá]s|sin|ni)\b.{0,35}\b(compromis\w*|cas\w*|enamor\w*|amor|confi\w*|quedar\w*|volver\w*|commit\w*|marry|love|trust|stay)\b",
        texto,
    ))
    rotacion = bool(re.search(r"\b(otra\w*|otro\w*|alguien|varias|varios|many|another|someone|everybody)\b", texto))
    deseo_no_apego = detectar_orientacion_corporal(texto) in {"sensual_ambiguo", "sensual_positivo"} and (
        desapego or rotacion or detectar_alarde_ego(frases, contexto)
    )
    return desapego or deseo_no_apego


def detectar_advertencia_afectiva(frases: list[str], contexto: str = "") -> bool:
    texto = normalizar_simple(f"{' '.join(frases)} {contexto}")
    if not texto:
        return False
    advertencia = bool(re.search(r"\b(cuidado|adv\w*|warning|careful|beware|no\s+te|dont|don't|do\s+not)\b", texto))
    dano_futuro = bool(re.search(
        r"\b(voy|puedo|puede|termin\w*|acabar\w*|romp\w*|lastim\w*|dañ\w*|dan\w*|herir\w*|hurt\w*|break\w*)\b.{0,45}\b(te|tu|corazon|heart|you)\b",
        texto,
    ))
    no_corresponder = bool(re.search(
        r"\b(no|nunca|jamas|jam[aá]s)\b.{0,45}\b(amar\w*|quer\w*|correspond\w*|confi\w*|love|trust|stay)\b",
        texto,
    ))
    return advertencia or dano_futuro or no_corresponder


def detectar_autoconciencia_problematica(frases: list[str], contexto: str = "") -> bool:
    texto = normalizar_simple(f"{' '.join(frases)} {contexto}")
    if not texto:
        return False
    primera = bool(re.search(r"\b(yo|me|mi|soy|estoy|myself|i|me|my)\b", texto))
    reconocimiento = bool(re.search(
        r"\b(no\s+se|no\s+s[eé]|no\s+entiendo|no\s+confio|no\s+puedo|quiero\s+cambiar|cambiar|arrepent\w*|culpa\w*|hago\s+dañ|hago\s+dan|dont\s+know|don't\s+know|cant|can't|change|regret|guilt|trust)\b",
        texto,
    ))
    identidad_problem = bool(re.search(r"\b(asi|as[ií]|problema|malo|mal|toxic\w*|bad|wrong|problem)\b", texto))
    return primera and reconocimiento and (identidad_problem or detectar_paradoja_placer_danio(texto) != "no_paradoja")


def detectar_funcion_narrativa_estrofa(
        estrofa: str,
        frases: list[str],
        contexto_cancion: str = "") -> list[str]:
    funciones = []
    calidad = calcular_calidad_textual_estrofa(frases)
    if all(es_tarareo_o_vocalizacion(frase) or es_adlib_o_fragmento_fonetico(frase) for frase in frases):
        funciones.append("tarareo_ruido")
    if sum(1 for frase in frases if es_coro_repetitivo_contextual(frase, frases)) >= max(2, len(frases) - 1):
        funciones.append("coro_repetitivo")
    if detectar_enumeracion_relacional(frases):
        funciones.append("enumeracion_relacional")
    if detectar_alarde_ego(frases, contexto_cancion):
        funciones.append("alarde_ego")
    if detectar_no_compromiso_afectivo(frases, contexto_cancion):
        funciones.append("no_compromiso")
    if detectar_advertencia_afectiva(frases, contexto_cancion):
        funciones.append("advertencia_afectiva")
    if detectar_autoconciencia_problematica(frases, contexto_cancion):
        funciones.append("autoconciencia_problematica")
    if detectar_paradoja_placer_danio(estrofa, contexto_cancion) != "no_paradoja":
        funciones.append("contradiccion_afectiva")
    orientacion = detectar_orientacion_corporal(f"{estrofa} {contexto_cancion}")
    if orientacion in {"sensual_ambiguo", "sensual_positivo"}:
        if "no_compromiso" in funciones or "alarde_ego" in funciones:
            funciones.append("deseo_sin_apego")
        else:
            funciones.append("emocion_directa")
    if calidad["nivel"] == "baja":
        funciones.append("baja_densidad_semantica")
    elif calidad["nivel"] == "media" and not any(f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica", "contradiccion_afectiva"}):
        funciones.append("jerga_fragmentada")
    if not funciones:
        funciones.append("emocion_directa")
    return list(dict.fromkeys(funciones))


def funcion_narrativa_a_texto(funciones: list[str]) -> str:
    ordenadas = sorted(funciones, key=lambda funcion: PRIORIDAD_FUNCION_NARRATIVA.get(funcion, 99))
    return " / ".join(ordenadas) if ordenadas else "emocion_directa"


def peso_emocional_por_funcion(funciones: list[str], calidad: dict[str, float | str]) -> str:
    if any(f in funciones for f in {"tarareo_ruido", "coro_repetitivo"}) and calidad["score"] < 0.45:
        return "nulo"
    if any(f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica", "confesion_vulnerable", "contradiccion_afectiva"}):
        return "alto"
    if any(f in funciones for f in {"enumeracion_relacional", "alarde_ego", "jerga_fragmentada", "baja_densidad_semantica"}):
        return "bajo"
    return "normal"


def aporte_global_por_funcion(funciones: list[str]) -> str:
    aportes = []
    if "autoconciencia_problematica" in funciones:
        aportes.append("autoconciencia_problematica")
    if "advertencia_afectiva" in funciones:
        aportes.append("advertencia_afectiva")
    if "no_compromiso" in funciones:
        aportes.append("no_compromiso")
    if "contradiccion_afectiva" in funciones:
        aportes.append("conflicto_interno")
    if "deseo_sin_apego" in funciones:
        aportes.append("deseo_sin_apego")
    if "alarde_ego" in funciones:
        aportes.append("alarde_ego")
    if "enumeracion_relacional" in funciones:
        aportes.append("enumeracion")
    return " + ".join(dict.fromkeys(aportes)) if aportes else "emocion_directa"


def ajustar_por_funcion_narrativa(
        vector: dict[str, float],
        funciones: list[str],
        calidad: dict[str, float | str],
        nivel: str) -> dict[str, float]:
    if not vector:
        return vector

    ajustado = dict(vector)
    baja_densidad = "baja_densidad_semantica" in funciones or calidad["score"] < 0.50
    enumerativo = any(f in funciones for f in {"enumeracion_relacional", "alarde_ego", "jerga_fragmentada"})
    no_compromiso = "no_compromiso" in funciones or "deseo_sin_apego" in funciones
    advertencia = "advertencia_afectiva" in funciones
    autoconciencia = "autoconciencia_problematica" in funciones

    if baja_densidad:
        for clave in {"gratitud", "aprobacion_validacion", "admiracion_aprecio", "amor", "alegria", "sorpresa_asombro"}:
            if clave in ajustado:
                ajustado[clave] *= 0.38
        if "neutral_contemplativa" in ajustado:
            ajustado["neutral_contemplativa"] = max(ajustado["neutral_contemplativa"], 0.10)

    if enumerativo:
        for clave in {"gratitud", "admiracion_aprecio", "amor", "cuidado_carino", "alegria", "aprobacion_validacion"}:
            if clave in ajustado:
                ajustado[clave] *= 0.42
        ajustado["diversion_ironia"] = max(ajustado.get("diversion_ironia", 0.0), 0.075)
        ajustado["orgullo_autovaloracion"] = max(ajustado.get("orgullo_autovaloracion", 0.0), 0.070)
        ajustado["aceptacion_desapego"] = max(ajustado.get("aceptacion_desapego", 0.0), 0.055)

    if no_compromiso:
        for clave in {"amor", "cuidado_carino", "gratitud", "admiracion_aprecio", "alegria"}:
            if clave in ajustado:
                ajustado[clave] *= 0.48
        for clave, piso in {
            "aceptacion_desapego": 0.105,
            "deseo_anhelo": 0.090,
            "nerviosismo_ansiedad": 0.070,
            "confusion": 0.060,
        }.items():
            ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    if advertencia:
        for clave in {"diversion_ironia", "amor", "alegria", "gratitud", "admiracion_aprecio"}:
            if clave in ajustado:
                ajustado[clave] *= 0.45
        for clave, piso in {
            "decepcion_desamor": 0.105,
            "remordimiento_culpa": 0.100,
            "verguenza_vulnerabilidad": 0.090,
            "nerviosismo_ansiedad": 0.085,
            "confusion": 0.075,
            "aceptacion_desapego": 0.070,
        }.items():
            ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    if autoconciencia:
        for clave in {"diversion_ironia", "orgullo_autovaloracion", "alegria", "gratitud", "admiracion_aprecio"}:
            if clave in ajustado:
                ajustado[clave] *= 0.50
        for clave, piso in {
            "confusion": 0.120,
            "remordimiento_culpa": 0.115,
            "nerviosismo_ansiedad": 0.105,
            "verguenza_vulnerabilidad": 0.095,
            "decepcion_desamor": 0.070,
        }.items():
            ajustado[clave] = max(ajustado.get(clave, 0.0), piso)

    if nivel.startswith("estrofa") and any(f in funciones for f in {"advertencia_afectiva", "autoconciencia_problematica"}):
        ajustado = {clave: valor * (1.05 if clave in PARADOJA_MODULADORES | {"decepcion_desamor", "aceptacion_desapego"} else 1.0)
                    for clave, valor in ajustado.items()}

    return normalizar_vector(ajustado)


def reforzar_por_mayoria_romantica(
        vector: dict[str, float],
        vectores_frases: list[dict[str, float]],
        texto_estrofa: str) -> dict[str, float]:
    if not vector or not vectores_frases:
        return vector

    claves_positivas = {
        "amor", "alegria", "admiracion_aprecio", "cuidado_carino",
        "gratitud", "deseo_anhelo", "aprobacion_validacion",
    }
    dominantes = [top_vector(vector_frase, 1)[0][0] for vector_frase in vectores_frases if vector_frase]
    positivas = sum(1 for clave in dominantes if clave in claves_positivas)
    minimo = max(2, len(dominantes) // 2)
    contexto_romantico = (
        hay_contexto_positivo_correspondido(texto_estrofa)
        or hay_halago_romantico(texto_estrofa)
        or hay_metafora_corporal_romantica(texto_estrofa)
    )

    if not contexto_romantico or positivas < minimo or hay_contexto_negativo_duro(texto_estrofa):
        return vector

    ajustado = dict(vector)
    ajustado["amor"] = max(ajustado.get("amor", 0.0), 0.24)
    ajustado["alegria"] = max(ajustado.get("alegria", 0.0), 0.16)
    ajustado["admiracion_aprecio"] = max(ajustado.get("admiracion_aprecio", 0.0), 0.15)
    ajustado["cuidado_carino"] = max(ajustado.get("cuidado_carino", 0.0), 0.14)
    ajustado["deseo_anhelo"] = max(ajustado.get("deseo_anhelo", 0.0), 0.13)
    ajustado["gratitud"] = max(ajustado.get("gratitud", 0.0), 0.10)

    for clave in {
        "decepcion_desamor", "tristeza", "duelo_pena", "molestia_fastidio",
        "aceptacion_desapego", "confusion", "realizacion_darse_cuenta",
    }:
        if clave in ajustado:
            ajustado[clave] *= 0.42

    for clave in {"redencion_renacer", "optimismo_esperanza", "neutral_contemplativa"}:
        if clave in ajustado:
            ajustado[clave] *= 0.65

    return normalizar_vector(ajustado)


def ajustar_vector_cancion_por_coherencia(vector: dict[str, float]) -> dict[str, float]:
    if not vector:
        return vector

    positivas = sum(vector.get(clave, 0.0) for clave in {
        "amor", "deseo_anhelo", "cuidado_carino", "alegria",
        "admiracion_aprecio", "gratitud", "entusiasmo_emocion",
    })
    negativas = sum(vector.get(clave, 0.0) for clave in {
        "decepcion_desamor", "tristeza", "duelo_pena", "molestia_fastidio",
        "ira", "miedo_angustia", "remordimiento_culpa", "desaprobacion_rechazo",
    })
    romance = vector.get("amor", 0.0) + vector.get("deseo_anhelo", 0.0) + vector.get("cuidado_carino", 0.0)

    if romance < 0.18 or positivas <= negativas * 1.15:
        return vector

    ajustado = dict(vector)
    for clave in {"decepcion_desamor", "tristeza", "duelo_pena", "molestia_fastidio"}:
        if clave in ajustado:
            ajustado[clave] *= 0.72
    for clave in {"amor", "deseo_anhelo", "cuidado_carino", "alegria"}:
        if clave in ajustado:
            ajustado[clave] *= 1.08
    return normalizar_vector(ajustado)


def ajustar_vector_coherencia_contextual(
        vector: dict[str, float],
        texto_actual: str,
        contexto_limite: str,
        nivel: str) -> dict[str, float]:
    """
    Corrige el vector sin salir de la ventana permitida.

    En frase, contexto_limite es frase anterior + actual + siguiente dentro de
    la misma estrofa. En estrofa, contexto_limite puede incluir estrofas vecinas.
    """
    if not vector:
        return vector

    ajustado = dict(vector)
    actual = normalizar_simple(texto_actual)
    contexto = normalizar_simple(contexto_limite)
    negativo_contextual = hay_contexto_negativo_relacional(contexto)
    positivo_correspondido = hay_contexto_positivo_correspondido(contexto)
    negativo_duro = hay_contexto_negativo_duro(contexto)
    if positivo_correspondido and not negativo_duro:
        negativo_contextual = False

    amor_mencionado_no_sentido = hay_amor_mencionado_no_sentido(contexto)
    reflexion_conflicto = tiene_marcador(contexto, MARCADORES_REFLEXION) and negativo_contextual
    reproche_imperativo = tiene_marcador(contexto, MARCADORES_REPROCHE_IMPERATIVO)
    cambio_negativo = (
        tiene_marcador(contexto, MARCADORES_CAMBIO_NEGATIVO)
        and tiene_marcador(contexto, ["distinto", "distinta", "diferente", "cambiaste", "cambio", "cambiar"])
    )
    repeticion_generica = es_repeticion_emocional_generica(actual)
    inversion_emocional = hay_inversion_emocional(contexto)
    metafora_negativa = hay_metafora_negativa(contexto)
    sospecha_amorosa = hay_sospecha_amorosa(contexto)
    autoafirmacion_desapego = hay_autoafirmacion_desapego(contexto)
    contraste_negado = hay_contraste_negado(contexto)
    dependencia_amorosa = hay_dependencia_amorosa_sin_ruptura(contexto)
    halago_romantico = hay_halago_romantico(contexto) or hay_halago_romantico(actual)
    metafora_corporal_romantica = hay_metafora_corporal_romantica(contexto)
    frase_soporte_romantica = es_frase_soporte_romantica(actual)
    proteccion_romantica = (
        not negativo_duro
        and (
            positivo_correspondido
            or halago_romantico
            or metafora_corporal_romantica
            or (frase_soporte_romantica and intensidad_positiva_correspondida(contexto) >= 1)
        )
    )

    if proteccion_romantica:
        ajustado["amor"] = max(ajustado.get("amor", 0.0), 0.24)
        ajustado["admiracion_aprecio"] = max(ajustado.get("admiracion_aprecio", 0.0), 0.18 if halago_romantico else 0.13)
        ajustado["cuidado_carino"] = max(ajustado.get("cuidado_carino", 0.0), 0.13 if halago_romantico else 0.11)
        ajustado["alegria"] = max(ajustado.get("alegria", 0.0), 0.14)
        ajustado["gratitud"] = max(ajustado.get("gratitud", 0.0), 0.09)
        if (
            tiene_marcador(contexto, ["me enamore", "querias", "quieres", "para los dos", "gran amor"])
            or metafora_corporal_romantica
            or tiene_marcador(contexto, ["tentacion", "mirar de tentacion"])
        ):
            ajustado["deseo_anhelo"] = max(ajustado.get("deseo_anhelo", 0.0), 0.12)
        if metafora_corporal_romantica:
            ajustado["nerviosismo_ansiedad"] = max(ajustado.get("nerviosismo_ansiedad", 0.0), 0.08)

        for clave in {"decepcion_desamor", "tristeza", "duelo_pena", "molestia_fastidio", "aceptacion_desapego", "confusion"}:
            if clave in ajustado:
                ajustado[clave] *= 0.30 if (halago_romantico or metafora_corporal_romantica) else 0.38
        if halago_romantico or metafora_corporal_romantica or frase_soporte_romantica:
            for clave in {"realizacion_darse_cuenta", "redencion_renacer", "optimismo_esperanza", "neutral_contemplativa"}:
                if clave in ajustado:
                    ajustado[clave] *= 0.55

    if contraste_negado:
        for clave in {"decepcion_desamor", "tristeza", "duelo_pena", "miedo_angustia", "molestia_fastidio"}:
            if clave in ajustado:
                ajustado[clave] *= 0.35
        ajustado["realizacion_darse_cuenta"] = max(ajustado.get("realizacion_darse_cuenta", 0.0), 0.16)
        ajustado["alivio_liberacion"] = max(ajustado.get("alivio_liberacion", 0.0), 0.12)
        ajustado["optimismo_esperanza"] = max(ajustado.get("optimismo_esperanza", 0.0), 0.10)

    if dependencia_amorosa:
        for clave in {"decepcion_desamor", "duelo_pena", "tristeza", "aceptacion_desapego"}:
            if clave in ajustado:
                ajustado[clave] *= 0.45
        ajustado["amor"] = max(ajustado.get("amor", 0.0), 0.16)
        ajustado["deseo_anhelo"] = max(ajustado.get("deseo_anhelo", 0.0), 0.14)
        ajustado["cuidado_carino"] = max(ajustado.get("cuidado_carino", 0.0), 0.10)

    if amor_mencionado_no_sentido:
        for clave in POSITIVAS_DIRECTAS:
            if clave in ajustado:
                ajustado[clave] *= 0.28 if nivel == "estrofa" else 0.35
        if "deseo_anhelo" in ajustado and not tiene_marcador(contexto, MARCADORES_DESEO):
            ajustado["deseo_anhelo"] *= 0.35 if nivel == "estrofa" else 0.45

        if "amor" in actual and tiene_marcador(contexto, ["confundi", "confundiste", "rival", "derechos", "derecho"]):
            ajustado["confusion"] = max(ajustado.get("confusion", 0.0), 0.18)
            ajustado["molestia_fastidio"] = max(ajustado.get("molestia_fastidio", 0.0), 0.16)
            ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.15)
        elif tiene_marcador(contexto, ["rival", "contra", "mal", "fallaste", "dano", "danaste"]):
            ajustado["molestia_fastidio"] = max(ajustado.get("molestia_fastidio", 0.0), 0.18)
            ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.16)
        else:
            ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.15)

    if reflexion_conflicto:
        ajustado["realizacion_darse_cuenta"] = max(ajustado.get("realizacion_darse_cuenta", 0.0), 0.18)
        for clave in {"amor", "alegria", "admiracion_aprecio", "gratitud", "aprobacion_validacion"}:
            if clave in ajustado:
                ajustado[clave] *= 0.55

    if reproche_imperativo:
        for clave in {"amor", "alegria", "admiracion_aprecio", "gratitud", "aprobacion_validacion", "diversion_ironia"}:
            if clave in ajustado:
                ajustado[clave] *= 0.42 if nivel.startswith("estrofa") else 0.50
        ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.17)
        ajustado["verguenza_vulnerabilidad"] = max(ajustado.get("verguenza_vulnerabilidad", 0.0), 0.13)
        ajustado["remordimiento_culpa"] = max(ajustado.get("remordimiento_culpa", 0.0), 0.12)

    if cambio_negativo:
        for clave in {"admiracion_aprecio", "amor", "alegria", "aprobacion_validacion"}:
            if clave in ajustado:
                ajustado[clave] *= 0.48
        ajustado["realizacion_darse_cuenta"] = max(ajustado.get("realizacion_darse_cuenta", 0.0), 0.17)
        ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.15)

    if inversion_emocional:
        for clave in POSITIVAS_DIRECTAS | {"deseo_anhelo"}:
            if clave in ajustado:
                ajustado[clave] *= 0.35 if nivel.startswith("estrofa") else 0.45
        ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.22)
        ajustado["tristeza"] = max(ajustado.get("tristeza", 0.0), 0.16)
        ajustado["duelo_pena"] = max(ajustado.get("duelo_pena", 0.0), 0.14)
        ajustado["aceptacion_desapego"] = max(ajustado.get("aceptacion_desapego", 0.0), 0.12)

    if metafora_negativa:
        if tiene_marcador(contexto, ["hielo", "frio", "fria", "congelo", "congelar", "congelado"]):
            if "deseo_anhelo" in ajustado:
                ajustado["deseo_anhelo"] *= 0.45
            if "amor" in ajustado:
                ajustado["amor"] *= 0.45
            ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.18)
            ajustado["aceptacion_desapego"] = max(ajustado.get("aceptacion_desapego", 0.0), 0.13)
        if tiene_marcador(contexto, ["piedra", "vacio", "vacia", "silencio", "oscuridad", "sombra", "desierto"]):
            for clave in {"amor", "alegria", "admiracion_aprecio", "deseo_anhelo"}:
                if clave in ajustado:
                    ajustado[clave] *= 0.55
            ajustado["tristeza"] = max(ajustado.get("tristeza", 0.0), 0.17)
            ajustado["duelo_pena"] = max(ajustado.get("duelo_pena", 0.0), 0.15)

    if sospecha_amorosa:
        for clave in {"admiracion_aprecio", "amor", "alegria", "aprobacion_validacion", "deseo_anhelo"}:
            if clave in ajustado:
                ajustado[clave] *= 0.42
        ajustado["realizacion_darse_cuenta"] = max(ajustado.get("realizacion_darse_cuenta", 0.0), 0.18)
        ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.18)
        ajustado["tristeza"] = max(ajustado.get("tristeza", 0.0), 0.12)

    if autoafirmacion_desapego:
        for clave in {"amor", "alegria", "admiracion_aprecio", "aprobacion_validacion", "deseo_anhelo"}:
            if clave in ajustado:
                ajustado[clave] *= 0.38
        ajustado["aceptacion_desapego"] = max(ajustado.get("aceptacion_desapego", 0.0), 0.18)
        ajustado["molestia_fastidio"] = max(ajustado.get("molestia_fastidio", 0.0), 0.17)
        ajustado["realizacion_darse_cuenta"] = max(ajustado.get("realizacion_darse_cuenta", 0.0), 0.14)

    if repeticion_generica:
        if negativo_contextual:
            for clave in POSITIVAS_DIRECTAS | {"diversion_ironia"}:
                if clave in ajustado:
                    ajustado[clave] *= 0.25
            ajustado["deseo_anhelo"] = max(ajustado.get("deseo_anhelo", 0.0), 0.14)
            ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.16)
        else:
            for clave in POSITIVAS_DIRECTAS:
                if clave in ajustado:
                    ajustado[clave] *= 0.70

    if negativo_contextual:
        positiva_total = sum(ajustado.get(clave, 0.0) for clave in POSITIVAS_DIRECTAS)
        negativa_total = sum(ajustado.get(clave, 0.0) for clave in NEGATIVAS_CONTEXTO)
        if positiva_total > negativa_total * 1.15 and not tiene_marcador(contexto, MARCADORES_AMOR_AFIRMATIVO):
            for clave in POSITIVAS_DIRECTAS:
                if clave in ajustado:
                    ajustado[clave] *= 0.50
            if tiene_marcador(contexto, ["confundi", "confundiste", "no entiendo", "no se"]):
                ajustado["confusion"] = max(ajustado.get("confusion", 0.0), 0.16)
            elif tiene_marcador(contexto, ["mal", "rival", "contra", "derechos"]):
                ajustado["molestia_fastidio"] = max(ajustado.get("molestia_fastidio", 0.0), 0.16)
            else:
                ajustado["decepcion_desamor"] = max(ajustado.get("decepcion_desamor", 0.0), 0.14)

    ajustado = ajustar_por_orientacion_corporal(
        ajustado,
        texto_actual,
        contexto_limite,
        None,
        nivel,
    )
    ajustado = ajustar_por_conflicto_narrativo(
        ajustado,
        texto_actual,
        contexto_limite,
        nivel,
    )
    ajustado = ajustar_por_devastacion_interna(
        ajustado,
        texto_actual,
        contexto_limite,
        nivel,
    )
    ajustado = ajustar_por_resiliencia_amorosa(
        ajustado,
        texto_actual,
        contexto_limite,
        nivel,
    )
    ajustado = ajustar_por_protesta_social(
        ajustado,
        texto_actual,
        contexto_limite,
        nivel,
    )
    ajustado = ajustar_por_paradoja_placer_danio(
        ajustado,
        texto_actual,
        contexto_limite,
        nivel,
    )
    ajustado = ajustar_por_peticion_afectiva(ajustado, contexto_limite, nivel)
    return normalizar_vector(ajustado)


def analizar_jerarquico(
        archivo_txt: str,
        out_dir: str,
        modelo: str,
        frases_por_estrofa: int,
        top_k_frase: int,
        top_k_estrofa: int,
        peso_estrofa_completa: float,
        peso_promedio_frases: float,
        peso_contexto_estrofas: float):
    preparar_salida(out_dir)
    lineas = leer_lineas_letra(archivo_txt)
    estrofas = agrupar_en_estrofas(lineas, frases_por_estrofa)
    classifier = cargar_zero_shot(modelo)

    peso_estrofa_completa, peso_frases, peso_contexto_estrofas = normalizar_pesos(
        peso_estrofa_completa,
        peso_promedio_frases,
        peso_contexto_estrofas,
    )
    labels = clave_a_label()

    filas_frases = []
    filas_estrofas = []
    vectores_estrofa = []

    for idx_estrofa, estrofa in enumerate(estrofas):
        vectores_frases = []
        pesos_frases = []
        tipos_frases = []
        for idx, frase in enumerate(estrofa["frases"], start=1):
            tipo_frase, peso_frase = tipo_y_peso_frase(frase)
            if peso_frase > 0 and es_coro_repetitivo_contextual(frase, estrofa["frases"]):
                if motivo_tematico_con_carga(frase):
                    tipo_frase = "motivo_tematico_repetido"
                    peso_frase = max(0.40, min(peso_frase, 0.60))
                else:
                    tipo_frase = "coro_repetitivo"
                    peso_frase = 0.0
            tipos_frases.append(tipo_frase)
            contexto = contexto_frase(estrofas, idx_estrofa, idx - 1)
            if peso_frase <= 0:
                etiquetas = []
                vector_frase_crudo = {}
                vector_frase = {}
            else:
                etiquetas = clasificar_segmento(
                    classifier,
                    frase,
                    contexto,
                    threshold=0.0,
                    top_k=top_k_frase,
                )
                vector_frase_crudo = vector_desde_etiquetas(etiquetas, normalizar=True)
                vector_frase = ajustar_vector_coherencia_contextual(
                    vector_frase_crudo,
                    frase,
                    contexto,
                    nivel="frase",
                )
            vectores_frases.append(vector_frase)
            pesos_frases.append(peso_frase)
            top_principal = top_vector(vector_frase, 1)[0] if vector_frase else ("neutral_sin_contenido", 0.0)
            orientacion_corporal_frase = combinar_orientaciones(
                detectar_orientacion_corporal(frase, vector_frase),
                detectar_orientacion_corporal(contexto, vector_frase),
            )

            filas_frases.append({
                "estrofa": estrofa["estrofa"],
                "frase": idx,
                "texto": frase,
                "contexto": contexto,
                "tipo_frase": tipo_frase,
                "peso_frase": round(peso_frase, 4),
                "orientacion_corporal": orientacion_corporal_frase,
                "dominante": top_principal[0],
                "dominante_pct": round(top_principal[1] * 100, 4),
                "top_emociones": vector_a_texto(vector_frase, top_k_frase),
                "top_pre_filtro": vector_a_texto(vector_frase_crudo, top_k_frase),
                "etiquetas_crudas": "; ".join(f"{e['clave']}:{e['score']}" for e in etiquetas),
            })

        vector_promedio_frases = combinar_vectores(vectores_frases, pesos_frases)
        texto_estrofa_contenido = texto_con_contenido(estrofa["frases"])
        contexto_extendido = contexto_estrofa_contenido(estrofas, idx_estrofa)
        estrofa_sin_contenido = sum(pesos_frases) <= 0
        if estrofa_sin_contenido:
            etiquetas_estrofa = []
            vector_estrofa_directo_crudo = {}
            vector_estrofa_directo = {}
            vector_contexto_crudo = {}
            vector_contexto_estrofas = {}
        else:
            etiquetas_estrofa = clasificar_segmento(
                classifier,
                texto_estrofa_contenido,
                texto_estrofa_contenido,
                threshold=0.0,
                top_k=top_k_estrofa,
            )
            vector_estrofa_directo_crudo = vector_desde_etiquetas(etiquetas_estrofa, normalizar=True)
            vector_estrofa_directo = ajustar_vector_coherencia_contextual(
                vector_estrofa_directo_crudo,
                texto_estrofa_contenido,
                texto_estrofa_contenido,
                nivel="estrofa",
            )
            etiquetas_contexto = clasificar_segmento(
                classifier,
                contexto_extendido,
                contexto_extendido,
                threshold=0.0,
                top_k=top_k_estrofa,
            )
            vector_contexto_crudo = vector_desde_etiquetas(etiquetas_contexto, normalizar=True)
            vector_contexto_estrofas = ajustar_vector_coherencia_contextual(
                vector_contexto_crudo,
                texto_estrofa_contenido,
                contexto_extendido,
                nivel="estrofa_vecina",
            )
        contradiccion_frases = hay_contradiccion_frases(vectores_frases, pesos_frases)
        resolucion_positiva = detectar_resolucion_positiva(vectores_frases, pesos_frases)
        if resolucion_positiva and cierre_invalido_por_conflicto(estrofa["frases"], pesos_frases):
            resolucion_positiva = False
        ambientacion = detectar_ambientacion(tipos_frases, pesos_frases, vectores_frases)
        modo_conflicto = detectar_modo_conflicto_relacional(texto_estrofa_contenido, contexto_extendido)
        modo_devastacion = detectar_modo_devastacion_interna(texto_estrofa_contenido, contexto_extendido)
        modo_resiliencia = detectar_modo_resiliencia_amorosa(texto_estrofa_contenido, contexto_extendido)
        modo_protesta = detectar_modo_protesta_social(texto_estrofa_contenido, contexto_extendido)
        modo_paradoja = detectar_paradoja_placer_danio(texto_estrofa_contenido, contexto_extendido)
        calidad_textual = calcular_calidad_textual_estrofa(estrofa["frases"])
        funciones_narrativas = detectar_funcion_narrativa_estrofa(
            texto_estrofa_contenido,
            estrofa["frases"],
            contexto_extendido,
        )
        peso_emocional = peso_emocional_por_funcion(funciones_narrativas, calidad_textual)
        aporte_global = aporte_global_por_funcion(funciones_narrativas)
        if (
            modo_conflicto != "no_conflictivo"
            or modo_devastacion != "no_devastacion"
            or modo_protesta != "no_protesta"
            or modo_paradoja != "no_paradoja"
            or any(
                f in funciones_narrativas
                for f in {
                    "advertencia_afectiva", "autoconciencia_problematica",
                    "enumeracion_relacional", "alarde_ego", "jerga_fragmentada",
                    "baja_densidad_semantica", "no_compromiso", "deseo_sin_apego",
                }
            )
        ):
            resolucion_positiva = False
        pesos_combinacion = (
            [peso_frases * 0.55, peso_estrofa_completa * 1.20, peso_contexto_estrofas * 1.25]
            if contradiccion_frases
            else [peso_frases, peso_estrofa_completa, peso_contexto_estrofas]
        )
        vector_estrofa = combinar_vectores(
            [vector_promedio_frases, vector_estrofa_directo, vector_contexto_estrofas],
            pesos_combinacion,
        )
        vector_estrofa = ajustar_vector_coherencia_contextual(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_orientacion_corporal(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            vector_contexto_estrofas,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_conflicto_narrativo(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_devastacion_interna(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_resiliencia_amorosa(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_protesta_social(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_paradoja_placer_danio(
            vector_estrofa,
            texto_estrofa_contenido,
            contexto_extendido,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_funcion_narrativa(
            vector_estrofa,
            funciones_narrativas,
            calidad_textual,
            nivel="estrofa_vecina",
        )
        vector_estrofa = ajustar_por_cierre_semantico(
            vector_estrofa,
            vectores_frases,
            pesos_frases,
            tipos_frases,
        )
        vector_estrofa = ajustar_por_contexto_romantico_global(
            vector_estrofa,
            vector_contexto_estrofas,
        )
        if not any(
            f in funciones_narrativas
            for f in {"enumeracion_relacional", "alarde_ego", "no_compromiso", "deseo_sin_apego", "advertencia_afectiva", "autoconciencia_problematica"}
        ):
            vector_estrofa = reforzar_por_mayoria_romantica(
                vector_estrofa,
                vectores_frases,
                estrofa["texto"],
            )
        vector_estrofa = normalizar_vector(vector_estrofa)
        orientacion_corporal_estrofa = detectar_orientacion_corporal(
            f"{texto_estrofa_contenido} {contexto_extendido}",
            vector_estrofa,
            vector_contexto_estrofas,
        )
        if orientacion_corporal_estrofa in {"sensual_positivo", "sensual_ambiguo"}:
            if "autoconciencia_problematica" in funciones_narrativas:
                orientacion_corporal_estrofa = "deseo_con_autocritica"
            elif "alarde_ego" in funciones_narrativas:
                orientacion_corporal_estrofa = "deseo_con_alarde"
            elif "no_compromiso" in funciones_narrativas or "deseo_sin_apego" in funciones_narrativas:
                orientacion_corporal_estrofa = "deseo_sin_apego"
        vectores_estrofa.append({
            "estrofa": estrofa["estrofa"],
            "texto": estrofa["texto"],
            "frases": calcular_peso_estrofa_resumen(
                pesos_frases,
                tipos_frases,
                modo_conflicto,
                modo_devastacion,
                modo_protesta,
                modo_paradoja,
                funciones_narrativas,
                calidad_textual,
            ),
            "vector": vector_estrofa,
            "vector_directo": vector_estrofa_directo,
            "vector_frases": vector_promedio_frases,
            "vector_contexto": vector_contexto_estrofas,
            "modo_conflicto": modo_conflicto,
            "modo_devastacion": modo_devastacion,
            "modo_resiliencia": modo_resiliencia,
            "modo_protesta": modo_protesta,
            "modo_paradoja": modo_paradoja,
            "funciones_narrativas": funciones_narrativas,
            "aporte_global": aporte_global,
            "calidad_textual": calidad_textual,
            "peso_emocional": peso_emocional,
        })

        top_principal = top_vector(vector_estrofa, 1)[0] if vector_estrofa else ("sin_etiqueta", 0.0)
        estado_semantico, lectura_sugerida = evaluar_estado_semantico(
            vector_estrofa,
            modo_resiliencia,
            modo_paradoja,
            funciones_narrativas,
            calidad_textual,
        )
        dominante_salida = top_principal[0]
        dominante_pct_salida = round(top_principal[1] * 100, 4)
        filas_estrofas.append({
            "estrofa": estrofa["estrofa"],
            "frases": len(estrofa["frases"]),
            "peso_frases": round(sum(pesos_frases), 4),
            "contradiccion_frases": int(contradiccion_frases),
            "resolucion_positiva": int(resolucion_positiva),
            "ambientacion": int(ambientacion),
            "modo_conflicto": modo_conflicto,
            "modo_devastacion": modo_devastacion,
            "modo_resiliencia": modo_resiliencia,
            "modo_protesta": modo_protesta,
            "modo_paradoja": modo_paradoja,
            "funcion_narrativa": funcion_narrativa_a_texto(funciones_narrativas),
            "aporte_global": aporte_global,
            "calidad_textual": calidad_textual.get("nivel", "alta"),
            "calidad_score": calidad_textual.get("score", 1.0),
            "peso_emocional": peso_emocional,
            "estado_semantico": estado_semantico,
            "lectura_sugerida": lectura_sugerida,
            "orientacion_corporal": orientacion_corporal_estrofa,
            "texto": estrofa["texto"],
            "texto_contenido": texto_estrofa_contenido,
            "dominante": dominante_salida,
            "dominante_pct": dominante_pct_salida,
            "top_combinado": vector_a_texto(vector_estrofa, top_k_estrofa),
            "top_estrofa_completa": vector_a_texto(vector_estrofa_directo, top_k_estrofa),
            "top_promedio_frases": vector_a_texto(vector_promedio_frases, top_k_estrofa),
            "top_contexto_estrofas": vector_a_texto(vector_contexto_estrofas, top_k_estrofa),
            "top_estrofa_completa_pre_filtro": vector_a_texto(vector_estrofa_directo_crudo, top_k_estrofa),
            "top_contexto_estrofas_pre_filtro": vector_a_texto(vector_contexto_crudo, top_k_estrofa),
            "contexto_estrofas": contexto_extendido,
        })

    # Modo escucha: las estrofas pesan por cantidad de frases, por lo que repeticion cuenta.
    vector_cancion_escucha = combinar_vectores(
        [item["vector"] for item in vectores_estrofa],
        [item["frases"] for item in vectores_estrofa],
    )
    vector_cancion_escucha = ajustar_vector_cancion_por_coherencia(normalizar_vector(vector_cancion_escucha))
    vector_cancion_escucha = ajustar_vector_cancion_por_modo_narrativo(vector_cancion_escucha, vectores_estrofa)
    vector_cancion_escucha = ajustar_vector_cancion_por_modo_devastacion(vector_cancion_escucha, vectores_estrofa)
    vector_cancion_escucha = ajustar_vector_cancion_por_modo_resiliencia(vector_cancion_escucha, vectores_estrofa)
    vector_cancion_escucha = ajustar_vector_cancion_por_modo_protesta(vector_cancion_escucha, vectores_estrofa)
    vector_cancion_escucha = ajustar_vector_cancion_por_modo_paradoja(vector_cancion_escucha, vectores_estrofa)
    vector_cancion_escucha = ajustar_vector_cancion_por_funcion_narrativa(vector_cancion_escucha, vectores_estrofa)

    # Modo narrativo: estrofas duplicadas exactas pesan una sola vez.
    vistos = set()
    vectores_unicos = []
    for item in vectores_estrofa:
        firma = " ".join(item["texto"].lower().split())
        if firma in vistos:
            continue
        vistos.add(firma)
        vectores_unicos.append(item)
    vector_cancion_narrativo = ajustar_vector_cancion_por_coherencia(
        normalizar_vector(combinar_vectores([item["vector"] for item in vectores_unicos], [item["frases"] for item in vectores_unicos]))
    )
    vector_cancion_narrativo = ajustar_vector_cancion_por_modo_narrativo(vector_cancion_narrativo, vectores_unicos)
    vector_cancion_narrativo = ajustar_vector_cancion_por_modo_devastacion(vector_cancion_narrativo, vectores_unicos)
    vector_cancion_narrativo = ajustar_vector_cancion_por_modo_resiliencia(vector_cancion_narrativo, vectores_unicos)
    vector_cancion_narrativo = ajustar_vector_cancion_por_modo_protesta(vector_cancion_narrativo, vectores_unicos)
    vector_cancion_narrativo = ajustar_vector_cancion_por_modo_paradoja(vector_cancion_narrativo, vectores_unicos)
    vector_cancion_narrativo = ajustar_vector_cancion_por_funcion_narrativa(vector_cancion_narrativo, vectores_unicos)

    filas_cancion = []
    for modo, vector, items_modo in [
        ("escucha", vector_cancion_escucha, vectores_estrofa),
        ("narrativo", vector_cancion_narrativo, vectores_unicos),
    ]:
        top_lista = top_vector(vector, 3)
        top_principal = top_lista[0] if top_lista else ("sin_etiqueta", 0.0)
        modo_paradoja_global = modo_paradoja_dominante(items_modo)
        resumen_narrativo = agregar_funciones_narrativas_cancion(items_modo)
        modo_narrativo_global = resumen_narrativo.get("modo_dominante", "no_definido")
        estado_semantico, lectura_sugerida = evaluar_estado_semantico(
            vector,
            modo_paradoja=modo_paradoja_global,
        )
        margen = (top_lista[0][1] - top_lista[1][1]) if len(top_lista) > 1 else 1.0
        dominante_cancion = top_principal[0]
        dominante_pct_cancion = round(top_principal[1] * 100, 4)
        emocion_top_real = top_principal[0]
        emocion_top_real_pct = round(top_principal[1] * 100, 4)
        if top_principal[1] < 0.12 or margen < 0.025:
            dominante_cancion = "sin_dominante_clara"
            dominante_pct_cancion = 0.0
            estado_semantico = "mixto_baja_confianza"
            if not lectura_sugerida:
                lectura_sugerida = "perfil emocional asociado sin dominante fuerte"
        filas_cancion.append({
            "modo": modo,
            "dominante": dominante_cancion,
            "dominante_pct": dominante_pct_cancion,
            "emocion_top_real": emocion_top_real,
            "emocion_top_real_pct": emocion_top_real_pct,
            "modo_narrativo": modo_narrativo_global,
            "submodos_narrativos": resumen_narrativo.get("submodos", ""),
            "perfil_narrativo": resumen_narrativo.get("perfil", ""),
            "lectura_narrativa": resumen_narrativo.get("lectura_sugerida", ""),
            "modo_paradoja": modo_paradoja_global,
            "estado_semantico": estado_semantico,
            "lectura_sugerida": lectura_sugerida,
            "top_emociones": vector_a_texto(vector, 8),
        })

    guardar_csv(
        os.path.join(out_dir, "frases_clasificadas.csv"),
        filas_frases,
        [
            "estrofa", "frase", "texto", "contexto", "tipo_frase", "peso_frase",
            "orientacion_corporal", "dominante", "dominante_pct", "top_emociones", "top_pre_filtro",
            "etiquetas_crudas",
        ],
    )
    guardar_csv(
        os.path.join(out_dir, "estrofas_resumen.csv"),
        filas_estrofas,
        [
            "estrofa", "frases", "peso_frases", "contradiccion_frases",
            "resolucion_positiva", "ambientacion", "modo_conflicto", "modo_devastacion",
            "modo_resiliencia", "modo_protesta", "modo_paradoja",
            "funcion_narrativa", "aporte_global", "calidad_textual", "calidad_score", "peso_emocional",
            "estado_semantico", "lectura_sugerida", "orientacion_corporal",
            "texto", "texto_contenido", "dominante", "dominante_pct",
            "top_combinado", "top_estrofa_completa", "top_promedio_frases",
            "top_contexto_estrofas", "top_estrofa_completa_pre_filtro",
            "top_contexto_estrofas_pre_filtro", "contexto_estrofas",
        ],
    )
    guardar_csv(
        os.path.join(out_dir, "cancion_resumen.csv"),
        filas_cancion,
        [
            "modo", "dominante", "dominante_pct", "modo_narrativo", "submodos_narrativos",
            "perfil_narrativo", "lectura_narrativa", "modo_paradoja", "estado_semantico",
            "lectura_sugerida", "emocion_top_real", "emocion_top_real_pct", "top_emociones",
        ],
    )
    guardar_reporte(
        out_dir,
        archivo_txt,
        lineas,
        filas_frases,
        filas_estrofas,
        filas_cancion,
        labels,
        frases_por_estrofa,
        peso_estrofa_completa,
        peso_frases,
        peso_contexto_estrofas,
    )
    return filas_frases, filas_estrofas, filas_cancion


def guardar_csv(path: str, filas: list[dict], campos: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in filas:
            writer.writerow({campo: fila.get(campo, "") for campo in campos})


def guardar_reporte(
        out_dir: str,
        archivo_txt: str,
        lineas: list[str],
        filas_frases: list[dict],
        filas_estrofas: list[dict],
        filas_cancion: list[dict],
        labels: dict[str, str],
        frases_por_estrofa: int,
        peso_estrofa_completa: float,
        peso_frases: float,
        peso_contexto_estrofas: float):
    path = os.path.join(out_dir, "reporte_jerarquico.txt")
    with open(path, "w", encoding="utf-8") as archivo:
        archivo.write("ANALISIS JERARQUICO GOEMOTIONS ADAPTADO\n")
        archivo.write(f"Archivo: {os.path.abspath(archivo_txt)}\n")
        archivo.write(f"Frases detectadas: {len(lineas)}\n")
        archivo.write(f"Frases por estrofa: {frases_por_estrofa}\n")
        archivo.write(f"Peso estrofa completa: {peso_estrofa_completa:.2f}\n")
        archivo.write(f"Peso promedio de frases: {peso_frases:.2f}\n")
        archivo.write(f"Peso contexto estrofas vecinas: {peso_contexto_estrofas:.2f}\n\n")

        archivo.write("RESUMEN EMOCIONAL\n")
        for fila in filas_cancion:
            archivo.write(f"- {fila['modo']}:\n")
            archivo.write(f"  Estado: {fila.get('estado_semantico', 'definido')}\n")
            archivo.write(
                f"  Emocion dominante: {fila['dominante']}"
                f"{f' ({fila['dominante_pct']:.2f}%)' if fila['dominante'] != 'sin_dominante_clara' else ''}\n"
            )
            if fila["dominante"] == "sin_dominante_clara":
                archivo.write(
                    f"  Emocion con mayor peso real: {fila.get('emocion_top_real', '')} "
                    f"({fila.get('emocion_top_real_pct', 0.0):.2f}%)\n"
                )
            archivo.write(f"  Perfil emocional: {fila['top_emociones']}\n")
            if fila.get("lectura_sugerida"):
                archivo.write(f"  Lectura emocional: {fila['lectura_sugerida']}\n")
        archivo.write("\n")

        archivo.write("RESUMEN NARRATIVO\n")
        for fila in filas_cancion:
            archivo.write(
                f"- {fila['modo']}: modo dominante={fila.get('modo_narrativo', 'no_definido')}"
                f" | submodos={fila.get('submodos_narrativos', '') or 'sin_submodos'}"
                f" | perfil={fila.get('perfil_narrativo', '') or 'sin_perfil'}\n"
            )
            if fila.get("lectura_narrativa"):
                archivo.write(f"  Lectura sugerida: {fila['lectura_narrativa']}\n")
        archivo.write("\n")

        archivo.write("ESTROFAS\n")
        for estrofa in filas_estrofas:
            archivo.write(
                f"\nEstrofa {estrofa['estrofa']} -> {estrofa['dominante']} "
                f"({estrofa['dominante_pct']:.2f}%)\n"
            )
            archivo.write(f"Emocion principal: {estrofa['dominante']} ({estrofa['dominante_pct']:.2f}%)\n")
            archivo.write(f"Top combinado: {estrofa['top_combinado']}\n")
            archivo.write(f"Top contexto estrofas vecinas: {estrofa['top_contexto_estrofas']}\n")
            archivo.write(
                f"Peso frases utiles: {estrofa.get('peso_frases', '')} | "
                f"Contradiccion interna: {estrofa.get('contradiccion_frases', 0)} | "
                f"Cierre positivo: {estrofa.get('resolucion_positiva', 0)} | "
                f"Ambientacion: {estrofa.get('ambientacion', 0)} | "
                f"Conflicto: {estrofa.get('modo_conflicto', 'no_conflictivo')} | "
                f"Devastacion: {estrofa.get('modo_devastacion', 'no_devastacion')} | "
                f"Resiliencia: {estrofa.get('modo_resiliencia', 'no_resiliencia')} | "
                f"Protesta: {estrofa.get('modo_protesta', 'no_protesta')} | "
                f"Paradoja: {estrofa.get('modo_paradoja', 'no_paradoja')} | "
                f"Funcion: {estrofa.get('funcion_narrativa', 'emocion_directa')} | "
                f"Aporte: {estrofa.get('aporte_global', 'emocion_directa')} | "
                f"Calidad: {estrofa.get('calidad_textual', 'alta')} ({estrofa.get('calidad_score', 1.0)}) | "
                f"Peso emocional: {estrofa.get('peso_emocional', 'normal')} | "
                f"Estado: {estrofa.get('estado_semantico', 'definido')} | "
                f"Orientacion corporal: {estrofa.get('orientacion_corporal', 'no_sensual')}\n"
            )
            if estrofa.get("lectura_sugerida"):
                archivo.write(f"Lectura sugerida: {estrofa['lectura_sugerida']}\n")
            archivo.write(f"Texto: {estrofa['texto']}\n")
            if estrofa.get("texto_contenido") and estrofa["texto_contenido"] != estrofa["texto"]:
                archivo.write(f"Texto con contenido: {estrofa['texto_contenido']}\n")

            frases = [fila for fila in filas_frases if fila["estrofa"] == estrofa["estrofa"]]
            for frase in frases:
                archivo.write(
                    f"  [{frase['frase']}] {frase['dominante']} "
                    f"({frase['dominante_pct']:.2f}%, tipo={frase.get('tipo_frase')}, "
                    f"peso={frase.get('peso_frase')}, corporal={frase.get('orientacion_corporal')}): "
                    f"{frase['texto']}\n"
                )

        archivo.write("\nLEYENDA TOP LABELS\n")
        for clave, label in sorted(labels.items()):
            archivo.write(f"{clave}: {label}\n")

    with open(os.path.join(out_dir, "parametros.txt"), "w", encoding="utf-8") as archivo:
        archivo.write(f"archivo_txt={os.path.abspath(archivo_txt)}\n")
        archivo.write(f"frases_detectadas={len(lineas)}\n")
        archivo.write(f"frases_por_estrofa={frases_por_estrofa}\n")
        archivo.write(f"peso_estrofa_completa={peso_estrofa_completa}\n")
        archivo.write(f"peso_promedio_frases={peso_frases}\n")
        archivo.write(f"peso_contexto_estrofas={peso_contexto_estrofas}\n")


def main(argv: Iterable[str] = sys.argv[1:]):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)

    print("=== Prueba aislada: clasificacion jerarquica frase -> estrofa -> cancion ===")
    print(f"Archivo: {os.path.abspath(args.archivo_txt)}")
    print(f"Salida: {out_dir}")
    print(f"Frases por estrofa: {args.frases_por_estrofa}\n")

    filas_frases, filas_estrofas, filas_cancion = analizar_jerarquico(
        archivo_txt=args.archivo_txt,
        out_dir=out_dir,
        modelo=args.modelo,
        frases_por_estrofa=args.frases_por_estrofa,
        top_k_frase=args.top_k_frase,
        top_k_estrofa=args.top_k_estrofa,
        peso_estrofa_completa=args.peso_estrofa_completa,
        peso_promedio_frases=args.peso_promedio_frases,
        peso_contexto_estrofas=args.peso_contexto_estrofas,
    )

    print("\n=== Resumen ===")
    print(f"Frases clasificadas: {len(filas_frases)}")
    print(f"Estrofas clasificadas: {len(filas_estrofas)}")
    for fila in filas_cancion:
        print(
            f"Cancion ({fila['modo']}): {fila['dominante']} "
            f"| Estado: {fila.get('estado_semantico', 'definido')} "
            f"| Top real: {fila.get('emocion_top_real', fila['dominante'])} "
            f"({fila.get('emocion_top_real_pct', fila['dominante_pct']):.2f}%) "
            f"| Narrativa: {fila.get('modo_narrativo', 'no_definido')} "
            f"({fila.get('submodos_narrativos', '')}) | "
            f"{fila['top_emociones']}"
        )
    print(f"\nRevisa: {os.path.join(out_dir, 'reporte_jerarquico.txt')}")


if __name__ == "__main__":
    main()
