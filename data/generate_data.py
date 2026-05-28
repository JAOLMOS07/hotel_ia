"""
FASE 1: Generación y Preprocesamiento de Datos
Sistema Inteligente de Monitoreo y Predicción de Ocupación Turística
Valledupar - Cesar - Colombia

Este módulo genera datos sintéticos basados en los patrones reales documentados:
- Cotelco Cesar: 98% ocupación en festival vs 38% en temporada baja
- Festival Leyenda Vallenata: último fin de semana de abril / primera semana de mayo
- Fuentes: DANE-EMA, EGIT, RNT-Valledupar, Cámara de Comercio Valledupar
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Semilla para reproducibilidad
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE EVENTOS Y TEMPORADAS (basada en datos reales)
# ─────────────────────────────────────────────────────────────────

EVENTOS_VALLEDUPAR = {
    # Festival de la Leyenda Vallenata (último fin de semana abril - 1ra semana mayo)
    "festival_vallenato": {
        "meses": [4, 5],
        "dias_mes": list(range(26, 32)) + list(range(1, 7)),
        "boost_ocupacion": 0.60,   # +60 pp → llega al 98%
        "descripcion": "Festival de la Leyenda Vallenata"
    },
    # Semana Santa
    "semana_santa": {
        "meses": [3, 4],
        "boost_ocupacion": 0.25,
        "descripcion": "Semana Santa"
    },
    # Fiestas de fin de año
    "fin_anio": {
        "meses": [12],
        "dias_mes": list(range(20, 32)),
        "boost_ocupacion": 0.20,
        "descripcion": "Temporada Navidad/Año Nuevo"
    },
    # Vacaciones junio-julio (sistema educativo colombiano)
    "vacaciones_mitad_anio": {
        "meses": [6, 7],
        "boost_ocupacion": 0.10,
        "descripcion": "Vacaciones mitad de año"
    }
}

PUENTES_FESTIVOS_COLOMBIA = [
    # Formato (mes, dia) - festivos nacionales relevantes
    (1, 1), (1, 6), (3, 25), (5, 1), (6, 16), (6, 23),
    (7, 4), (7, 20), (8, 7), (8, 18), (10, 13), (11, 3),
    (11, 17), (12, 8), (12, 25)
]

# ─────────────────────────────────────────────────────────────────
# 1. DATASET DE OCUPACIÓN HOTELERA (Series Temporales para LSTM)
#    Basado en DANE-EMA y Cotelco Capítulo Cesar
# ─────────────────────────────────────────────────────────────────

def generar_serie_ocupacion(anio_inicio=2015, anio_fin=2025):
    """
    Genera serie histórica mensual de ocupación hotelera en Valledupar.
    Patrón base: 38% temporada baja → 98% festival (Cotelco Cesar, 2023)
    """
    fechas = pd.date_range(
        start=f"{anio_inicio}-01-01",
        end=f"{anio_fin}-12-31",
        freq="D"
    )

    registros = []

    for fecha in fechas:
        mes = fecha.month
        dia = fecha.day
        anio = fecha.year
        dia_semana = fecha.weekday()  # 0=lunes, 6=domingo

        # Tasa base según mes (estacionalidad anual documentada)
        tasas_base_mensual = {
            1: 0.45, 2: 0.42, 3: 0.48,  # Ene-Mar: moderado
            4: 0.55, 5: 0.52, 6: 0.48,  # Abr-Jun: festival y semana santa
            7: 0.50, 8: 0.38, 9: 0.36,  # Jul-Sep: temporada baja
            10: 0.38, 11: 0.42, 12: 0.55  # Oct-Dic: recuperación
        }
        tasa_base = tasas_base_mensual[mes]

        # Efecto fin de semana (+5% a +8%)
        if dia_semana >= 5:
            tasa_base += np.random.uniform(0.05, 0.08)

        # Efecto Festival de la Leyenda Vallenata
        es_festival = False
        if mes == 4 and dia >= 26:
            es_festival = True
            tasa_base = np.random.uniform(0.92, 0.99)  # 98% promedio real
        elif mes == 5 and dia <= 6:
            es_festival = True
            tasa_base = np.random.uniform(0.88, 0.99)

        # Efecto Semana Santa (variable según año)
        if mes == 3 and dia >= 20 and dia <= 31:
            tasa_base = min(tasa_base + 0.25, 0.95)
        elif mes == 4 and dia <= 6:
            tasa_base = min(tasa_base + 0.20, 0.90)

        # Efecto puentes festivos (+10% a +15%)
        es_festivo = any(m == mes and d == dia for m, d in PUENTES_FESTIVOS_COLOMBIA)
        if es_festivo:
            tasa_base = min(tasa_base + np.random.uniform(0.10, 0.15), 0.98)

        # Efecto vacaciones mitad año
        if mes in [6, 7]:
            tasa_base = min(tasa_base + 0.08, 0.80)

        # Tendencia de crecimiento turístico (post-UNESCO 2015: +2% anual)
        factor_tendencia = 1 + (anio - 2015) * 0.018
        tasa_base = min(tasa_base * factor_tendencia, 0.99)

        # Ruido aleatorio realista (±3%)
        ruido = np.random.normal(0, 0.025)
        tasa_final = max(0.20, min(0.99, tasa_base + ruido))

        # Ingresos estimados (correlacionados con ocupación)
        # Base: hotel promedio 328 establecimientos, ~30 hab c/u, tarifa ~$180,000 COP
        capacidad_estimada = 328 * 30
        ingresos_millones = (tasa_final * capacidad_estimada * 180_000) / 1_000_000
        ingresos_millones += np.random.normal(0, ingresos_millones * 0.05)

        registros.append({
            "fecha": fecha,
            "anio": anio,
            "mes": mes,
            "dia": dia,
            "dia_semana": dia_semana,
            "nombre_dia": fecha.strftime("%A"),
            "tasa_ocupacion": round(tasa_final, 4),
            "tasa_ocupacion_pct": round(tasa_final * 100, 2),
            "es_festival_vallenato": int(es_festival),
            "es_festivo_nacional": int(es_festivo),
            "es_fin_semana": int(dia_semana >= 5),
            "temporada": clasificar_temporada(mes, dia, es_festival),
            "ingresos_millones_cop": round(max(0, ingresos_millones), 2)
        })

    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def clasificar_temporada(mes, dia, es_festival):
    """Clasifica el período en temporada según patrones de Valledupar"""
    if es_festival:
        return "FESTIVAL_VALLENATO"
    elif mes == 12 and dia >= 20:
        return "ALTA_NAVIDENA"
    elif mes in [1] and dia <= 8:
        return "ALTA_NAVIDENA"
    elif mes == 3 and dia >= 20:
        return "SEMANA_SANTA"
    elif mes == 4 and dia <= 6:
        return "SEMANA_SANTA"
    elif mes in [6, 7]:
        return "VACACIONES_MITAD_ANIO"
    elif mes in [8, 9, 10]:
        return "BAJA"
    else:
        return "MEDIA"


# ─────────────────────────────────────────────────────────────────
# 2. DATASET DE PERFILES DE TURISTAS (para K-Means)
#    Basado en EGIT-DANE 2023/2024 y estudio GITUD-UPC 2020
# ─────────────────────────────────────────────────────────────────

SEGMENTOS_REALES_GITUD = {
    # Basado en: GITUD-UPC (2020) + EGIT-DANE (2023-2024)
    "turista_cultural_tradicional": {
        "proporcion": 0.32,
        "edad_media": 42, "edad_std": 10,
        "estancia_media": 4.5, "estancia_std": 1.5,
        "gasto_medio": 2_800_000, "gasto_std": 600_000,
        "origen_bogota": 0.15, "origen_costa": 0.60, "origen_interior": 0.25,
        "tipo_alojamiento_pref": "hotel_3_estrellas"
    },
    "visitante_extranjero": {
        "proporcion": 0.08,
        "edad_media": 38, "edad_std": 12,
        "estancia_media": 6.0, "estancia_std": 2.0,
        "gasto_medio": 5_200_000, "gasto_std": 1_500_000,
        "origen_bogota": 0.05, "origen_costa": 0.10, "origen_interior": 0.05,
        "tipo_alojamiento_pref": "hotel_4_5_estrellas"
    },
    "turista_familiar": {
        "proporcion": 0.28,
        "edad_media": 36, "edad_std": 8,
        "estancia_media": 3.5, "estancia_std": 1.2,
        "gasto_medio": 3_500_000, "gasto_std": 800_000,
        "origen_bogota": 0.20, "origen_costa": 0.55, "origen_interior": 0.25,
        "tipo_alojamiento_pref": "hostal_familiar"
    },
    "joven_aficionado_vallenato": {
        "proporcion": 0.22,
        "edad_media": 24, "edad_std": 5,
        "estancia_media": 2.5, "estancia_std": 1.0,
        "gasto_medio": 1_200_000, "gasto_std": 400_000,
        "origen_bogota": 0.30, "origen_costa": 0.45, "origen_interior": 0.25,
        "tipo_alojamiento_pref": "hostal_economico"
    },
    "turista_negocios_cultural": {
        "proporcion": 0.10,
        "edad_media": 45, "edad_std": 9,
        "estancia_media": 2.0, "estancia_std": 0.8,
        "gasto_medio": 4_100_000, "gasto_std": 900_000,
        "origen_bogota": 0.50, "origen_costa": 0.25, "origen_interior": 0.25,
        "tipo_alojamiento_pref": "hotel_4_5_estrellas"
    }
}

DEPARTAMENTOS_COLOMBIA = [
    "Cesar", "Bolívar", "Atlántico", "Magdalena", "Guajira",
    "Bogotá D.C.", "Cundinamarca", "Antioquia", "Valle del Cauca",
    "Santander", "Norte de Santander", "Tolima", "Huila",
    "Córdoba", "Sucre", "Extranjero"
]

ACTIVIDADES_TURISTICAS = [
    "conciertos_vallenato", "visita_monumentos", "gastronomia_local",
    "artesanias", "turismo_naturaleza", "eventos_culturales",
    "negocios", "visita_museos", "fotografia", "descanso_familiar"
]

TIPOS_ALOJAMIENTO = [
    "hotel_5_estrellas", "hotel_4_estrellas", "hotel_3_estrellas",
    "hotel_2_estrellas", "hostal_familiar", "hostal_economico",
    "apartahotel", "casa_huespedes", "airbnb_particular"
]


def generar_perfiles_turistas(n_turistas=5000):
    """
    Genera perfiles de turistas con 15 variables para segmentación K-Means.
    Basado en EGIT-DANE 2023/2024 y estudio GITUD-UPC 2020.
    """
    registros = []
    segmentos = list(SEGMENTOS_REALES_GITUD.keys())
    proporciones = [SEGMENTOS_REALES_GITUD[s]["proporcion"] for s in segmentos]

    for i in range(n_turistas):
        # Asignar segmento real (para validación posterior)
        segmento_real = np.random.choice(segmentos, p=proporciones)
        params = SEGMENTOS_REALES_GITUD[segmento_real]

        # Variables demográficas
        edad = max(18, min(80, int(np.random.normal(params["edad_media"], params["edad_std"]))))
        sexo = np.random.choice(["M", "F"], p=[0.52, 0.48])
        nivel_educativo = np.random.choice(
            ["primaria", "secundaria", "tecnico", "universitario", "posgrado"],
            p=[0.05, 0.18, 0.22, 0.38, 0.17]
        )

        # Origen (departamento)
        r = np.random.random()
        if r < params["origen_costa"]:
            departamento = np.random.choice(
                ["Cesar", "Bolívar", "Atlántico", "Magdalena", "Guajira", "Córdoba", "Sucre"],
                p=[0.30, 0.20, 0.18, 0.12, 0.08, 0.07, 0.05]
            )
        elif r < params["origen_costa"] + params["origen_bogota"]:
            departamento = "Bogotá D.C."
        else:
            departamento = np.random.choice(
                ["Antioquia", "Valle del Cauca", "Santander", "Cundinamarca", "Tolima", "Extranjero"],
                p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
            )

        # Variables de comportamiento de viaje
        dias_estancia = max(1, round(np.random.normal(params["estancia_media"], params["estancia_std"])))
        gasto_total = max(200_000, round(np.random.normal(params["gasto_medio"], params["gasto_std"]) / 1000) * 1000)
        gasto_alojamiento_pct = np.random.uniform(0.30, 0.50)

        # Tipo de alojamiento (influenciado por segmento)
        pref = params["tipo_alojamiento_pref"]
        pesos_aloj = {t: 0.05 for t in TIPOS_ALOJAMIENTO}
        pesos_aloj[pref] = 0.50
        # normalizar
        total_p = sum(pesos_aloj.values())
        tipos_list = list(pesos_aloj.keys())
        pesos_list = [pesos_aloj[t] / total_p for t in tipos_list]
        tipo_alojamiento = np.random.choice(tipos_list, p=pesos_list)

        # Motivo principal del viaje
        if segmento_real == "turista_negocios_cultural":
            motivo = "negocios"
        elif segmento_real == "turista_cultural_tradicional":
            motivo = np.random.choice(["festival", "cultura", "ocio"], p=[0.65, 0.25, 0.10])
        elif segmento_real == "joven_aficionado_vallenato":
            motivo = np.random.choice(["festival", "ocio", "visita_familiar"], p=[0.75, 0.15, 0.10])
        else:
            motivo = np.random.choice(["festival", "ocio", "visita_familiar", "cultura"], p=[0.40, 0.30, 0.20, 0.10])

        # Satisfacción (1-5)
        satisfaccion_base = 4.0 if segmento_real == "visitante_extranjero" else 3.8
        satisfaccion = round(min(5.0, max(1.0, np.random.normal(satisfaccion_base, 0.7))), 1)

        # Actividades realizadas (multi-hot encoding → resumido como score 0-10)
        n_actividades = np.random.randint(2, 7)
        actividades_realizadas = np.random.choice(ACTIVIDADES_TURISTICAS, size=n_actividades, replace=False)
        score_cultural = sum(1 for a in actividades_realizadas if a in
                           ["conciertos_vallenato", "visita_monumentos", "eventos_culturales", "visita_museos"])

        # Variables numéricas para clustering
        es_primera_visita = int(np.random.random() < (0.55 if segmento_real == "joven_aficionado_vallenato" else 0.30))
        grupo_tamano = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.30, 0.20, 0.20, 0.10, 0.05])

        registros.append({
            "turista_id": f"TUR_{i+1:05d}",
            "segmento_real": segmento_real,  # Solo para validación
            # Variables demográficas
            "edad": edad,
            "sexo": sexo,
            "nivel_educativo": nivel_educativo,
            "departamento_origen": departamento,
            "es_extranjero": int(departamento == "Extranjero"),
            # Variables de viaje
            "dias_estancia": dias_estancia,
            "gasto_total_cop": gasto_total,
            "gasto_alojamiento_cop": round(gasto_total * gasto_alojamiento_pct / 1000) * 1000,
            "tipo_alojamiento": tipo_alojamiento,
            "motivo_viaje": motivo,
            "grupo_tamano": grupo_tamano,
            "es_primera_visita": es_primera_visita,
            # Variables de preferencia
            "score_actividad_cultural": score_cultural,
            "satisfaccion_destino": satisfaccion,
            "reservo_con_anticipacion": int(np.random.random() < 0.60),
            "uso_plataforma_digital": int(np.random.random() < 0.72)
        })

    return pd.DataFrame(registros)


# ─────────────────────────────────────────────────────────────────
# 3. CATÁLOGO DE ESTABLECIMIENTOS (para Sistema de Recomendación)
#    Basado en RNT-Valledupar: 328 establecimientos activos
# ─────────────────────────────────────────────────────────────────

ESTABLECIMIENTOS_REALES = [
    # Hoteles de alta categoría
    {"nombre": "Hotel Sicarare Valledupar", "tipo": "hotel_5_estrellas", "precio_noche": 450000},
    {"nombre": "Hotel Vajamar", "tipo": "hotel_4_estrellas", "precio_noche": 320000},
    {"nombre": "Hotel Tamacá Valledupar", "tipo": "hotel_4_estrellas", "precio_noche": 290000},
    {"nombre": "Hotel Provenzal", "tipo": "hotel_3_estrellas", "precio_noche": 220000},
    {"nombre": "Hotel Cacique Upar", "tipo": "hotel_3_estrellas", "precio_noche": 195000},
    {"nombre": "Hotel La Vallenata", "tipo": "hotel_3_estrellas", "precio_noche": 180000},
    {"nombre": "Apart-Hotel El Prado", "tipo": "apartahotel", "precio_noche": 240000},
    {"nombre": "Hostal Casa Grande", "tipo": "hostal_familiar", "precio_noche": 120000},
    {"nombre": "Hostal La Cañaguatera", "tipo": "hostal_familiar", "precio_noche": 110000},
    {"nombre": "Hostal Vallenato Inn", "tipo": "hostal_economico", "precio_noche": 80000},
]

ACTIVIDADES_CATALOGO = [
    {"nombre": "Festival de la Leyenda Vallenata", "categoria": "festival", "precio": 180000},
    {"nombre": "Tour Casa de Carlos Vives", "categoria": "cultura", "precio": 35000},
    {"nombre": "Visita al Parque de la Leyenda Vallenata", "categoria": "cultura", "precio": 25000},
    {"nombre": "Tour Gastronómico Centro Histórico", "categoria": "gastronomia", "precio": 85000},
    {"nombre": "Ruta Manaure - Sierra Nevada", "categoria": "naturaleza", "precio": 120000},
    {"nombre": "Taller de Acordeón Vallenato", "categoria": "cultura", "precio": 60000},
    {"nombre": "Visita Rancho La Cañahuate", "categoria": "naturaleza", "precio": 45000},
    {"nombre": "Degustación Gastronomía Vallenata", "categoria": "gastronomia", "precio": 55000},
    {"nombre": "Tour Fotográfico Valledupar", "categoria": "cultura", "precio": 70000},
    {"nombre": "Concierto de Vallenato Tradicional", "categoria": "festival", "precio": 95000},
]


def generar_catalogo_establecimientos(n_total=328):
    """
    Genera catálogo completo de 328 establecimientos (RNT-Valledupar)
    con los 10 establecimientos reales + establecimientos sintéticos.
    """
    registros = []

    # Primero los establecimientos reales documentados
    for i, est in enumerate(ESTABLECIMIENTOS_REALES):
        categoria_num = {"hotel_5_estrellas": 5, "hotel_4_estrellas": 4,
                        "hotel_3_estrellas": 3, "hotel_2_estrellas": 2,
                        "hostal_familiar": 2, "hostal_economico": 1,
                        "apartahotel": 3, "casa_huespedes": 1}.get(est["tipo"], 3)
        registros.append({
            "establecimiento_id": f"EST_{i+1:04d}",
            "nombre": est["nombre"],
            "tipo": est["tipo"],
            "categoria_estrellas": categoria_num,
            "precio_noche_promedio": est["precio_noche"],
            "capacidad_habitaciones": np.random.randint(20, 80),
            "puntuacion_promedio": round(np.random.uniform(3.8, 4.9), 1),
            "n_resenas": np.random.randint(50, 500),
            "zona": np.random.choice(["Centro Historico", "La Granja", "El Parque", "Las Delicias", "Los Almendros"]),
            "acepta_reservas_online": True,
            "tiene_restaurante": est["tipo"] in ["hotel_4_estrellas", "hotel_5_estrellas", "apartahotel"],
            "tiene_piscina": est["tipo"] in ["hotel_4_estrellas", "hotel_5_estrellas"],
            "tiene_estacionamiento": True,
            "permite_mascotas": np.random.random() < 0.25,
            "apto_familias": est["tipo"] in ["hostal_familiar", "hotel_3_estrellas", "hotel_4_estrellas", "hotel_5_estrellas"]
        })

    # Establecimientos adicionales sintéticos hasta completar 328
    tipos_dist = {
        "hotel_3_estrellas": 0.25, "hotel_2_estrellas": 0.20,
        "hostal_familiar": 0.20, "hostal_economico": 0.15,
        "casa_huespedes": 0.10, "apartahotel": 0.07,
        "hotel_4_estrellas": 0.02, "hotel_5_estrellas": 0.01
    }
    barrios = ["Centro Historico", "La Granja", "El Parque", "Las Delicias",
               "Los Almendros", "El Estadio", "Nueva Castilla", "Alfonso Lopez"]

    for i in range(len(ESTABLECIMIENTOS_REALES), n_total):
        tipo = np.random.choice(list(tipos_dist.keys()), p=list(tipos_dist.values()))
        cat = {"hotel_5_estrellas": 5, "hotel_4_estrellas": 4, "hotel_3_estrellas": 3,
               "hotel_2_estrellas": 2, "hostal_familiar": 2, "hostal_economico": 1,
               "apartahotel": 3, "casa_huespedes": 1}.get(tipo, 2)
        precio_base = {"hotel_5_estrellas": 420000, "hotel_4_estrellas": 280000,
                      "hotel_3_estrellas": 190000, "hotel_2_estrellas": 140000,
                      "hostal_familiar": 105000, "hostal_economico": 70000,
                      "apartahotel": 210000, "casa_huespedes": 65000}.get(tipo, 150000)

        registros.append({
            "establecimiento_id": f"EST_{i+1:04d}",
            "nombre": f"Alojamiento {tipo.replace('_', ' ').title()} #{i+1}",
            "tipo": tipo,
            "categoria_estrellas": cat,
            "precio_noche_promedio": round(precio_base * np.random.uniform(0.85, 1.20) / 1000) * 1000,
            "capacidad_habitaciones": np.random.randint(6, 60),
            "puntuacion_promedio": round(np.random.uniform(3.0, 4.8), 1),
            "n_resenas": np.random.randint(5, 200),
            "zona": np.random.choice(barrios),
            "acepta_reservas_online": np.random.random() < 0.65,
            "tiene_restaurante": np.random.random() < 0.30,
            "tiene_piscina": np.random.random() < 0.15,
            "tiene_estacionamiento": np.random.random() < 0.70,
            "permite_mascotas": np.random.random() < 0.20,
            "apto_familias": np.random.random() < 0.60
        })

    return pd.DataFrame(registros)


def generar_interacciones_usuarios(df_turistas, df_establecimientos, n_interacciones=15000):
    """
    Genera matriz de interacciones usuario-ítem para el sistema de recomendación.
    Simula reservas históricas y calificaciones.
    """
    registros = []
    turistas_ids = df_turistas["turista_id"].tolist()
    est_ids = df_establecimientos["establecimiento_id"].tolist()

    for _ in range(n_interacciones):
        turista_id = np.random.choice(turistas_ids)
        turista = df_turistas[df_turistas["turista_id"] == turista_id].iloc[0]

        # Preferencia por tipo de alojamiento según perfil
        tipo_pref = turista["tipo_alojamiento"]
        est_preferidos = df_establecimientos[df_establecimientos["tipo"] == tipo_pref]["establecimiento_id"].tolist()

        if est_preferidos and np.random.random() < 0.60:
            est_id = np.random.choice(est_preferidos)
        else:
            est_id = np.random.choice(est_ids)

        est = df_establecimientos[df_establecimientos["establecimiento_id"] == est_id].iloc[0]

        # Rating basado en satisfacción del turista y calidad del establecimiento
        rating_base = (turista["satisfaccion_destino"] + est["puntuacion_promedio"]) / 2
        rating = round(min(5.0, max(1.0, np.random.normal(rating_base, 0.5))), 1)

        registros.append({
            "turista_id": turista_id,
            "establecimiento_id": est_id,
            "rating": rating,
            "fecha_reserva": f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
            "noches": turista["dias_estancia"],
            "gasto_total": round(turista["gasto_alojamiento_cop"] / 1000) * 1000
        })

    return pd.DataFrame(registros).drop_duplicates(subset=["turista_id", "establecimiento_id"])


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def generar_todos_los_datasets(output_dir="outputs"):
    """Genera y guarda todos los datasets del sistema."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FASE 1: Generación de Datos - Sistema Turístico Valledupar")
    print("=" * 60)

    # 1. Serie de ocupación hotelera
    print("\n[1/4] Generando serie histórica de ocupación hotelera (2015-2025)...")
    df_ocupacion = generar_serie_ocupacion(2015, 2025)
    df_ocupacion.to_csv(f"{output_dir}/ocupacion_hotelera_valledupar.csv", index=False)
    print(f"      → {len(df_ocupacion):,} registros diarios generados")
    print(f"      → Ocupación promedio festival: {df_ocupacion[df_ocupacion['es_festival_vallenato']==1]['tasa_ocupacion_pct'].mean():.1f}%")
    print(f"      → Ocupación promedio temporada baja: {df_ocupacion[df_ocupacion['temporada']=='BAJA']['tasa_ocupacion_pct'].mean():.1f}%")

    # 2. Perfiles de turistas
    print("\n[2/4] Generando perfiles de turistas (EGIT-DANE + GITUD-UPC)...")
    df_turistas = generar_perfiles_turistas(5000)
    df_turistas.to_csv(f"{output_dir}/perfiles_turistas_valledupar.csv", index=False)
    print(f"      → {len(df_turistas):,} perfiles de turistas generados")
    print(f"      → Distribución por segmento:")
    for seg, cnt in df_turistas["segmento_real"].value_counts().items():
        print(f"         • {seg}: {cnt} ({cnt/len(df_turistas)*100:.1f}%)")

    # 3. Catálogo de establecimientos
    print("\n[3/4] Generando catálogo RNT Valledupar (328 establecimientos)...")
    df_establecimientos = generar_catalogo_establecimientos(328)
    df_establecimientos.to_csv(f"{output_dir}/establecimientos_rnt_valledupar.csv", index=False)
    print(f"      → {len(df_establecimientos)} establecimientos registrados")

    # 4. Interacciones usuario-ítem
    print("\n[4/4] Generando interacciones usuario-ítem...")
    df_interacciones = generar_interacciones_usuarios(df_turistas, df_establecimientos, 15000)
    df_interacciones.to_csv(f"{output_dir}/interacciones_reservas.csv", index=False)
    print(f"      → {len(df_interacciones):,} interacciones únicas generadas")

    print("\n✓ Todos los datasets guardados en:", output_dir)
    print("=" * 60)

    return df_ocupacion, df_turistas, df_establecimientos, df_interacciones


if __name__ == "__main__":
    generar_todos_los_datasets("outputs")
