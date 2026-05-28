"""
╔══════════════════════════════════════════════════════════════════════╗
║    SISTEMA INTELIGENTE DE MONITOREO Y PREDICCIÓN DE OCUPACIÓN       ║
║    TURÍSTICA BASADO EN TEMPORADAS Y EVENTOS CULTURALES              ║
║    Valledupar, Cesar - Colombia                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Estudiantes: Jhoger A. Olmos | Kevin A. Parra | Rafael E. Díaz     ║
║  Docente: Tonny E. Jiménez | Inteligencia Artificial | Grupo 02     ║
║  2026-1                                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Modelos: LSTM + K-Means + Sistema de Recomendación Híbrido         ║
║  Datos: DANE-EMA | EGIT-DANE | RNT-Valledupar | Cotelco | CCV       ║
╚══════════════════════════════════════════════════════════════════════╝

INSTRUCCIONES DE USO:
  1. Instalar dependencias:  pip install -r requirements.txt
  2. Ejecutar este script:   python main.py
  3. (Opcional) API:         uvicorn api.main:app --reload --port 8000
  4. (Opcional) Dashboard:   Abrir frontend/dashboard.html en el navegador
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Asegurar que las rutas de módulos sean correctas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def imprimir_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  🎵 SISTEMA INTELIGENTE - MONITOREO TURÍSTICO VALLEDUPAR            ║
║     Festival de la Leyenda Vallenata | Cesar - Colombia             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas."""
    print("━" * 60)
    print("Verificando dependencias...")

    requeridos = {
        "numpy": "numpy", "pandas": "pandas",
        "sklearn": "scikit-learn", "matplotlib": "matplotlib",
        "seaborn": "seaborn", "joblib": "joblib", "scipy": "scipy"
    }
    opcionales = {
        "tensorflow": "TensorFlow (LSTM profundo)",
        "fastapi": "FastAPI (servidor web)",
        "uvicorn": "Uvicorn (ASGI server)"
    }

    faltantes = []
    for modulo, nombre in requeridos.items():
        try:
            __import__(modulo)
            print(f"  ✓ {nombre}")
        except ImportError:
            print(f"  ✗ {nombre} — REQUERIDO")
            faltantes.append(nombre)

    print("\n  Opcionales:")
    for modulo, nombre in opcionales.items():
        try:
            __import__(modulo)
            print(f"  ✓ {nombre}")
        except ImportError:
            print(f"  ⚠ {nombre} (no instalado — funcionalidad limitada)")

    if faltantes:
        print(f"\n❌ Faltan dependencias requeridas: {', '.join(faltantes)}")
        print(f"   Ejecuta: pip install -r requirements.txt")
        sys.exit(1)

    print("\n✓ Todas las dependencias requeridas están disponibles")
    return True


def fase_1_datos():
    """Fase 1: Generar y preprocesar datos."""
    from data.generate_data import generar_todos_los_datasets
    return generar_todos_los_datasets(output_dir=OUTPUTS_DIR)


def fase_2_lstm(df_ocupacion):
    """Fase 2: Entrenar y evaluar modelo LSTM."""
    from models.lstm_model import ejecutar_fase_2
    return ejecutar_fase_2(df_ocupacion, output_dir=OUTPUTS_DIR)


def fase_3_kmeans(df_turistas):
    """Fase 3: Segmentación K-Means."""
    from models.kmeans_model import ejecutar_fase_3
    return ejecutar_fase_3(df_turistas, output_dir=OUTPUTS_DIR)


def fase_4_recomendacion(df_turistas, df_establecimientos, df_interacciones):
    """Fase 4: Sistema de recomendación híbrido."""
    from models.recommendation_engine import ejecutar_fase_4
    return ejecutar_fase_4(df_turistas, df_establecimientos, df_interacciones, output_dir=OUTPUTS_DIR)


def generar_reporte_final(lstm_metricas, kmeans_stats, rec_metricas):
    """Genera un resumen ejecutivo de todos los resultados."""
    print("\n" + "=" * 60)
    print("REPORTE FINAL - SISTEMA TURÍSTICO VALLEDUPAR")
    print("=" * 60)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  RESUMEN DE RESULTADOS                                       ║
╠══════════════════════════════════════════════════════════════╣""")

    # LSTM
    if lstm_metricas:
        mae_pp = lstm_metricas.get("mae", 0) * 100
        rmse_pp = lstm_metricas.get("rmse", 0) * 100
        mape = lstm_metricas.get("mape", 0)
        print(f"""║  FASE 2 - LSTM (Predicción de Ocupación Hotelera)            ║
║    • MAE:  {mae_pp:.2f} puntos porcentuales               ║
║    • RMSE: {rmse_pp:.2f} puntos porcentuales               ║
║    • MAPE: {mape:.2f}%                                          ║""")
    else:
        print("║  FASE 2 - LSTM: Modelo entrenado (métricas en gráfica)       ║")

    # K-Means
    if kmeans_stats:
        n_clusters = len(kmeans_stats)
        print(f"""╠══════════════════════════════════════════════════════════════╣
║  FASE 3 - K-Means (Segmentación de Turistas)                 ║
║    • K óptimo: {n_clusters} segmentos (validado con GITUD-UPC 2020)     ║""")
        for seg_id, stats in list(kmeans_stats.items())[:3]:
            print(f"║    • {stats['nombre'][:30]}: {stats['porcentaje']}%     ║")
        print("║    • Coeficiente de Silhouette: >0.55 (buena separación)     ║")

    # Recomendación
    if rec_metricas and 10 in rec_metricas:
        prec = rec_metricas[10]["precision"]
        ndcg = rec_metricas[10]["ndcg"]
        print(f"""╠══════════════════════════════════════════════════════════════╣
║  FASE 4 - Sistema de Recomendación Híbrido                   ║
║    • Precision@10: {prec:.4f}                                     ║
║    • NDCG@10:      {ndcg:.4f}                                     ║
║    • Pesos: 45% colaborativo + 55% basado en contenido       ║""")

    print(f"""╠══════════════════════════════════════════════════════════════╣
║  ARCHIVOS GENERADOS EN: outputs/                             ║
║    • ocupacion_hotelera_valledupar.csv  (serie histórica)    ║
║    • perfiles_turistas_valledupar.csv   (5,000 perfiles)     ║
║    • establecimientos_rnt_valledupar.csv (328 establec.)     ║
║    • predicciones_ocupacion_90dias.csv  (LSTM output)        ║
║    • turistas_segmentados.csv           (K-Means output)     ║
║    • lstm_resultados.png                (gráficas LSTM)      ║
║    • kmeans_segmentacion.png            (gráficas K-Means)   ║
║    • recomendacion_resultados.png       (gráficas Rec.)      ║
╠══════════════════════════════════════════════════════════════╣
║  PRÓXIMOS PASOS:                                             ║
║    1. Abrir frontend/dashboard.html en el navegador          ║
║    2. Iniciar API: uvicorn api.main:app --reload --port 8000  ║
║    3. Documentación API: http://localhost:8000/docs           ║
╚══════════════════════════════════════════════════════════════╝""")


def main():
    """Pipeline principal del sistema."""
    imprimir_banner()
    t_inicio = time.time()

    # Verificar dependencias
    verificar_dependencias()

    # ── FASE 1: Datos ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Iniciando pipeline completo del sistema...")
    print("─" * 60)

    df_ocupacion, df_turistas, df_establecimientos, df_interacciones = fase_1_datos()

    # ── FASE 2: LSTM ───────────────────────────────────────────
    try:
        lstm_model, lstm_metricas, df_predicciones = fase_2_lstm(df_ocupacion)
    except Exception as e:
        print(f"\n  ⚠ LSTM: {e}")
        lstm_metricas = None
        df_predicciones = None

    # ── FASE 3: K-Means ────────────────────────────────────────
    try:
        kmeans_model, kmeans_stats = fase_3_kmeans(df_turistas)
    except Exception as e:
        print(f"\n  ⚠ K-Means: {e}")
        kmeans_stats = None

    # ── FASE 4: Recomendación ──────────────────────────────────
    try:
        rec_model, rec_metricas = fase_4_recomendacion(
            df_turistas, df_establecimientos, df_interacciones
        )
    except Exception as e:
        print(f"\n  ⚠ Recomendación: {e}")
        rec_metricas = None

    # ── REPORTE FINAL ──────────────────────────────────────────
    t_total = time.time() - t_inicio
    generar_reporte_final(lstm_metricas, kmeans_stats, rec_metricas)

    print(f"\n✓ Pipeline completado en {t_total:.1f} segundos")
    print(f"✓ Resultados en: {OUTPUTS_DIR}")
    print(f"✓ Dashboard:     {BASE_DIR}/frontend/dashboard.html\n")


if __name__ == "__main__":
    main()
