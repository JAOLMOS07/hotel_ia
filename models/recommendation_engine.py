"""
FASE 4: Sistema de Recomendación Híbrido
Sistema Inteligente de Monitoreo y Predicción de Ocupación Turística
Valledupar - Cesar - Colombia

Implementa el sistema de recomendación combinando:
1. Filtrado Colaborativo: similitud coseno entre usuarios (interacciones históricas)
2. Filtrado Basado en Contenido: perfil K-Means vs características de establecimientos
3. Combinación Híbrida: pondera ambos enfoques según disponibilidad de historial

Métricas: Precision@K, Recall@K, NDCG (Normalized Discounted Cumulative Gain)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL DEL SISTEMA DE RECOMENDACIÓN
# ─────────────────────────────────────────────────────────────────

class SistemaRecomendacionHibrido:
    """
    Sistema de recomendación que combina filtrado colaborativo y basado en contenido.
    Diseñado para el ecosistema turístico de Valledupar con 328 establecimientos (RNT).
    """

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.matriz_usuario_item = None
        self.similitud_usuarios = None
        self.similitud_items = None
        self.df_establecimientos = None
        self.df_turistas = None
        self.df_interacciones = None
        self.scaler_contenido = MinMaxScaler()
        os.makedirs(output_dir, exist_ok=True)

        # Pesos del sistema híbrido (ajustados para destino emergente)
        self.peso_colaborativo = 0.45
        self.peso_contenido = 0.55  # Mayor peso contenido para turistas sin historial

    def cargar_datos(self, df_turistas, df_establecimientos, df_interacciones):
        """Carga los datos necesarios para el sistema de recomendación."""
        self.df_turistas = df_turistas
        self.df_establecimientos = df_establecimientos
        self.df_interacciones = df_interacciones

        print(f"  Turistas: {len(df_turistas):,}")
        print(f"  Establecimientos: {len(df_establecimientos)}")
        print(f"  Interacciones: {len(df_interacciones):,}")

    def construir_matriz_usuario_item(self):
        """
        Construye la matriz de interacción usuario-ítem para filtrado colaborativo.
        Filas: turistas | Columnas: establecimientos | Valores: rating (1-5) o 0
        """
        print("\n  Construyendo matriz usuario-ítem...")

        # Usar solo los primeros 500 turistas para eficiencia
        turistas_sample = self.df_turistas["turista_id"].unique()[:500]
        establecimientos_all = self.df_establecimientos["establecimiento_id"].unique()

        self.turistas_idx = {t: i for i, t in enumerate(turistas_sample)}
        self.establecimientos_idx = {e: j for j, e in enumerate(establecimientos_all)}

        n_usuarios = len(turistas_sample)
        n_items = len(establecimientos_all)

        matriz = np.zeros((n_usuarios, n_items))

        interacciones_filtradas = self.df_interacciones[
            self.df_interacciones["turista_id"].isin(turistas_sample)
        ]

        for _, row in interacciones_filtradas.iterrows():
            if row["turista_id"] in self.turistas_idx and row["establecimiento_id"] in self.establecimientos_idx:
                u_idx = self.turistas_idx[row["turista_id"]]
                i_idx = self.establecimientos_idx[row["establecimiento_id"]]
                matriz[u_idx, i_idx] = row["rating"]

        self.matriz_usuario_item = matriz
        densidad = (matriz > 0).sum() / (n_usuarios * n_items) * 100
        print(f"  Matriz: {n_usuarios} × {n_items} | Densidad: {densidad:.2f}%")

        return matriz

    def calcular_similitud_usuarios(self):
        """Calcula similitud coseno entre usuarios para filtrado colaborativo."""
        print("  Calculando similitud entre usuarios (coseno)...")
        self.similitud_usuarios = cosine_similarity(self.matriz_usuario_item)
        np.fill_diagonal(self.similitud_usuarios, 0)  # Excluir auto-similitud
        return self.similitud_usuarios

    def construir_perfil_contenido(self):
        """
        Construye vectores de características para filtrado basado en contenido.
        Combina perfil del turista (K-Means) con características de establecimientos.
        """
        print("  Construyendo perfiles de contenido...")

        features_contenido = [
            "categoria_estrellas",
            "precio_noche_promedio",
            "capacidad_habitaciones",
            "puntuacion_promedio",
            "n_resenas",
            "acepta_reservas_online",
            "tiene_restaurante",
            "tiene_piscina",
            "tiene_estacionamiento",
            "permite_mascotas",
            "apto_familias"
        ]

        df_est = self.df_establecimientos.copy()
        # Convertir booleanos a numérico
        for col in ["acepta_reservas_online", "tiene_restaurante", "tiene_piscina",
                   "tiene_estacionamiento", "permite_mascotas", "apto_familias"]:
            df_est[col] = df_est[col].astype(int)

        X_contenido = df_est[features_contenido].fillna(df_est[features_contenido].median()).values
        self.X_contenido_scaled = self.scaler_contenido.fit_transform(X_contenido)
        self.similitud_items = cosine_similarity(self.X_contenido_scaled)
        self.features_contenido = features_contenido

        return self.X_contenido_scaled

    def recomendar_colaborativo(self, turista_id, n_recomendaciones=10):
        """
        Filtrado colaborativo: recomienda basándose en gustos de usuarios similares.
        """
        if turista_id not in self.turistas_idx:
            return []

        u_idx = self.turistas_idx[turista_id]
        similitudes = self.similitud_usuarios[u_idx]

        # Top-20 usuarios más similares
        top_usuarios = np.argsort(similitudes)[::-1][:20]

        # Predicciones ponderadas por similitud
        predicciones = np.zeros(self.matriz_usuario_item.shape[1])
        suma_similitudes = np.zeros(self.matriz_usuario_item.shape[1])

        for vecino_idx in top_usuarios:
            sim = similitudes[vecino_idx]
            if sim > 0:
                ratings_vecino = self.matriz_usuario_item[vecino_idx]
                predicciones += sim * ratings_vecino
                suma_similitudes += sim * (ratings_vecino > 0).astype(float)

        # Evitar división por cero
        with np.errstate(divide="ignore", invalid="ignore"):
            predicciones = np.where(suma_similitudes > 0, predicciones / suma_similitudes, 0)

        # Excluir ítems ya valorados por el usuario
        ya_valorados = self.matriz_usuario_item[u_idx] > 0
        predicciones[ya_valorados] = 0

        # Top-N recomendaciones
        top_idx = np.argsort(predicciones)[::-1][:n_recomendaciones]
        establecimientos_lista = list(self.establecimientos_idx.keys())

        recomendaciones = []
        for idx in top_idx:
            if predicciones[idx] > 0 and idx < len(establecimientos_lista):
                est_id = establecimientos_lista[idx]
                recomendaciones.append({
                    "establecimiento_id": est_id,
                    "score_colaborativo": round(float(predicciones[idx]), 4)
                })

        return recomendaciones

    def recomendar_por_contenido(self, perfil_turista, n_recomendaciones=10):
        """
        Filtrado basado en contenido: recomienda según perfil K-Means del turista.
        """
        # Construir vector de preferencias según perfil del turista
        tipo_aloj_pref = perfil_turista.get("tipo_alojamiento", "hotel_3_estrellas")
        presupuesto = perfil_turista.get("gasto_alojamiento_cop", 800000)
        grupo_tamano = perfil_turista.get("grupo_tamano", 2)
        segmento = perfil_turista.get("segmento_nombre", "Turista Cultural Tradicional")

        # Mapeo de tipo de alojamiento a categoría preferida
        cat_pref = {
            "hotel_5_estrellas": 5, "hotel_4_estrellas": 4,
            "hotel_3_estrellas": 3, "hotel_2_estrellas": 2,
            "hostal_familiar": 2, "hostal_economico": 1,
            "apartahotel": 3, "casa_huespedes": 1
        }.get(tipo_aloj_pref, 3)

        # Calcular scores de contenido para cada establecimiento
        df_est = self.df_establecimientos.copy()
        scores = []

        for _, est in df_est.iterrows():
            # Score por categoría preferida
            diff_cat = abs(est["categoria_estrellas"] - cat_pref)
            score_cat = max(0, 1 - diff_cat / 5)

            # Score por precio (dentro del presupuesto)
            precio_unit = presupuesto / max(1, perfil_turista.get("dias_estancia", 3))
            diff_precio = abs(est["precio_noche_promedio"] - precio_unit) / max(precio_unit, 1)
            score_precio = max(0, 1 - min(diff_precio, 1))

            # Score por calidad (puntuación y reseñas)
            score_calidad = (est["puntuacion_promedio"] / 5) * min(1, est["n_resenas"] / 100)

            # Score por amenidades según perfil
            score_amenidades = 0
            if grupo_tamano >= 3 and est["apto_familias"]:
                score_amenidades += 0.2
            if est["acepta_reservas_online"]:
                score_amenidades += 0.1
            if "Premium" in segmento or "Negocios" in segmento:
                if est["tiene_restaurante"]:
                    score_amenidades += 0.1
                if est["tiene_piscina"]:
                    score_amenidades += 0.1

            score_total = (0.30 * score_cat + 0.25 * score_precio +
                          0.35 * score_calidad + 0.10 * score_amenidades)
            scores.append(score_total)

        df_est["score_contenido"] = scores

        top_est = df_est.nlargest(n_recomendaciones, "score_contenido")
        return [{"establecimiento_id": row["establecimiento_id"],
                "score_contenido": round(row["score_contenido"], 4)}
               for _, row in top_est.iterrows()]

    def recomendar_hibrido(self, turista_id, perfil_turista, n_recomendaciones=10):
        """
        Sistema híbrido: combina filtrado colaborativo y basado en contenido.
        Ajusta los pesos según la disponibilidad de historial del usuario.
        """
        # Obtener recomendaciones de ambos enfoques
        rec_colaborativo = self.recomendar_colaborativo(turista_id, n_recomendaciones * 2)
        rec_contenido = self.recomendar_por_contenido(perfil_turista, n_recomendaciones * 2)

        # Determinar pesos según historial disponible
        tiene_historial = len(rec_colaborativo) >= 3
        w_col = self.peso_colaborativo if tiene_historial else 0.15
        w_cont = 1 - w_col

        # Combinar scores en diccionario unificado
        scores_combinados = {}

        max_score_col = max([r["score_colaborativo"] for r in rec_colaborativo], default=1)
        for rec in rec_colaborativo:
            est_id = rec["establecimiento_id"]
            score_norm = rec["score_colaborativo"] / max(max_score_col, 0.01)
            scores_combinados[est_id] = scores_combinados.get(est_id, 0) + w_col * score_norm

        max_score_cont = max([r["score_contenido"] for r in rec_contenido], default=1)
        for rec in rec_contenido:
            est_id = rec["establecimiento_id"]
            score_norm = rec["score_contenido"] / max(max_score_cont, 0.01)
            scores_combinados[est_id] = scores_combinados.get(est_id, 0) + w_cont * score_norm

        # Ordenar y obtener Top-N
        top_ids = sorted(scores_combinados, key=scores_combinados.get, reverse=True)[:n_recomendaciones]

        # Enriquecer con información del establecimiento
        recomendaciones_finales = []
        for rank, est_id in enumerate(top_ids, 1):
            est_info = self.df_establecimientos[
                self.df_establecimientos["establecimiento_id"] == est_id
            ]
            if len(est_info) > 0:
                est = est_info.iloc[0]
                recomendaciones_finales.append({
                    "rank": rank,
                    "establecimiento_id": est_id,
                    "nombre": est["nombre"],
                    "tipo": est["tipo"],
                    "categoria_estrellas": int(est["categoria_estrellas"]),
                    "precio_noche": int(est["precio_noche_promedio"]),
                    "puntuacion": float(est["puntuacion_promedio"]),
                    "zona": est["zona"],
                    "score_hibrido": round(scores_combinados[est_id], 4),
                    "tiene_restaurante": bool(est["tiene_restaurante"]),
                    "apto_familias": bool(est["apto_familias"]),
                    "acepta_reservas_online": bool(est["acepta_reservas_online"])
                })

        return recomendaciones_finales

    def evaluar_sistema(self, test_size=0.2):
        """
        Evalúa el sistema con Precision@K, Recall@K y NDCG.
        """
        print("\n" + "─" * 50)
        print("EVALUACIÓN DEL SISTEMA DE RECOMENDACIÓN")
        print("─" * 50)

        turistas_test = list(self.turistas_idx.keys())[:int(len(self.turistas_idx) * test_size)]
        K_valores = [5, 10]

        resultados_metricas = {}

        for K in K_valores:
            precision_list, recall_list, ndcg_list = [], [], []

            for turista_id in turistas_test[:50]:  # Muestra para eficiencia
                u_idx = self.turistas_idx[turista_id]
                ratings_reales = self.matriz_usuario_item[u_idx]
                items_relevantes = set(np.where(ratings_reales >= 4.0)[0])

                if len(items_relevantes) == 0:
                    continue

                # Predicciones colaborativas
                predicciones = self.similitud_usuarios[u_idx] @ self.matriz_usuario_item
                suma_sim = np.abs(self.similitud_usuarios[u_idx]).sum()
                if suma_sim > 0:
                    predicciones /= suma_sim

                # Excluir ya vistos
                predicciones[ratings_reales > 0] = -1
                top_k_indices = set(np.argsort(predicciones)[::-1][:K])

                # Precision@K
                hits = len(top_k_indices & items_relevantes)
                precision = hits / K
                recall = hits / len(items_relevantes) if items_relevantes else 0

                # NDCG@K
                dcg = 0
                for rank, idx in enumerate(np.argsort(predicciones)[::-1][:K], 1):
                    if idx in items_relevantes:
                        dcg += 1 / np.log2(rank + 1)
                idcg = sum(1 / np.log2(i + 1) for i in range(1, min(K, len(items_relevantes)) + 1))
                ndcg = dcg / idcg if idcg > 0 else 0

                precision_list.append(precision)
                recall_list.append(recall)
                ndcg_list.append(ndcg)

            resultados_metricas[K] = {
                "precision": np.mean(precision_list) if precision_list else 0,
                "recall": np.mean(recall_list) if recall_list else 0,
                "ndcg": np.mean(ndcg_list) if ndcg_list else 0
            }

            print(f"\n  Métricas @K={K}:")
            print(f"    Precision@{K}: {resultados_metricas[K]['precision']:.4f}")
            print(f"    Recall@{K}:    {resultados_metricas[K]['recall']:.4f}")
            print(f"    NDCG@{K}:      {resultados_metricas[K]['ndcg']:.4f}")

        self.metricas_evaluacion = resultados_metricas
        return resultados_metricas

    def graficar_resultados(self):
        """Genera visualizaciones del sistema de recomendación."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Sistema de Recomendación Híbrido\nValledupar, Cesar - Colombia",
                    fontsize=14, fontweight="bold")

        # 1. Distribución de ratings
        ax1 = axes[0]
        ratings = self.df_interacciones["rating"].dropna()
        ax1.hist(ratings, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
        ax1.axvline(x=ratings.mean(), color="red", linestyle="--",
                   label=f"Promedio: {ratings.mean():.2f}")
        ax1.set_title("Distribución de Ratings de Usuarios")
        ax1.set_xlabel("Rating (1-5)")
        ax1.set_ylabel("Frecuencia")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Top establecimientos más reservados
        ax2 = axes[1]
        top_est = (self.df_interacciones.groupby("establecimiento_id")
                  .agg(n_reservas=("rating", "count"), rating_medio=("rating", "mean"))
                  .nlargest(10, "n_reservas"))
        nombres_cortos = [f"EST_{i+1}" for i in range(len(top_est))]
        bars = ax2.barh(nombres_cortos, top_est["n_reservas"].values, color="#2ecc71", alpha=0.8)
        ax2.set_title("Top 10 Establecimientos Más Reservados")
        ax2.set_xlabel("Número de Reservas")
        ax2.grid(True, alpha=0.3, axis="x")

        # 3. Métricas de evaluación
        ax3 = axes[2]
        if hasattr(self, "metricas_evaluacion"):
            k_vals = list(self.metricas_evaluacion.keys())
            metricas_nombres = ["precision", "recall", "ndcg"]
            colores_met = ["#e74c3c", "#3498db", "#2ecc71"]
            x = np.arange(len(k_vals))
            width = 0.25

            for i, (metrica, color) in enumerate(zip(metricas_nombres, colores_met)):
                vals = [self.metricas_evaluacion[k][metrica] for k in k_vals]
                ax3.bar(x + i * width, vals, width, label=metrica.upper(),
                       color=color, alpha=0.8)
                for j, v in enumerate(vals):
                    ax3.text(x[j] + i * width, v + 0.005, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

            ax3.set_xticks(x + width)
            ax3.set_xticklabels([f"@K={k}" for k in k_vals])
            ax3.set_title("Métricas de Evaluación del Sistema")
            ax3.set_ylabel("Score")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis="y")
            ax3.set_ylim(0, 0.8)

        plt.tight_layout()
        ruta = f"{self.output_dir}/recomendacion_resultados.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Gráfica guardada: {ruta}")

    def guardar_modelo(self):
        """Serializa el modelo para la API."""
        joblib.dump(self.scaler_contenido, f"{self.output_dir}/recomendacion_scaler.pkl")
        if self.similitud_usuarios is not None:
            np.save(f"{self.output_dir}/similitud_usuarios.npy", self.similitud_usuarios)
        print(f"  ✓ Sistema de recomendación guardado: {self.output_dir}/")


def ejecutar_fase_4(df_turistas, df_establecimientos, df_interacciones, output_dir="outputs"):
    """Función principal de la Fase 4."""
    print("\n" + "=" * 60)
    print("FASE 4: Sistema de Recomendación Híbrido")
    print("=" * 60)

    recomendador = SistemaRecomendacionHibrido(output_dir=output_dir)
    recomendador.cargar_datos(df_turistas, df_establecimientos, df_interacciones)
    recomendador.construir_matriz_usuario_item()
    recomendador.calcular_similitud_usuarios()
    recomendador.construir_perfil_contenido()
    metricas = recomendador.evaluar_sistema()

    # Ejemplo de recomendación
    print("\n  Ejemplo de recomendación híbrida:")
    turista_ejemplo_id = df_turistas["turista_id"].iloc[0]
    perfil_ejemplo = {
        "tipo_alojamiento": "hotel_3_estrellas",
        "gasto_alojamiento_cop": 600000,
        "dias_estancia": 3,
        "grupo_tamano": 2,
        "segmento_nombre": "Turista Cultural Tradicional"
    }
    recs = recomendador.recomendar_hibrido(turista_ejemplo_id, perfil_ejemplo, n_recomendaciones=5)
    print(f"  Top-5 recomendaciones para {turista_ejemplo_id}:")
    for rec in recs:
        print(f"    #{rec['rank']} {rec['nombre']} ({rec['tipo']}) - "
              f"${rec['precio_noche']:,} COP/noche - ⭐{rec['puntuacion']} "
              f"(score: {rec['score_hibrido']:.3f})")

    recomendador.graficar_resultados()
    recomendador.guardar_modelo()

    return recomendador, metricas


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_data import (generar_perfiles_turistas, generar_catalogo_establecimientos,
                                     generar_interacciones_usuarios)
    df_t = generar_perfiles_turistas(2000)
    df_e = generar_catalogo_establecimientos(328)
    df_i = generar_interacciones_usuarios(df_t, df_e, 10000)
    ejecutar_fase_4(df_t, df_e, df_i, "outputs")
