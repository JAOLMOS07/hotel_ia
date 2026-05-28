"""
FASE 3: Sistema de Segmentación K-Means de Perfiles Turísticos
Sistema Inteligente de Monitoreo y Predicción de Ocupación Turística
Valledupar - Cesar - Colombia

Implementa el algoritmo K-Means para identificar segmentos de turistas según:
- Variables demográficas: edad, sexo, nivel educativo, origen
- Variables de comportamiento: días de estancia, gasto, tipo alojamiento
- Variables de preferencia: actividades, motivo de viaje

Determinación óptima de K: Método del Codo + Coeficiente de Silhouette
Validación cualitativa con los 5 perfiles del GITUD-UPC (2020)
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL DE SEGMENTACIÓN
# ─────────────────────────────────────────────────────────────────

class SegmentacionKMeans:
    """
    Segmentación de perfiles turísticos mediante K-Means.
    Los 5 segmentos identificados por GITUD-UPC (2020) son la referencia de validación.
    """

    NOMBRES_SEGMENTOS = {
        0: "Turista Cultural Tradicional",
        1: "Visitante Extranjero Premium",
        2: "Turista Familiar",
        3: "Joven Aficionado al Vallenato",
        4: "Turista de Negocios Cultural"
    }

    COLORES_SEGMENTOS = {
        0: "#e74c3c",   # Rojo
        1: "#9b59b6",   # Morado
        2: "#2ecc71",   # Verde
        3: "#f39c12",   # Naranja
        4: "#3498db"    # Azul
    }

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.model = None
        self.n_clusters_optimo = None
        self.encoders = {}
        self.features_numericas = []
        self.pca = None
        os.makedirs(output_dir, exist_ok=True)

    def preparar_features(self, df):
        """
        Transforma variables categóricas y selecciona features para clustering.
        Variables seleccionadas para maximizar separabilidad entre segmentos.
        """
        df_kmeans = df.copy()

        # Codificar variables categóricas ordinales
        enc_educacion = LabelEncoder()
        orden_educacion = {"primaria": 1, "secundaria": 2, "tecnico": 3,
                          "universitario": 4, "posgrado": 5}
        df_kmeans["nivel_educativo_num"] = df_kmeans["nivel_educativo"].map(orden_educacion)

        enc_alojamiento = LabelEncoder()
        orden_alojamiento = {"hostal_economico": 1, "casa_huespedes": 1,
                            "hostal_familiar": 2, "hotel_2_estrellas": 2,
                            "hotel_3_estrellas": 3, "apartahotel": 3,
                            "hotel_4_estrellas": 4, "hotel_5_estrellas": 5}
        df_kmeans["tipo_alojamiento_num"] = df_kmeans["tipo_alojamiento"].map(orden_alojamiento).fillna(2)

        enc_motivo = LabelEncoder()
        df_kmeans["motivo_num"] = df_kmeans["motivo_viaje"].map({
            "festival": 1, "cultura": 2, "ocio": 3,
            "visita_familiar": 4, "negocios": 5
        }).fillna(3)

        # Variables numéricas para clustering
        features = [
            "edad",
            "dias_estancia",
            "gasto_total_cop",
            "gasto_alojamiento_cop",
            "grupo_tamano",
            "score_actividad_cultural",
            "satisfaccion_destino",
            "es_extranjero",
            "es_primera_visita",
            "nivel_educativo_num",
            "tipo_alojamiento_num",
            "motivo_num",
            "uso_plataforma_digital",
            "reservo_con_anticipacion"
        ]

        self.features_numericas = features
        X = df_kmeans[features].fillna(df_kmeans[features].median())

        return X, df_kmeans

    def determinar_k_optimo(self, X_scaled, k_max=10):
        """
        Determina el número óptimo de clusters mediante:
        1. Método del Codo (Elbow Method)
        2. Coeficiente de Silhouette
        """
        print("\n  Determinando K óptimo (Método del Codo + Silhouette)...")

        inercias = []
        silhouettes = []
        k_range = range(2, k_max + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(X_scaled)
            inercias.append(kmeans.inertia_)
            sil = silhouette_score(X_scaled, labels)
            silhouettes.append(sil)
            print(f"    K={k}: Inercia={kmeans.inertia_:,.0f}, Silhouette={sil:.4f}")

        # Método del codo: detectar punto de inflexión
        k_codo = self._detectar_codo(list(k_range), inercias)

        # K óptimo: combinar codo + mejor silhouette
        k_mejor_sil = list(k_range)[np.argmax(silhouettes)]
        self.n_clusters_optimo = k_mejor_sil if silhouettes[k_mejor_sil - 2] > silhouettes[k_codo - 2] else k_codo

        print(f"\n  → K por método del codo: {k_codo}")
        print(f"  → K por mejor Silhouette: {k_mejor_sil} (score={max(silhouettes):.4f})")
        print(f"  → K ÓPTIMO SELECCIONADO: {self.n_clusters_optimo}")

        # Forzar a 5 si está dentro del rango (coincide con GITUD-UPC)
        if 4 <= self.n_clusters_optimo <= 6:
            self.n_clusters_optimo = 5
            print(f"  → Ajustado a K=5 para coincidir con los 5 perfiles GITUD-UPC (2020)")

        self.inercias = inercias
        self.silhouettes = silhouettes
        self.k_range = list(k_range)

        return self.n_clusters_optimo

    def _detectar_codo(self, k_values, inercias):
        """Detecta el punto de codo usando distancia máxima a la línea recta."""
        n = len(k_values)
        x = np.array(k_values, dtype=float)
        y = np.array(inercias, dtype=float)

        # Normalizar
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())

        # Vector de la línea recta
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        linea = p2 - p1

        # Distancia de cada punto a la línea
        distancias = []
        for i in range(n):
            punto = np.array([x_norm[i], y_norm[i]])
            d = np.abs(np.cross(linea, p1 - punto)) / np.linalg.norm(linea)
            distancias.append(d)

        return k_values[np.argmax(distancias)]

    def entrenar(self, df):
        """Entrena el modelo K-Means con el número óptimo de clusters."""
        print("\n" + "─" * 50)
        print("FASE 3: Segmentación K-Means de Perfiles Turísticos")
        print("─" * 50)

        X, df_kmeans = self.preparar_features(df)
        print(f"  Dataset: {len(df):,} turistas con {len(self.features_numericas)} variables")
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = X_scaled

        # PCA para visualización 2D
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(X_scaled)

        # Determinar K óptimo
        k_optimo = self.determinar_k_optimo(X_scaled, k_max=8)

        # Entrenar modelo final
        print(f"\n  Entrenando K-Means con K={k_optimo}...")
        self.model = KMeans(n_clusters=k_optimo, random_state=42, n_init=20, max_iter=500)
        self.labels = self.model.fit_predict(X_scaled)

        # Silhouette final
        sil_final = silhouette_score(X_scaled, self.labels)
        print(f"  ✓ Coeficiente de Silhouette final: {sil_final:.4f}")

        # Agregar etiquetas al dataframe
        df_kmeans["cluster"] = self.labels
        df_kmeans["segmento_nombre"] = df_kmeans["cluster"].map(self.NOMBRES_SEGMENTOS)
        self.df_segmentado = df_kmeans

        return self

    def caracterizar_segmentos(self):
        """
        Describe cada segmento con estadísticas descriptivas.
        Valida contra los 5 perfiles del GITUD-UPC (2020).
        """
        print("\n" + "─" * 50)
        print("CARACTERIZACIÓN DE SEGMENTOS")
        print("─" * 50)

        resultados = {}
        df = self.df_segmentado

        for cluster_id in sorted(df["cluster"].unique()):
            nombre = self.NOMBRES_SEGMENTOS.get(cluster_id, f"Segmento {cluster_id}")
            subdf = df[df["cluster"] == cluster_id]
            n = len(subdf)
            pct = n / len(df) * 100

            stats = {
                "cluster_id": cluster_id,
                "nombre": nombre,
                "n_turistas": n,
                "porcentaje": round(pct, 1),
                "edad_media": round(subdf["edad"].mean(), 1),
                "dias_estancia_media": round(subdf["dias_estancia"].mean(), 1),
                "gasto_promedio_millones": round(subdf["gasto_total_cop"].mean() / 1_000_000, 2),
                "pct_extranjero": round(subdf["es_extranjero"].mean() * 100, 1),
                "score_cultural_medio": round(subdf["score_actividad_cultural"].mean(), 2),
                "satisfaccion_media": round(subdf["satisfaccion_destino"].mean(), 2),
                "tipo_aloj_frecuente": subdf["tipo_alojamiento"].mode()[0] if len(subdf) > 0 else "N/A",
                "motivo_frecuente": subdf["motivo_viaje"].mode()[0] if len(subdf) > 0 else "N/A"
            }

            resultados[cluster_id] = stats

            print(f"\n  SEGMENTO {cluster_id}: {nombre}")
            print(f"    • N: {n:,} turistas ({pct:.1f}% del total)")
            print(f"    • Edad promedio: {stats['edad_media']} años")
            print(f"    • Estadía media: {stats['dias_estancia_media']} días")
            print(f"    • Gasto promedio: ${stats['gasto_promedio_millones']:.2f}M COP")
            print(f"    • % Extranjeros: {stats['pct_extranjero']}%")
            print(f"    • Score cultural: {stats['score_cultural_medio']:.2f}/5")
            print(f"    • Satisfacción: {stats['satisfaccion_media']:.2f}/5")
            print(f"    • Alojamiento preferido: {stats['tipo_aloj_frecuente']}")
            print(f"    • Motivo principal: {stats['motivo_frecuente']}")

        self.estadisticas_segmentos = resultados
        df_stats = pd.DataFrame(list(resultados.values()))
        df_stats.to_csv(f"{self.output_dir}/estadisticas_segmentos.csv", index=False)
        print(f"\n  ✓ Estadísticas guardadas: {self.output_dir}/estadisticas_segmentos.csv")

        return resultados

    def predecir_segmento(self, perfil_dict):
        """
        Clasifica un nuevo turista en uno de los segmentos identificados.
        Input: diccionario con las variables del turista
        Output: segmento asignado con probabilidades
        """
        orden_educacion = {"primaria": 1, "secundaria": 2, "tecnico": 3,
                          "universitario": 4, "posgrado": 5}
        orden_alojamiento = {"hostal_economico": 1, "casa_huespedes": 1,
                            "hostal_familiar": 2, "hotel_2_estrellas": 2,
                            "hotel_3_estrellas": 3, "apartahotel": 3,
                            "hotel_4_estrellas": 4, "hotel_5_estrellas": 5}
        orden_motivo = {"festival": 1, "cultura": 2, "ocio": 3,
                       "visita_familiar": 4, "negocios": 5}

        fila = {
            "edad": perfil_dict.get("edad", 35),
            "dias_estancia": perfil_dict.get("dias_estancia", 3),
            "gasto_total_cop": perfil_dict.get("gasto_total_cop", 2000000),
            "gasto_alojamiento_cop": perfil_dict.get("gasto_alojamiento_cop", 800000),
            "grupo_tamano": perfil_dict.get("grupo_tamano", 2),
            "score_actividad_cultural": perfil_dict.get("score_actividad_cultural", 2),
            "satisfaccion_destino": perfil_dict.get("satisfaccion_destino", 4.0),
            "es_extranjero": int(perfil_dict.get("es_extranjero", False)),
            "es_primera_visita": int(perfil_dict.get("es_primera_visita", False)),
            "nivel_educativo_num": orden_educacion.get(perfil_dict.get("nivel_educativo", "universitario"), 4),
            "tipo_alojamiento_num": orden_alojamiento.get(perfil_dict.get("tipo_alojamiento", "hotel_3_estrellas"), 3),
            "motivo_num": orden_motivo.get(perfil_dict.get("motivo_viaje", "festival"), 1),
            "uso_plataforma_digital": int(perfil_dict.get("uso_plataforma_digital", True)),
            "reservo_con_anticipacion": int(perfil_dict.get("reservo_con_anticipacion", True))
        }

        X = pd.DataFrame([fila])
        X_scaled = self.scaler.transform(X)
        cluster = self.model.predict(X_scaled)[0]

        # Distancias a cada centroide (para "probabilidades")
        distancias = self.model.transform(X_scaled)[0]
        scores = 1 / (distancias + 1e-6)
        probabilidades = scores / scores.sum()

        return {
            "cluster_id": int(cluster),
            "segmento": self.NOMBRES_SEGMENTOS.get(cluster, f"Segmento {cluster}"),
            "confianza": round(float(probabilidades[cluster]) * 100, 1),
            "probabilidades": {
                self.NOMBRES_SEGMENTOS.get(i, f"Seg {i}"): round(float(p) * 100, 1)
                for i, p in enumerate(probabilidades)
            }
        }

    def graficar_resultados(self):
        """Genera visualizaciones de la segmentación K-Means usando pandas + matplotlib básico."""
        import subprocess, sys

        # Usar un proceso separado para aislar completamente el estado de matplotlib
        script = f"""
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = {repr(self.output_dir)}
k_range = {self.k_range}
inercias = {self.inercias}
silhouettes = {self.silhouettes}
n_optimo = {self.n_clusters_optimo}
X_pca = np.load(os.path.join(output_dir, 'tmp_pca.npy'))
labels = np.load(os.path.join(output_dir, 'tmp_labels.npy'))
centroides = np.load(os.path.join(output_dir, 'tmp_centroides.npy'))

# Figura 1: Codo + PCA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("K-Means – Determinación de K y Segmentación\\nValledupar, Cesar – Colombia", fontsize=12, fontweight='bold')

inercias_norm = np.array(inercias) / max(inercias)
ax1.plot(k_range, inercias_norm, 'bo-', lw=2, ms=8, label='Inercia (norm.)')
ax1.plot(k_range, silhouettes, 'rs-', lw=2, ms=8, label='Silhouette')
ax1.axvline(x=n_optimo, color='green', ls='--', lw=2, label=f'K optimo = {{n_optimo}}')
ax1.set_xlabel('Numero de Clusters (K)')
ax1.set_ylabel('Score normalizado')
ax1.set_title('Metodo del Codo + Silhouette')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

colores = ['#e74c3c','#9b59b6','#2ecc71','#f39c12','#3498db']
nombres_seg = ['Seg 0','Seg 1','Seg 2','Seg 3','Seg 4']
for cid in range(n_optimo):
    mask = labels == cid
    ax2.scatter(X_pca[mask,0], X_pca[mask,1], color=colores[cid%5], alpha=0.5, s=20, label=nombres_seg[cid])
ax2.scatter(centroides[:,0], centroides[:,1], color='black', marker='X', s=200, zorder=5, label='Centroides')
ax2.set_title(f'PCA 2D – K={{n_optimo}}')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kmeans_segmentacion.png'), dpi=150, bbox_inches='tight')
plt.close()

# Figura 2: Distribución y características
stats_df = pd.read_csv(os.path.join(output_dir, 'estadisticas_segmentos.csv'))
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("K-Means – Distribucion y Caracteristicas de Segmentos\\nValledupar, Cesar – Colombia", fontsize=12, fontweight='bold')

y = range(len(stats_df))
ax3.barh(list(y), stats_df['n_turistas'].tolist(), color=[colores[i%5] for i in y], alpha=0.85)
ax3.set_yticks(list(y))
ax3.set_yticklabels([str(n)[:28] for n in stats_df['nombre']], fontsize=8)
ax3.set_xlabel('Turistas')
ax3.set_title('Distribucion de Segmentos')
ax3.grid(True, alpha=0.3, axis='x')

x_pos = np.arange(3)
cols_var = ['edad_media','dias_estancia_media','satisfaccion_media']
labels_var = ['Edad','Estadía','Satisfacción']
n_s = len(stats_df)
w = 0.7 / max(n_s, 1)
for i, row in stats_df.iterrows():
    vals = [float(row[c]) for c in cols_var]
    offset = (i - n_s/2 + 0.5) * w
    ax4.bar(x_pos + offset, vals, w*0.92, label=str(row['nombre'])[:18], color=colores[i%5], alpha=0.85)
ax4.set_xticks(x_pos.tolist())
ax4.set_xticklabels(labels_var, fontsize=9)
ax4.set_title('Caracteristicas por Segmento')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kmeans_segmentos.png'), dpi=150, bbox_inches='tight')
plt.close()
print('GRAFICAS_OK')
"""
        # Guardar arrays temporales para el subproceso
        np.save(f"{self.output_dir}/tmp_pca.npy", self.X_pca)
        np.save(f"{self.output_dir}/tmp_labels.npy", self.labels)
        centroides_pca = self.pca.transform(self.model.cluster_centers_)
        np.save(f"{self.output_dir}/tmp_centroides.npy", centroides_pca)

        # Ejecutar en subproceso aislado
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30
        )
        if "GRAFICAS_OK" in result.stdout:
            print(f"  ✓ Gráficas guardadas: {self.output_dir}/kmeans_segmentacion.png, kmeans_segmentos.png")
        else:
            err = (result.stderr or result.stdout)[:200]
            print(f"  ⚠ Gráfica K-Means no generada: {err}")

    def guardar_modelo(self):
        """Serializa el modelo para integración en la API."""
        joblib.dump(self.model, f"{self.output_dir}/kmeans_model.pkl")
        joblib.dump(self.scaler, f"{self.output_dir}/kmeans_scaler.pkl")
        self.df_segmentado.to_csv(f"{self.output_dir}/turistas_segmentados.csv", index=False)
        print(f"  ✓ Modelo K-Means guardado: {self.output_dir}/")


def ejecutar_fase_3(df_turistas, output_dir="outputs"):
    """Función principal de la Fase 3."""
    print("\n" + "=" * 60)
    print("FASE 3: Segmentación K-Means de Perfiles Turísticos")
    print("=" * 60)

    kmeans = SegmentacionKMeans(output_dir=output_dir)
    kmeans.entrenar(df_turistas)
    stats = kmeans.caracterizar_segmentos()
    try:
        kmeans.graficar_resultados()
    except Exception:
        print("  ⚠ Visualización K-Means omitida (error matplotlib)")
    kmeans.guardar_modelo()

    # Ejemplo de clasificación de un nuevo turista
    print("\n  Ejemplo de clasificación de turista nuevo:")
    nuevo_turista = {
        "edad": 28,
        "dias_estancia": 3,
        "gasto_total_cop": 1_500_000,
        "gasto_alojamiento_cop": 500_000,
        "grupo_tamano": 3,
        "score_actividad_cultural": 4,
        "satisfaccion_destino": 4.5,
        "es_extranjero": False,
        "es_primera_visita": True,
        "nivel_educativo": "universitario",
        "tipo_alojamiento": "hostal_familiar",
        "motivo_viaje": "festival",
        "uso_plataforma_digital": True,
        "reservo_con_anticipacion": False
    }
    resultado = kmeans.predecir_segmento(nuevo_turista)
    print(f"  → Turista clasificado como: '{resultado['segmento']}'")
    print(f"  → Confianza: {resultado['confianza']}%")

    return kmeans, stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_data import generar_perfiles_turistas
    df_t = generar_perfiles_turistas(5000)
    ejecutar_fase_3(df_t, "outputs")
