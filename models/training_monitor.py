"""
Monitor de Entrenamiento - Sistema Turístico Valledupar
=======================================================
Módulo de logging, métricas ampliadas y visualización avanzada
del proceso de entrenamiento de los tres modelos de IA.

Clases:
  - EntrenamientoLogger    : Logger con barra de progreso y timestamps
  - MetricasComparativas   : Tabla comparativa LSTM vs baselines
  - VisualizadorTraining   : Dashboard de curvas de aprendizaje
  - ReporteEntrenamiento   : Resumen ejecutivo al finalizar cada fase
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ──────────────────────────────────────────────────────────────────
# LOGGER CON TIMESTAMPS Y NIVELES
# ──────────────────────────────────────────────────────────────────

class EntrenamientoLogger:
    """
    Logger enriquecido para el proceso de entrenamiento.
    Escribe en consola y en un archivo .log persistente.
    """

    NIVELES = {
        "INFO":    ("\033[94m", "ℹ"),    # Azul
        "OK":      ("\033[92m", "✓"),    # Verde
        "WARN":    ("\033[93m", "⚠"),    # Amarillo
        "ERROR":   ("\033[91m", "✗"),    # Rojo
        "FASE":    ("\033[95m", "▶"),    # Magenta
        "METRICA": ("\033[96m", "◆"),    # Cyan
    }
    RESET = "\033[0m"

    def __init__(self, nombre_fase: str, output_dir: str = "outputs"):
        self.nombre_fase = nombre_fase
        self.output_dir = output_dir
        self.t_inicio = time.time()
        self.eventos = []
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "entrenamiento.log")
        self._escribir_separador()

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _escribir_separador(self):
        sep = "=" * 65
        msg = f"\n{sep}\n  {self.nombre_fase}  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{sep}"
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log(self, mensaje: str, nivel: str = "INFO"):
        color, icono = self.NIVELES.get(nivel, ("\033[0m", "·"))
        ts = self._ts()
        elapsed = time.time() - self.t_inicio
        linea_consola = f"  {color}{icono}{self.RESET} [{ts}] {mensaje}"
        linea_log     = f"  {icono} [{ts}] (+{elapsed:.1f}s) {mensaje}"
        print(linea_consola)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(linea_log + "\n")
        self.eventos.append({"ts": ts, "nivel": nivel, "msg": mensaje, "elapsed": elapsed})

    def progreso(self, paso: int, total: int, extra: str = ""):
        barra_len = 30
        lleno = int(barra_len * paso / total)
        barra = "█" * lleno + "░" * (barra_len - lleno)
        pct = paso / total * 100
        print(f"\r  [{barra}] {pct:5.1f}%  {extra}", end="", flush=True)
        if paso == total:
            print()

    def tabla_metricas(self, datos: dict, titulo: str = "Métricas"):
        """Imprime tabla formateada de métricas."""
        ancho = 55
        print(f"\n  {'─'*ancho}")
        print(f"  {'':3}{titulo:^{ancho-3}}")
        print(f"  {'─'*ancho}")
        for modelo, metricas in datos.items():
            print(f"  {'':3}{modelo}")
            for k, v in metricas.items():
                if isinstance(v, float):
                    print(f"  {'':6}{k:<20} {v:.6f}")
                else:
                    print(f"  {'':6}{k:<20} {v}")
        print(f"  {'─'*ancho}\n")

    def finalizar(self):
        elapsed = time.time() - self.t_inicio
        self.log(f"Fase completada en {elapsed:.1f} segundos", "OK")
        return elapsed


# ──────────────────────────────────────────────────────────────────
# MÉTRICAS COMPARATIVAS EXTENDIDAS
# ──────────────────────────────────────────────────────────────────

class MetricasComparativas:
    """
    Calcula métricas extendidas y compara modelos contra múltiples baselines.

    Métricas implementadas:
      - MAE, RMSE, MAPE  (error de predicción)
      - R²               (coeficiente de determinación)
      - SMAPE            (error porcentual absoluto simétrico)
      - IA               (índice de Theil / improvement accuracy)
    """

    @staticmethod
    def calcular_todas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        eps = 1e-8

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + eps)

        # Índice de Theil: U < 1 es mejor que la predicción naive
        naive = np.roll(y_true, 1)
        naive[0] = y_true[0]
        theil_num = np.sqrt(np.mean((y_true - y_pred) ** 2))
        theil_den = np.sqrt(np.mean((y_true - naive) ** 2))
        theil_u = theil_num / (theil_den + eps)

        return {
            "MAE":      mae,
            "RMSE":     rmse,
            "MAPE (%)": mape,
            "SMAPE (%)":smape,
            "R²":       r2,
            "Theil-U":  theil_u,
        }

    @staticmethod
    def tabla_comparativa(resultados: dict) -> pd.DataFrame:
        """
        resultados: {nombre_modelo: {metrica: valor}}
        Retorna DataFrame con fila por modelo y columna por métrica.
        """
        df = pd.DataFrame(resultados).T
        df.index.name = "Modelo"
        return df.round(5)

    @staticmethod
    def baselines(y_true: np.ndarray) -> dict:
        """Genera predicciones de tres baselines clásicos."""
        y = np.array(y_true, dtype=float)

        # 1. Naive (persistencia): predice el valor anterior
        naive = np.roll(y, 1); naive[0] = y[0]

        # 2. Media móvil 7 días
        ma7 = np.convolve(y, np.ones(7)/7, mode='same')

        # 3. Media histórica constante
        media = np.full_like(y, np.mean(y))

        return {
            "Naive (t-1)":       naive,
            "Media Móvil 7d":    ma7,
            "Media Histórica":   media,
        }


# ──────────────────────────────────────────────────────────────────
# VISUALIZADOR DE CURVAS DE ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────

class VisualizadorTraining:
    """
    Genera dashboards de entrenamiento con:
    - Curvas loss/val_loss
    - Distribución de errores
    - Scatter real vs predicho
    - Tabla comparativa de modelos
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def dashboard_lstm(self, history, y_real, y_pred, comparativas: dict,
                       titulo: str = "LSTM — Dashboard de Entrenamiento"):
        """
        Genera dashboard de 6 paneles para la Fase 2 (LSTM).
        """
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(titulo, fontsize=15, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

        colores = {"LSTM": "#e74c3c", "Naive (t-1)": "#95a5a6",
                   "Media Móvil 7d": "#3498db", "Media Histórica": "#2ecc71"}

        # ── Panel 1: Curvas de pérdida ──────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        if history and hasattr(history, "history"):
            epochs = range(1, len(history.history["loss"]) + 1)
            ax1.plot(epochs, history.history["loss"],   "b-", lw=1.5, label="Train Loss")
            ax1.plot(epochs, history.history["val_loss"],"r--", lw=1.5, label="Val Loss")
            ax1.set_title("Curvas de Aprendizaje", fontweight="bold")
            ax1.set_xlabel("Épocas"); ax1.set_ylabel("MSE Loss")
            ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
            # Marcar mejor época
            mejor_ep = np.argmin(history.history["val_loss"]) + 1
            ax1.axvline(mejor_ep, color="green", linestyle=":", alpha=0.8,
                        label=f"Mejor época ({mejor_ep})")
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, "Modelo sklearn\n(sin curvas Keras)", ha="center", va="center",
                    transform=ax1.transAxes, fontsize=10, color="gray")
            ax1.set_title("Curvas de Aprendizaje", fontweight="bold")

        # ── Panel 2: Real vs Predicho ───────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        n = min(365, len(y_real))
        ax2.plot(range(n), y_real[:n]*100, "b-",  alpha=0.6, lw=1,   label="Real")
        ax2.plot(range(n), y_pred[:n]*100, "r--", alpha=0.9, lw=1.5, label="LSTM")
        ax2.set_title("Real vs Predicho (Test)", fontweight="bold")
        ax2.set_xlabel("Días"); ax2.set_ylabel("Ocupación (%)")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

        # ── Panel 3: Scatter real vs predicho ──────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(y_real*100, y_pred*100, alpha=0.3, s=8, c="#e74c3c")
        mn, mx = min(y_real.min(), y_pred.min())*100, max(y_real.max(), y_pred.max())*100
        ax3.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="Predicción perfecta")
        ax3.set_title("Scatter: Real vs LSTM", fontweight="bold")
        ax3.set_xlabel("Real (%)"); ax3.set_ylabel("Predicho (%)")
        ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

        # ── Panel 4: Distribución de errores ───────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        errores = (y_pred - y_real) * 100
        ax4.hist(errores, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
        ax4.axvline(0, color="red", lw=1.5, linestyle="--")
        ax4.axvline(errores.mean(), color="orange", lw=1.5, linestyle="-",
                   label=f"Media: {errores.mean():.2f} pp")
        ax4.set_title("Distribución de Errores", fontweight="bold")
        ax4.set_xlabel("Error (pp)"); ax4.set_ylabel("Frecuencia")
        ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

        # ── Panel 5: Comparativa de métricas (bar chart) ───────────
        ax5 = fig.add_subplot(gs[1, 1])
        if comparativas:
            modelos = list(comparativas.keys())
            maes    = [comparativas[m].get("MAE", 0) * 100 for m in modelos]
            bar_colors = [colores.get(m, "#7f8c8d") for m in modelos]
            bars = ax5.bar(range(len(modelos)), maes, color=bar_colors, edgecolor="white")
            ax5.set_xticks(range(len(modelos)))
            ax5.set_xticklabels(modelos, rotation=15, ha="right", fontsize=8)
            ax5.set_title("Comparativa MAE (pp)", fontweight="bold")
            ax5.set_ylabel("MAE (puntos porcentuales)")
            for bar, v in zip(bars, maes):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax5.grid(alpha=0.3, axis="y")

        # ── Panel 6: Tabla de métricas ──────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        if comparativas:
            col_labels = ["Modelo", "MAE(pp)", "RMSE(pp)", "MAPE(%)", "R²"]
            rows = []
            for m, met in comparativas.items():
                rows.append([
                    m,
                    f"{met.get('MAE',0)*100:.3f}",
                    f"{met.get('RMSE',0)*100:.3f}",
                    f"{met.get('MAPE (%)',0):.2f}",
                    f"{met.get('R²',0):.4f}",
                ])
            tbl = ax6.table(cellText=rows, colLabels=col_labels,
                           loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.1, 1.6)
            # Color verde para LSTM (primera fila de datos)
            for j in range(len(col_labels)):
                tbl[1, j].set_facecolor("#ffeaa7")
            ax6.set_title("Tabla Comparativa de Modelos", fontweight="bold", pad=10)

        plt.savefig(f"{self.output_dir}/lstm_training_dashboard.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Dashboard LSTM guardado: {self.output_dir}/lstm_training_dashboard.png")

    def dashboard_kmeans(self, inertias: list, silhouettes: list,
                         X_pca: np.ndarray, labels: np.ndarray,
                         stats_segmentos: dict, titulo: str = "K-Means — Dashboard"):
        """
        Genera dashboard de 4 paneles para la Fase 3 (K-Means).
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(titulo, fontsize=14, fontweight="bold")
        ks = range(2, 2 + len(inertias))

        # ── Panel 1: Método del Codo ─────────────────────────────
        ax1 = axes[0, 0]
        ax1.plot(ks, inertias, "bo-", lw=2, ms=6)
        ax1.fill_between(ks, inertias, alpha=0.1, color="blue")
        ax1.set_title("Método del Codo (Inercia)", fontweight="bold")
        ax1.set_xlabel("Número de Clusters K"); ax1.set_ylabel("Inercia (WCSS)")
        ax1.grid(alpha=0.3)

        # ── Panel 2: Coeficiente Silhouette ──────────────────────
        ax2 = axes[0, 1]
        sil_colors = ["green" if s == max(silhouettes) else "#3498db" for s in silhouettes]
        bars = ax2.bar(ks, silhouettes, color=sil_colors, edgecolor="white")
        ax2.axhline(0.5, color="red", lw=1.5, linestyle="--", label="Umbral buena sep. (0.50)")
        ax2.set_title("Coeficiente de Silhouette por K", fontweight="bold")
        ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")
        for bar, v in zip(bars, silhouettes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # ── Panel 3: Scatter PCA ──────────────────────────────────
        ax3 = axes[1, 0]
        colores_seg = ["#e74c3c", "#9b59b6", "#2ecc71", "#f39c12", "#3498db"]
        for i in range(len(np.unique(labels))):
            mask = labels == i
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colores_seg[i % len(colores_seg)],
                       alpha=0.5, s=10,
                       label=stats_segmentos.get(i, {}).get("nombre", f"Seg {i}")[:20])
        ax3.set_title("Segmentos en Espacio PCA (2D)", fontweight="bold")
        ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
        ax3.legend(fontsize=7, loc="upper right"); ax3.grid(alpha=0.3)

        # ── Panel 4: Distribución de segmentos ───────────────────
        ax4 = axes[1, 1]
        nombres = [stats_segmentos.get(i, {}).get("nombre", f"Seg {i}")[:22]
                   for i in sorted(stats_segmentos.keys())]
        pcts    = [stats_segmentos.get(i, {}).get("porcentaje", 0)
                   for i in sorted(stats_segmentos.keys())]
        bars4 = ax4.barh(range(len(nombres)), pcts,
                        color=colores_seg[:len(nombres)], edgecolor="white")
        ax4.set_yticks(range(len(nombres)))
        ax4.set_yticklabels(nombres, fontsize=8)
        ax4.set_title("Distribución de Turistas por Segmento", fontweight="bold")
        ax4.set_xlabel("Porcentaje (%)")
        for bar, v in zip(bars4, pcts):
            ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=9)
        ax4.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/kmeans_training_dashboard.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Dashboard K-Means guardado: {self.output_dir}/kmeans_training_dashboard.png")


# ──────────────────────────────────────────────────────────────────
# REPORTE DE ENTRENAMIENTO (RESUMEN EJECUTIVO)
# ──────────────────────────────────────────────────────────────────

class ReporteEntrenamiento:
    """
    Genera y guarda un reporte CSV/TXT de métricas finales de entrenamiento.
    Útil para comparar corridas y documentar el modelo.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.registros = []
        os.makedirs(output_dir, exist_ok=True)

    def registrar(self, fase: str, modelo: str, metricas: dict, params: dict = None):
        entrada = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fase":   fase,
            "modelo": modelo,
            **metricas,
        }
        if params:
            entrada["params"] = str(params)
        self.registros.append(entrada)

    def guardar(self, nombre_archivo: str = "reporte_entrenamiento.csv"):
        if not self.registros:
            return
        df = pd.DataFrame(self.registros)
        ruta = os.path.join(self.output_dir, nombre_archivo)
        df.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(f"\n  ✓ Reporte de entrenamiento guardado: {ruta}")

        # También guardar versión legible .txt
        ruta_txt = ruta.replace(".csv", ".txt")
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write("REPORTE DE ENTRENAMIENTO — SISTEMA TURÍSTICO VALLEDUPAR\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 65 + "\n\n")
            for rec in self.registros:
                f.write(f"Fase:   {rec.get('fase','')}\n")
                f.write(f"Modelo: {rec.get('modelo','')}\n")
                for k, v in rec.items():
                    if k not in ("fecha", "fase", "modelo", "params"):
                        if isinstance(v, float):
                            f.write(f"  {k:<25} {v:.6f}\n")
                        else:
                            f.write(f"  {k:<25} {v}\n")
                if "params" in rec:
                    f.write(f"  Hiperparámetros: {rec['params']}\n")
                f.write("-" * 40 + "\n\n")
        print(f"  ✓ Reporte legible guardado:          {ruta_txt}")
        return df
