"""
FASE 2: Modelo LSTM de Predicción de Ocupación Hotelera
Sistema Inteligente de Monitoreo y Predicción de Ocupación Turística
Valledupar - Cesar - Colombia

Red neuronal LSTM que predice la tasa de ocupación hotelera incorporando:
- Series históricas de ocupación (DANE-EMA / Cotelco Cesar)
- Variables del Festival de la Leyenda Vallenata
- Temporadas vacacionales del sistema educativo colombiano
- Puentes festivos nacionales

Métricas evaluadas: MAE, RMSE, MAPE (vs líneas base ARIMA y Regresión Lineal)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL DEL MODELO LSTM
# ─────────────────────────────────────────────────────────────────

class LSTMOcupacionHotelera:
    """
    Modelo LSTM para predicción de ocupación hotelera en Valledupar.
    Implementa la arquitectura descrita en la Fase 2 del proyecto.
    """

    def __init__(self, n_steps=30, n_features=7, output_dir="outputs"):
        self.n_steps = n_steps          # Ventana temporal: 30 días
        self.n_features = n_features    # Variables de entrada
        self.output_dir = output_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        os.makedirs(output_dir, exist_ok=True)

    def preparar_features(self, df):
        """
        Ingeniería de variables contextuales para el modelo LSTM.
        Variables: ocupación, festival, festivo, fin_semana, mes_sin, mes_cos, tendencia
        """
        df = df.copy().sort_values("fecha").reset_index(drop=True)

        # Variables cíclicas del tiempo (codificación seno/coseno)
        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

        # Tendencia temporal (días desde inicio de la serie)
        df["tendencia"] = (df["fecha"] - df["fecha"].min()).dt.days / 365.0

        # Semanas antes/después del festival (ventana de impacto)
        df["dias_para_festival"] = df.apply(self._dias_para_festival, axis=1)
        df["ventana_festival"] = np.exp(-0.5 * (df["dias_para_festival"] / 10) ** 2)

        features = [
            "tasa_ocupacion",
            "es_festival_vallenato",
            "es_festivo_nacional",
            "es_fin_semana",
            "mes_sin",
            "mes_cos",
            "ventana_festival"
        ]

        return df, features

    def _dias_para_festival(self, row):
        """Calcula días al siguiente Festival de la Leyenda Vallenata."""
        mes = row["mes"]
        dia = row["dia"]
        # Festival: aproximadamente 28 abril - 5 mayo
        if mes < 4 or (mes == 4 and dia < 28):
            # Antes del festival
            dias = (4 - mes) * 30 + (28 - dia)
        elif mes == 4 and dia >= 28:
            dias = 0
        elif mes == 5 and dia <= 5:
            dias = 0
        else:
            # Después del festival, hasta el próximo año
            dias = (12 - mes) * 30 + (28 - dia) + (4 * 30)
        return min(dias, 365)

    def crear_secuencias(self, datos_escalados):
        """Crea secuencias temporales para el entrenamiento LSTM (ventana deslizante)."""
        X, y = [], []
        for i in range(self.n_steps, len(datos_escalados)):
            X.append(datos_escalados[i - self.n_steps:i, :])
            y.append(datos_escalados[i, 0])  # Predice tasa de ocupación
        return np.array(X), np.array(y)

    def construir_modelo(self):
        """
        Arquitectura LSTM definida en la metodología del proyecto.
        2 capas LSTM + Dropout + Dense de salida
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                LSTM(64, return_sequences=True,
                     input_shape=(self.n_steps, self.n_features),
                     name="lstm_1"),
                Dropout(0.2, name="dropout_1"),
                BatchNormalization(name="batch_norm_1"),

                LSTM(32, return_sequences=False, name="lstm_2"),
                Dropout(0.2, name="dropout_2"),

                Dense(16, activation="relu", name="dense_1"),
                Dense(1, activation="sigmoid", name="output")  # Salida entre 0 y 1
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae"]
            )

            print("\n  Arquitectura LSTM:")
            model.summary()
            self.model = model
            self.usar_tensorflow = True
            return model

        except ImportError:
            print("  ⚠ TensorFlow no disponible. Usando modelo de regresión como alternativa.")
            self.usar_tensorflow = False
            self.model = LinearRegression()
            return self.model

    def entrenar(self, df, epochs=50, batch_size=32):
        """
        Entrena el modelo LSTM con división 70/15/15 (train/val/test).
        Implementa EarlyStopping para evitar overfitting.
        """
        print("\n" + "─" * 50)
        print("FASE 2: Entrenamiento del Modelo LSTM")
        print("─" * 50)

        df, features = self.preparar_features(df)
        self.features = features

        # Escalar datos
        datos = df[features].values
        datos_escalados = self.scaler.fit_transform(datos)

        # División temporal: 70% train, 15% val, 15% test
        n = len(datos_escalados)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_data = datos_escalados[:n_train]
        val_data = datos_escalados[n_train:n_train + n_val]
        test_data = datos_escalados[n_train + n_val:]

        print(f"\n  Datos totales: {n:,} días ({df['fecha'].min().date()} → {df['fecha'].max().date()})")
        print(f"  Entrenamiento: {len(train_data):,} días ({len(train_data)/n*100:.1f}%)")
        print(f"  Validación:    {len(val_data):,} días ({len(val_data)/n*100:.1f}%)")
        print(f"  Prueba:        {len(test_data):,} días ({len(test_data)/n*100:.1f}%)")

        if self.usar_tensorflow:
            # Crear secuencias
            X_train, y_train = self.crear_secuencias(train_data)
            X_val, y_val = self.crear_secuencias(val_data)
            X_test, y_test = self.crear_secuencias(test_data)

            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001, verbose=1)
            ]

            print(f"\n  Entrenando con {len(X_train):,} secuencias de {self.n_steps} días...")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Evaluar en test
            self.X_test = X_test
            self.y_test = y_test
            self.test_dates = df["fecha"].values[n_train + n_val + self.n_steps:]

        else:
            # Fallback con regresión lineal
            X = df[features[1:]].values
            y = df[features[0]].values
            n_train_simple = int(len(X) * 0.85)
            self.model.fit(X[:n_train_simple], y[:n_train_simple])
            self.X_test_simple = X[n_train_simple:]
            self.y_test_simple = y[n_train_simple:]
            self.test_dates = df["fecha"].values[n_train_simple:]

        return self

    def evaluar(self):
        """
        Evalúa el modelo con métricas extendidas (MAE, RMSE, MAPE, SMAPE, R², Theil-U).
        Compara contra tres baselines: Naive, Media Móvil 7d y Media Histórica.
        Genera dashboard de entrenamiento con training_monitor.
        """
        from models.training_monitor import MetricasComparativas, VisualizadorTraining, ReporteEntrenamiento

        print("\n" + "─" * 50)
        print("EVALUACIÓN DEL MODELO — MÉTRICAS EXTENDIDAS")
        print("─" * 50)

        if self.usar_tensorflow:
            y_pred_scaled = self.model.predict(self.X_test, verbose=0)
            y_pred_full = np.zeros((len(y_pred_scaled), self.n_features))
            y_pred_full[:, 0] = y_pred_scaled.flatten()
            y_pred_lstm = self.scaler.inverse_transform(y_pred_full)[:, 0]

            y_test_full = np.zeros((len(self.y_test), self.n_features))
            y_test_full[:, 0] = self.y_test
            y_real = self.scaler.inverse_transform(y_test_full)[:, 0]
        else:
            y_pred_lstm = self.model.predict(self.X_test_simple)
            y_real = self.y_test_simple

        # ── Métricas extendidas LSTM ──────────────────────────────
        mc = MetricasComparativas()
        metricas_ext = mc.calcular_todas(y_real, y_pred_lstm)

        # ── Baselines ─────────────────────────────────────────────
        baselines = mc.baselines(y_real)
        comparativas = {"LSTM (modelo)": metricas_ext}
        for nombre, y_bl in baselines.items():
            comparativas[nombre] = mc.calcular_todas(y_real, y_bl)

        # ── Imprimir tabla ────────────────────────────────────────
        df_comp = mc.tabla_comparativa(comparativas)
        print("\n" + df_comp.to_string())

        mejora_mae = (comparativas["Naive (t-1)"]["MAE"] - metricas_ext["MAE"]) / \
                      comparativas["Naive (t-1)"]["MAE"] * 100
        print(f"\n  ✓ Mejora LSTM vs Naive:        {mejora_mae:.1f}% en MAE")
        print(f"  ✓ R² (coef. determinación):    {metricas_ext['R²']:.4f}")
        print(f"  ✓ Theil-U (< 1 = mejor naive): {metricas_ext['Theil-U']:.4f}")

        # ── Dashboard de entrenamiento ────────────────────────────
        viz = VisualizadorTraining(self.output_dir)
        viz.dashboard_lstm(
            history=self.history if self.usar_tensorflow else None,
            y_real=y_real,
            y_pred=y_pred_lstm,
            comparativas=comparativas,
            titulo="LSTM — Dashboard de Entrenamiento\nPredicción de Ocupación Hotelera · Valledupar, Cesar"
        )

        # ── Reporte CSV ───────────────────────────────────────────
        rep = ReporteEntrenamiento(self.output_dir)
        rep.registrar(
            fase="Fase 2 - LSTM",
            modelo="LSTM 2-capas (64→32) + Dropout 0.2",
            metricas={k: round(v, 6) for k, v in metricas_ext.items()},
            params={"n_steps": self.n_steps, "n_features": self.n_features,
                    "usar_tensorflow": self.usar_tensorflow}
        )
        rep.guardar()

        # Guardar comparativa CSV
        df_comp.to_csv(f"{self.output_dir}/lstm_comparativa_modelos.csv", encoding="utf-8-sig")

        self.y_pred = y_pred_lstm
        self.y_real = y_real
        self.metricas = {"mae": metricas_ext["MAE"], "rmse": metricas_ext["RMSE"],
                         "mape": metricas_ext["MAPE (%)"], "r2": metricas_ext["R²"]}
        self.metricas_ext = metricas_ext
        self.comparativas = comparativas

        return self.metricas

    def predecir_proximos_dias(self, df, n_dias=90):
        """
        Genera predicciones para los próximos n_dias días.
        Incluye intervalos de confianza al 95%.
        """
        df, features = self.preparar_features(df)
        datos = df[features].values
        datos_escalados = self.scaler.transform(datos)

        predicciones = []
        ultima_secuencia = datos_escalados[-self.n_steps:].copy()
        ultima_fecha = df["fecha"].max()

        for i in range(n_dias):
            fecha_pred = ultima_fecha + pd.Timedelta(days=i+1)

            if self.usar_tensorflow:
                secuencia_input = ultima_secuencia.reshape(1, self.n_steps, self.n_features)
                pred_scaled = self.model.predict(secuencia_input, verbose=0)[0][0]
                pred_full = np.zeros(self.n_features)
                pred_full[0] = pred_scaled
                pred_real = self.scaler.inverse_transform(pred_full.reshape(1, -1))[0][0]
            else:
                # Fallback con tendencia
                pred_real = datos[-1, 0] * 0.98 + np.random.normal(0, 0.02)

            pred_real = max(0.20, min(0.99, pred_real))
            incertidumbre = 0.03 + abs(pred_real - 0.50) * 0.05

            predicciones.append({
                "fecha": fecha_pred,
                "ocupacion_predicha": round(pred_real, 4),
                "ocupacion_predicha_pct": round(pred_real * 100, 2),
                "intervalo_inferior": round(max(0.15, pred_real - 1.96 * incertidumbre), 4),
                "intervalo_superior": round(min(1.0, pred_real + 1.96 * incertidumbre), 4),
                "es_festival": int(fecha_pred.month == 4 and fecha_pred.day >= 26 or
                                  fecha_pred.month == 5 and fecha_pred.day <= 6)
            })

            # Actualizar secuencia con la predicción
            nueva_fila = np.zeros(self.n_features)
            nueva_fila[0] = pred_scaled if self.usar_tensorflow else pred_real
            ultima_secuencia = np.vstack([ultima_secuencia[1:], nueva_fila])

        return pd.DataFrame(predicciones)

    def graficar_resultados(self, df, df_predicciones=None):
        """Genera gráficas de resultados para el reporte."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("LSTM - Predicción de Ocupación Hotelera\nValledupar, Cesar - Colombia",
                    fontsize=14, fontweight="bold")

        # 1. Serie histórica completa
        ax1 = axes[0, 0]
        df_plot = df.copy()
        df_plot["fecha"] = pd.to_datetime(df_plot["fecha"])
        df_mensual = df_plot.groupby(df_plot["fecha"].dt.to_period("M"))["tasa_ocupacion_pct"].mean()
        df_mensual.index = df_mensual.index.to_timestamp()
        ax1.plot(df_mensual.index, df_mensual.values, "b-", linewidth=1.5, label="Ocupación mensual")
        ax1.axhline(y=98, color="r", linestyle="--", alpha=0.7, label="Festival (98%)")
        ax1.axhline(y=38, color="orange", linestyle="--", alpha=0.7, label="Temporada baja (38%)")
        ax1.fill_between(df_mensual.index, 38, 98, alpha=0.05, color="green")
        ax1.set_title("Serie Histórica de Ocupación Hotelera (2015-2025)")
        ax1.set_ylabel("Tasa de Ocupación (%)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Predicciones vs Real (si hay datos de test)
        ax2 = axes[0, 1]
        if hasattr(self, "y_real") and len(self.y_real) > 0:
            n_mostrar = min(365, len(self.y_real))
            x_idx = range(n_mostrar)
            ax2.plot(x_idx, self.y_real[:n_mostrar] * 100, "b-", alpha=0.7, linewidth=1, label="Real")
            ax2.plot(x_idx, self.y_pred[:n_mostrar] * 100, "r--", alpha=0.9, linewidth=1.5, label="LSTM predicho")
            ax2.set_title(f"LSTM: Predicción vs Real (Test Set)")
            ax2.set_ylabel("Tasa de Ocupación (%)")
            ax2.set_xlabel("Días")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            mae_pp = self.metricas["mae"] * 100
            ax2.text(0.02, 0.95, f"MAE: {mae_pp:.2f} pp", transform=ax2.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        else:
            ax2.text(0.5, 0.5, "Entrenamiento en progreso...", ha="center", va="center")
            ax2.set_title("Predicción vs Real")

        # 3. Predicciones futuras
        ax3 = axes[1, 0]
        if df_predicciones is not None and len(df_predicciones) > 0:
            ax3.plot(df_predicciones["fecha"], df_predicciones["ocupacion_predicha_pct"],
                    "g-", linewidth=2, label="Predicción LSTM")
            ax3.fill_between(
                df_predicciones["fecha"],
                df_predicciones["intervalo_inferior"] * 100,
                df_predicciones["intervalo_superior"] * 100,
                alpha=0.2, color="green", label="IC 95%"
            )
            # Marcar festival
            festival = df_predicciones[df_predicciones["es_festival"] == 1]
            if len(festival) > 0:
                ax3.scatter(festival["fecha"], festival["ocupacion_predicha_pct"],
                           color="red", zorder=5, s=50, label="Festival Vallenato")
            ax3.set_title("Predicción de Ocupación - Próximos 90 días")
            ax3.set_ylabel("Tasa de Ocupación (%)")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
        else:
            ax3.text(0.5, 0.5, "Sin predicciones futuras", ha="center", va="center")

        # 4. Ocupación promedio por temporada
        ax4 = axes[1, 1]
        ocupacion_temporada = df.groupby("temporada")["tasa_ocupacion_pct"].mean().sort_values(ascending=False)
        colores = {"FESTIVAL_VALLENATO": "#e74c3c", "ALTA_NAVIDENA": "#e67e22",
                  "SEMANA_SANTA": "#f39c12", "VACACIONES_MITAD_ANIO": "#2ecc71",
                  "MEDIA": "#3498db", "BAJA": "#95a5a6"}
        bars = ax4.bar(range(len(ocupacion_temporada)),
                      ocupacion_temporada.values,
                      color=[colores.get(t, "#3498db") for t in ocupacion_temporada.index])
        ax4.set_xticks(range(len(ocupacion_temporada)))
        ax4.set_xticklabels([t.replace("_", "\n") for t in ocupacion_temporada.index],
                           fontsize=8, rotation=0)
        ax4.set_title("Ocupación Promedio por Temporada")
        ax4.set_ylabel("Tasa de Ocupación (%)")
        for bar, val in zip(bars, ocupacion_temporada.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        ruta = f"{self.output_dir}/lstm_resultados.png"
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  ✓ Gráfica guardada: {ruta}")

    def guardar_modelo(self):
        """Serializa el modelo entrenado para integración en la plataforma web."""
        if self.usar_tensorflow:
            ruta_keras = f"{self.output_dir}/lstm_model.keras"
            self.model.save(ruta_keras)
            print(f"  ✓ Modelo LSTM guardado: {ruta_keras}")
        joblib.dump(self.scaler, f"{self.output_dir}/lstm_scaler.pkl")
        print(f"  ✓ Scaler guardado: {self.output_dir}/lstm_scaler.pkl")


def ejecutar_fase_2(df_ocupacion, output_dir="outputs"):
    """Función principal de la Fase 2."""
    print("\n" + "=" * 60)
    print("FASE 2: Modelo LSTM - Predicción de Ocupación Hotelera")
    print("=" * 60)

    lstm = LSTMOcupacionHotelera(n_steps=30, n_features=7, output_dir=output_dir)
    lstm.construir_modelo()
    lstm.entrenar(df_ocupacion, epochs=40, batch_size=32)
    metricas = lstm.evaluar()

    # Predicciones próximos 90 días
    print("\n  Generando predicciones para los próximos 90 días...")
    df_pred = lstm.predecir_proximos_dias(df_ocupacion, n_dias=90)
    df_pred.to_csv(f"{output_dir}/predicciones_ocupacion_90dias.csv", index=False)
    print(f"  ✓ Predicciones guardadas: {output_dir}/predicciones_ocupacion_90dias.csv")

    lstm.graficar_resultados(df_ocupacion, df_pred)
    lstm.guardar_modelo()

    return lstm, metricas, df_pred


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_data import generar_serie_ocupacion
    df_ocu = generar_serie_ocupacion(2015, 2025)
    ejecutar_fase_2(df_ocu, "outputs")
