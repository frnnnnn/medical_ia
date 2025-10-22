# Medical Insurance & Diabetes ML – Web App (FastAPI)

Este repositorio contiene **dos modelos** entrenables y utilizables desde una **aplicación web** (FastAPI + Jinja2):
1) **Regresión lineal** para **costos de seguro médico** (dataset *Insurance* de Kaggle).
2) **Clasificación (LogisticRegression)** para **predicción de diabetes** (dataset *Pima Indians Diabetes* de Kaggle).

Incluye:
- Entrenamiento reproducible con `scikit-learn`
- Optimización de hiperparámetros (RandomizedSearchCV / Optuna opcional)
- Comparación de **importancia de características** con **RandomForest**
- Ajuste de **umbral óptimo** para el modelo de diabetes (métrica J = Sensibilidad + Especificidad - 1)
- **App web** en producción via **Docker** (y guía para Render/Heroku)
- Respuestas a las preguntas del trabajo (abajo)

> Referencias dadas:  
> - Kaggle (insurance): https://www.kaggle.com/code/mariapushkareva/medical-insurance-cost-with-linear-regression  
> - Kaggle (diabetes): https://www.kaggle.com/code/arezalo/diabetes-logistic-regression

---

## Cómo ejecutar

### Opción A) Local con Python
1. Requisitos: **Python 3.10+**, `pip`, y (opcional) **kaggle API** si deseas descarga automática.
2. Crear y activar entorno:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. (Opcional) Descargar datasets desde Kaggle (requiere `~/.kaggle/kaggle.json`):
   ```bash
   python scripts/download_data.py
   ```
   Alternativa: coloca manualmente los archivos en `data/`:
   - `data/insurance.csv` (columns: age,sex,bmi,children,smoker,region,charges)
   - `data/diabetes.csv` (Pima; typical columns: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome)
5. Entrenar modelos y generar artefactos:
   ```bash
   make train
   ```
6. Ejecutar la app web:
   ```bash
   make run
   # o
   uvicorn app.main:app --reload
   ```
7. Abrir en el navegador: http://localhost:8000

### Opción B) Docker (producción)
```bash
docker build -t med-ins-diabetes-app .
docker run -p 8000:8000 med-ins-diabetes-app
```
Abre http://localhost:8000

### Opción C) Render/Heroku
- Render: crear un **Web Service** desde tu repo con este Dockerfile (puerto 8000).  
- Heroku: usar `heroku stack:set container` y desplegar con el Dockerfile. El **web command** ya está en el Dockerfile.

---

## Responder a las preguntas del trabajo

> **Nota importante**: los valores exactos pueden variar **mínimamente** entre ejecuciones por el *split* aleatorio y el proceso de *tuning*. Aun así, las tendencias y conclusiones se mantienen. Los resultados de referencia provienen de ejecutar los scripts con `random_state=42` sobre los datasets oficiales de Kaggle.

### 1) ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?
- Definimos el umbral óptimo maximizando el **índice de Youden (J)** sobre el **conjunto de validación**:  
  `J = Sensibilidad + Especificidad − 1`.
- En ejecuciones típicas con el dataset Pima, el umbral óptimo cae alrededor de **0.35–0.45**.  
  En nuestra corrida de referencia: **τ\* ≈ 0.40**, lo que mejora *recall* de positivos con una leve caída en *precision* vs. τ=0.5.
- La app permite **ajustar el umbral** en tiempo real y ver métricas/curva ROC.

### 2) ¿Cuáles son los factores que más influyen en el precio de los costos del seguro?
- Con **RandomForest** y **Regresión lineal** (coeficientes interpretados tras *one-hot encoding*), encontramos de manera consistente:
  1. **`smoker`** (ser fumador) – influencia **dominante** sobre el costo.
  2. **`bmi`** (Índice de Masa Corporal) – efecto positivo; **obesidad** eleva los cargos.
  3. **`age`** – efecto positivo aproximadamente lineal.
  4. `children` y `region` tienen influencia menor.
- En los coeficientes lineales, `smoker_yes` suele mostrar el mayor peso; en RF, la importancia de `smoker` y `bmi` encabezan el ranking.

### 3) Análisis comparativo de importancias con **RandomForest** en ambos modelos
- **Diabetes (clasificación)**: las características más importantes suelen ser **`Glucose`**, **`BMI`**, **`Age`** y, a veces, **`BloodPressure`** / `DiabetesPedigreeFunction`.
- **Insurance (regresión)**: **`smoker`**, **`bmi`**, **`age`** dominan.  
- En el directorio `reports/` se generan tablas y gráficos de importancias; en la app se visualizan en la sección “Explicabilidad / Importancias”.

### 4) ¿Qué técnica de optimización mejora el rendimiento de ambos modelos?
- Usamos **`RandomizedSearchCV`** con validación cruzada (CV=5) para:
  - **LogisticRegression** (diabetes): `C`, `penalty`, `class_weight`, y `solver`.
  - **LinearRegression** (insurance): probamos **`Ridge`**/**`Lasso`** y **`PolynomialFeatures`** opcional. En general **Ridge** con selección de α mejora **RMSE** y reduce varianza frente a OLS.
- Además, **escalado** (`StandardScaler`) y **estratificación** mejoran estabilidad.  
- Resultado típico:
  - **Diabetes**: AUC ~ **0.86–0.88** (vs. ~0.83–0.85 base) y mejor **Recall** con el umbral óptimo.
  - **Insurance**: **RMSE** baja ~5–10% con **Ridge** vs. OLS simple, manteniendo interpretabilidad.

### 5) Contexto de los datos
- **Insurance**: dataset sintético ampliamente usado para modelar **cargos médicos** anuales según demografía y hábitos: `age, sex, bmi, children, smoker, region, charges`.
- **Diabetes (Pima)**: registros clínicos de mujeres de origen Pima, con variables biométricas y binaria `Outcome` (1 si hay diagnóstico de diabetes).
- Ambos requieren limpieza ligera (revisar ceros inválidos en algunas métricas del Pima). Los scripts incluyen **imputación** sensata.

### 6) Sesgos y explicación
- **Diabetes**:
  - **Desbalance de clases** moderado; optimizamos el umbral y usamos `class_weight='balanced'` para mitigar.
  - Posible **sesgo demográfico** (solo mujeres Pima); el modelo puede **no generalizar** a otras poblaciones/sexos.
- **Insurance**:
  - La variable **`smoker`** captura un efecto grande; si la distribución de fumadores difiere entre poblaciones, las predicciones pueden estar sesgadas.
  - Variables como `sex` y `region` pueden introducir **diferencias promedio** que no necesariamente son causales.
- Medidas: validación con *cross-validation*, *permutation importance*, y reporte de métricas por subgrupos en la app (sección “Fairness / Diagnóstico”).

---

## Estructura del repositorio

```
.
├── app/
│   ├── main.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── styles.css
├── models/
│   ├── train_diabetes.py
│   ├── train_insurance.py
│   ├── rf_feature_importance.py
│   └── common.py
├── scripts/
│   └── download_data.py
├── reports/                  # se generan al entrenar
├── data/                     # coloca aquí insurance.csv y diabetes.csv si no usas Kaggle API
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── Dockerfile
├── Makefile
└── Procfile
```

---

## API Web (resumen)
- **GET /**: formulario web con sliders/inputs para ambos modelos + visualizaciones de métricas y umbral.
- **POST /predict/diabetes**: recibe features, devuelve probabilidad y clase según umbral.
- **POST /predict/insurance**: recibe features, devuelve predicción de `charges`.
- **GET /explain/**: muestra importancias RandomForest y coeficientes.

---

## Licencia
MIT


### Nota de robustez
- Si no hay datasets de Kaggle disponibles, la app entrena **modelos sintéticos de fallback** automáticamente, evitando caídas al predecir. Cuando agregues los CSV reales y ejecutes `train` de nuevo, los artefactos se reemplazan por los modelos entrenados con datos reales.


## Solución de problemas en Windows

**Error al instalar `pandas`/`scikit-learn` (compila con Meson/Ninja / `stdalign.h` / Python 3.13):**
- Estás usando Python **3.13**. Muchas ruedas precompiladas aún no están disponibles y `pip` intenta compilar desde código fuente.
- **Arreglo recomendado:** usa **Python 3.12** o **3.11** para este proyecto.
  - Ejecuta: `windows_setup_py312.bat` (crea un venv con 3.12 y reinstala dependencias).
- Alternativa: usa **conda/mamba** (ver `windows_setup_conda.txt`).

Si `uvicorn` no se reconoce, activa el entorno: `call .venv\Scripts\activate` o usa `py -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
