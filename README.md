# Bot IA Inmobiliaria — Ames, Iowa

Sistema híbrido de Inteligencia Artificial que estima el precio de una vivienda a
partir de un mensaje escrito en lenguaje natural por Telegram.

Combina dos modelos con responsabilidades separadas: un **modelo de lenguaje**
(Mistral, ejecutado localmente con Ollama) que entiende el mensaje, y un **modelo de
Machine Learning** (Random Forest, entrenado en Google Colab) que calcula el precio.

> **Proyecto Integrador — Actividad individual**
> Universidad · Inteligencia Artificial

---

## Objetivo

Un usuario escribe por Telegram algo como:

> *Tengo una casa en Old Town de 120 metros cuadrados y 25 años.*

Y el sistema responde únicamente con el precio estimado:

> **$171,393**

---

## Arquitectura y flujo de datos

```
   Usuario en Telegram
          │  "Tengo una casa en Old Town de 120 m2 y 25 anios"
          ▼
   ┌──────────────────────┐
   │  Telegram Bot API    │   getUpdates (long polling)
   └──────────┬───────────┘
              ▼
   ┌──────────────────────────────────────────┐
   │  Ollama + Mistral  (local, sin internet) │
   │  Rol: EXTRACTOR NLU                      │
   │  NO calcula precios                      │
   └──────────┬───────────────────────────────┘
              │  {"sector": "OldTown", "metros": 120, "anios": 25}
              ▼
   ┌──────────────────────────────────────────┐
   │  Validación en Python                    │
   │  Normaliza el sector, verifica rangos    │
   │  Si falta un dato, corta aquí            │
   └──────────┬───────────────────────────────┘
              │  {"Neighborhood": "OldTown", "Area_m2": 120.0, "Antiguedad": 25.0}
              ▼
   ┌──────────────────────────────────────────┐
   │  Random Forest Regressor (modelo .pkl)   │
   │  Rol: PREDICTOR                          │
   │  Único componente que produce un precio  │
   └──────────┬───────────────────────────────┘
              │  171393.42
              ▼
   ┌──────────────────────┐
   │  Telegram Bot API    │   sendMessage
   └──────────┬───────────┘
              ▼
   Usuario recibe:  "$171,393"
```

### Por qué dos modelos y no uno

| | Mistral (LLM) | Random Forest (ML) |
|---|---|---|
| **Qué hace** | Entiende lenguaje humano | Estima un precio |
| **Entrada** | Texto libre | 3 datos estructurados |
| **Salida** | JSON con 3 campos | Un número en dólares |
| **Aprendió de** | Texto general de internet | 1.460 ventas reales de Ames |
| **Puede inventar** | Sí, por eso se valida | No, solo promedia datos reales |

Un LLM predice la palabra siguiente más probable; no sabe cuánto cuesta una casa y,
si se le pregunta, se inventaría una cifra plausible pero sin respaldo. El Random
Forest sí puede estimar el precio, pero no entiende español. Cada uno hace lo que
sabe hacer.

---

## Dataset

**Ames Housing / house_prices**, descargado desde OpenML (`data_id=42165`).
1.460 viviendas reales de Ames, Iowa, con 81 variables.

De esas 81 se usan 3 como entrada, porque son las únicas que un usuario puede
responder cómodamente en un mensaje:

| Dato que pide el bot | Variable original | Variable del modelo | Transformación |
|---|---|---|---|
| Sector | `Neighborhood` | `Neighborhood` | OneHotEncoder |
| Metros cuadrados | `GrLivArea` (pies²) | `Area_m2` | `× 0.092903` |
| Años de antigüedad | `YearBuilt`, `YrSold` | `Antiguedad` | `YrSold - YearBuilt` |
| **Precio (objetivo)** | `SalePrice` | `SalePrice` | — |

---

## Resultados del modelo

Evaluado sobre 292 viviendas que el modelo nunca vio durante el entrenamiento:

| Métrica | Valor | Qué significa |
|---|---|---|
| **MAE** | $24.043,85 | En promedio se equivoca en esa cantidad |
| **RMSE** | $36.414,53 | Igual que el MAE, pero penaliza más los errores grandes |
| **R²** | **0,827** | Explica el 82,7 % de la variación de los precios |

Configuración: `RandomForestRegressor(n_estimators=80, max_depth=10,
min_samples_leaf=2, random_state=42)`, división 80 / 20, 1.168 casas de
entrenamiento y 292 de prueba.

Un R² de 0,83 usando solo 3 de las 81 variables disponibles indica que el sector, el
tamaño y la antigüedad concentran la mayor parte de la información sobre el precio.
El 17 % restante depende de factores que el modelo no conoce: número de baños,
garaje, calidad de los acabados o estado de la cocina.

---

## Estructura del repositorio

```
bot_ia_inmobiliaria/
├── 01_entrenamiento_random_forest_colab.ipynb   Entrenamiento en Google Colab
├── bot_telegram_ollama.py                       Bot: Telegram + Mistral + modelo
├── modelo_casas.pkl                             Pipeline entrenado
├── sectores_ames.json                           Catálogo de los 25 sectores
├── metricas_modelo.json                         MAE, RMSE y R² obtenidos
├── requirements_local.txt                       Versiones exactas de librerías
├── .gitignore                                   Protege el token y el entorno
└── README.md
```

---

## Requisitos

- **Python 3.12** (el proyecto se probó con esta versión)
- **Ollama** instalado y en ejecución — https://ollama.com
- El modelo **Mistral** descargado
- Una cuenta de Telegram y un bot creado con [@BotFather](https://t.me/BotFather)

---

## Instalación

**1. Clonar el repositorio**

```bash
git clone <URL-DE-TU-REPOSITORIO>
cd bot_ia_inmobiliaria
```

**2. Crear el entorno virtual e instalar las dependencias**

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements_local.txt
```

`requirements_local.txt` fija las versiones exactas con las que se entrenó el modelo
en Colab (scikit-learn 1.6.1 y pandas 2.2.3). Es necesario respetarlas: si se carga
el `.pkl` con otra versión de scikit-learn, pueden aparecer advertencias o errores
de compatibilidad.

**3. Descargar el modelo de lenguaje**

```bash
ollama pull mistral
```

**4. Crear el bot de Telegram**

Abrir [@BotFather](https://t.me/BotFather), enviar `/newbot`, elegir un nombre y un
username, y guardar el token que devuelve.

> El token **no se escribe en el código ni se sube al repositorio**. El programa lo
> pide por teclado al iniciar, usando `getpass`, de modo que tampoco queda visible
> en pantalla ni en el historial de la terminal.

---

## Ejecución

Con Ollama abierto:

```bash
.venv\Scripts\python.exe bot_telegram_ollama.py
```

El programa realiza cuatro comprobaciones antes de empezar a escuchar:

```
[1/4] Cargando el modelo Random Forest...
      Modelo cargado. 25 sectores disponibles.
[2/4] Verificando Ollama y el modelo 'mistral'...
      Ollama responde correctamente.
[3/4] Token del bot de Telegram
      Pega tu token y presiona Enter: (oculto)
      Token valido. Bot conectado: @tu_bot
[4/4] Bot escuchando. Escribele desde Telegram.
```

Para detenerlo: `Ctrl + C`.

---

## Ejemplos de mensajes válidos

El bot acepta lenguaje natural, en cualquier orden y con distintas formas de nombrar
las cosas:

| Mensaje | Respuesta |
|---|---|
| Tengo una casa en Old Town de 120 metros cuadrados y 25 años. | `$171,393` |
| ¿Cuánto vale una vivienda de 95 m2, 10 años, ubicada en North Ames? | `$148,138` |
| Casa en Gilbert, 150 metros cuadrados, antigüedad 8 años. | `$184,134` |
| Quiero tasar mi propiedad de Northridge Heights, 250 metros, 12 años | `$400,422` |
| En Sawyer West. 140 m2. 20 años. | `$193,276` |

Cuando falta información, el bot **no inventa datos**: indica qué falta.

| Mensaje | Respuesta |
|---|---|
| Tengo una casa en Old Town | Me faltan datos para calcular el precio: los metros cuadrados, los años de antigüedad. |
| ¿Cuánto cuesta una casa en Quito de 120 metros y 10 años? | Me faltan datos: el sector. Solo trabajo con sectores de Ames, Iowa. |

Los 25 sectores válidos están en `sectores_ames.json`. El bot reconoce tanto el
código (`OldTown`) como el nombre completo (`Old Town`), sin importar mayúsculas,
tildes ni espacios.

---

## Decisiones técnicas

**Se usa un `Pipeline` de scikit-learn.** El archivo `.pkl` contiene el
OneHotEncoder y el bosque juntos, así que el bot puede pasarle el sector como texto
sin tener que reconstruir el preprocesamiento a mano. Esto elimina la fuente más
común de errores al pasar un modelo de Colab a producción.

**El catálogo de sectores es un diccionario, no una lista.** El formato
`{"OldTown": "Old Town"}` permite incluir en el prompt de Mistral la equivalencia
entre lo que escribe el usuario y el código con el que se entrenó el modelo.

**El prompt incluye ejemplos resueltos (*few-shot*).** Con instrucciones sin
ejemplos, el modelo de lenguaje fallaba al extraer los metros cuadrados: convertía
"95 m2" en 9.5 y devolvía `null` para "150 metros cuadrados". Añadiendo cinco
ejemplos resueltos, la extracción pasó a ser correcta en todas las pruebas y el
tiempo de respuesta bajó de 19 a 2,7 segundos.

**La validación vive en Python, no en el prompt.** Un modelo de lenguaje puede
ignorar instrucciones; una condición en Python no. Por eso el rango de metros, el
rango de años y la existencia del sector se comprueban en código.

---

## Advertencia académica

El resultado es una **estimación educativa** basada en datos históricos de ventas de
Ames, Iowa (Estados Unidos). No constituye una tasación profesional ni refleja los
precios actuales del mercado inmobiliario ecuatoriano.

---

## Fuentes

- [OpenML — Ames Housing / house_prices](https://www.openml.org/d/42165)
- [scikit-learn — RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Ollama — API chat](https://docs.ollama.com/api/chat)
- [Telegram Bot API](https://core.telegram.org/bots/api)
