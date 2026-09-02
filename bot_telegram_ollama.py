"""Bot de Telegram que estima el precio de una vivienda de Ames, Iowa.

Arquitectura del sistema (parte 2 de 2):

    Telegram  ->  Ollama + Mistral  ->  Random Forest  ->  Telegram
    (mensaje)     (extrae 3 datos)     (calcula precio)     (responde)

Reparto de responsabilidades, que es lo importante del proyecto:

    Mistral   entiende el lenguaje natural, pero NO calcula precios.
              Su unica tarea es convertir "una casa en Old Town de 120 m2
              y 25 anios" en {"sector": "OldTown", "metros": 120, "anios": 25}.

    Python    valida esos datos. Si Mistral inventa o se deja algo, aqui se corta.

    Random    es el unico que produce un numero en dolares, porque es el unico
    Forest    que fue entrenado con precios de venta reales.

Ejecutar con:
    python bot_telegram_ollama.py

El token de Telegram se pide por teclado al iniciar (getpass), de modo que
nunca queda escrito en el codigo ni se sube al repositorio.
"""

import getpass
import json
import time
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import requests

# ------------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------------
# Estas son las unicas lineas que normalmente necesitas tocar.

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODELO = "mistral"          # plan B si mistral va lento: "qwen2.5:3b"
TIEMPO_ESPERA_OLLAMA = 120         # segundos que damos a Mistral para responder

CARPETA = Path(__file__).parent
RUTA_MODELO = CARPETA / "modelo_casas.pkl"
RUTA_SECTORES = CARPETA / "sectores_ames.json"


# ------------------------------------------------------------------
# 1. CARGA DE RECURSOS
# ------------------------------------------------------------------

def cargar_recursos():
    """Carga el modelo entrenado en Colab y el catalogo de sectores.

    Devuelve:
        (modelo, sectores) donde sectores es un diccionario
        {"OldTown": "Old Town", "NAmes": "North Ames", ...}
    """
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No encuentro {RUTA_MODELO.name}. Ejecuta el notebook de Colab "
            f"y coloca el archivo en:\n  {CARPETA}"
        )
    if not RUTA_SECTORES.exists():
        raise FileNotFoundError(
            f"No encuentro {RUTA_SECTORES.name}. Ejecuta el notebook de Colab "
            f"y coloca el archivo en:\n  {CARPETA}"
        )

    modelo = joblib.load(RUTA_MODELO)

    with RUTA_SECTORES.open("r", encoding="utf-8") as archivo:
        sectores = json.load(archivo)

    # El notebook guarda un diccionario {codigo: nombre legible}.
    # Si por lo que sea llegara una lista, la convertimos para no romper nada.
    if isinstance(sectores, list):
        sectores = {codigo: codigo for codigo in sectores}

    return modelo, sectores


def verificar_ollama():
    """Comprueba que Ollama este encendido y que el modelo este descargado."""
    try:
        respuesta = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        respuesta.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            "No puedo conectarme con Ollama.\n"
            "Abre la aplicacion Ollama y vuelve a intentarlo."
        ) from error

    modelos_instalados = [
        modelo["name"] for modelo in respuesta.json().get("models", [])
    ]

    # Los nombres vienen como "mistral:latest", por eso comparamos el prefijo.
    disponible = any(
        nombre.split(":")[0] == OLLAMA_MODELO.split(":")[0]
        for nombre in modelos_instalados
    )

    if not disponible:
        raise RuntimeError(
            f"Ollama funciona, pero no tienes el modelo '{OLLAMA_MODELO}'.\n"
            f"Ejecuta en PowerShell:  ollama pull {OLLAMA_MODELO}\n"
            f"Modelos que si tienes: {', '.join(modelos_instalados) or 'ninguno'}"
        )


# ------------------------------------------------------------------
# 2. EXTRACCION CON MISTRAL (el LLM solo entiende, no calcula)
# ------------------------------------------------------------------

def construir_prompt(mensaje, sectores):
    """Arma las instrucciones que recibira Mistral.

    Le pasamos el catalogo completo de sectores para que traduzca lo que
    escribe el usuario ("Old Town") al codigo del dataset ("OldTown").
    """
    catalogo = "\n".join(
        f"- {codigo}  = {nombre}" for codigo, nombre in sectores.items()
    )

    return f"""Eres un extractor de datos. NO calculas precios ni das opiniones.

Tu unica tarea es leer el mensaje de un usuario sobre una vivienda de Ames, Iowa,
y copiar tres datos a un JSON:

1. "sector": el codigo del barrio, tomado de la lista de abajo.
2. "metros": el numero de metros cuadrados que aparece en el mensaje.
3. "anios": el numero de anios de antiguedad que aparece en el mensaje.

Codigos de sector permitidos (codigo = nombre que puede escribir el usuario):
{catalogo}

Reglas estrictas:
- COPIA los numeros tal como aparecen en el mensaje. No los conviertas,
  no los redondees, no los dividas y no los multipliques.
- "m2", "metros", "metros cuadrados" y "mts" significan todos lo mismo.
- Devuelve el CODIGO del sector, nunca el nombre largo.
- Si un dato NO aparece en el mensaje, pon null. Nunca lo inventes.
- Si el barrio mencionado no esta en la lista, pon null SOLO en "sector",
  pero extrae igual los metros y los anios si el mensaje los menciona.
- Responde UNICAMENTE con el JSON, sin explicaciones ni texto adicional.

Ejemplos resueltos:

Mensaje: "Tengo una casa en Old Town de 120 metros cuadrados y 25 anios."
Respuesta: {{"sector": "OldTown", "metros": 120, "anios": 25}}

Mensaje: "Cuanto vale una vivienda de 95 m2, 10 anios, ubicada en North Ames?"
Respuesta: {{"sector": "NAmes", "metros": 95, "anios": 10}}

Mensaje: "Casa en Gilbert, 150 metros cuadrados, antiguedad 8 anios."
Respuesta: {{"sector": "Gilbert", "metros": 150, "anios": 8}}

Mensaje: "Tengo un departamento en Somerset"
Respuesta: {{"sector": "Somerst", "metros": null, "anios": null}}

Mensaje: "Hola, buenas tardes"
Respuesta: {{"sector": null, "metros": null, "anios": null}}

Ahora resuelve este:

Mensaje: "{mensaje}"
Respuesta:"""


def extraer_datos_con_mistral(mensaje, sectores):
    """Convierte un mensaje libre en datos estructurados usando Mistral local.

    Devuelve un diccionario con las claves sector, metros y anios.
    Cualquiera de ellas puede venir como None si el usuario no la escribio.
    """
    peticion = {
        "model": OLLAMA_MODELO,
        "messages": [{"role": "user", "content": construir_prompt(mensaje, sectores)}],
        "format": "json",          # obliga a Ollama a devolver JSON valido
        "stream": False,
        "options": {"temperature": 0},  # 0 = respuestas estables, sin creatividad
    }

    try:
        respuesta = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=peticion,
            timeout=TIEMPO_ESPERA_OLLAMA,
        )
        respuesta.raise_for_status()
        contenido = respuesta.json()["message"]["content"]
    except requests.RequestException as error:
        raise RuntimeError("Ollama no respondio a tiempo.") from error
    except KeyError as error:
        raise RuntimeError("Ollama devolvio una respuesta inesperada.") from error

    try:
        return json.loads(contenido)
    except json.JSONDecodeError:
        # Red de seguridad: si el modelo acompanio el JSON con texto suelto,
        # recortamos desde la primera llave hasta la ultima.
        inicio = contenido.find("{")
        final = contenido.rfind("}")
        if inicio == -1 or final == -1:
            raise RuntimeError("Mistral no devolvio un JSON interpretable.")
        return json.loads(contenido[inicio:final + 1])


# ------------------------------------------------------------------
# 3. VALIDACION EN PYTHON (aqui se corta lo que Mistral haga mal)
# ------------------------------------------------------------------

def simplificar(texto):
    """Deja el texto en minusculas, sin tildes ni espacios, para poder comparar."""
    sin_tildes = unicodedata.normalize("NFKD", str(texto))
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return "".join(c for c in sin_tildes.lower() if c.isalnum())


def normalizar_sector(valor, sectores):
    """Devuelve el codigo oficial del sector, o None si no lo reconocemos.

    Acepta tanto el codigo ("OldTown") como el nombre largo ("Old Town"),
    ignorando mayusculas, tildes y espacios.
    """
    if not valor:
        return None

    buscado = simplificar(valor)

    for codigo, nombre in sectores.items():
        if buscado in (simplificar(codigo), simplificar(nombre)):
            return codigo

    return None


def a_numero(valor):
    """Convierte a numero decimal. Devuelve None si no es convertible."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def validar_datos(datos, sectores):
    """Revisa que los tres datos existan y sean coherentes.

    Devuelve:
        (fila, faltantes)
        fila      -> diccionario listo para el modelo, o None si algo falla.
        faltantes -> lista con los nombres de los datos que hay que pedir.
    """
    faltantes = []

    sector = normalizar_sector(datos.get("sector"), sectores)
    if sector is None:
        faltantes.append("el sector")

    metros = a_numero(datos.get("metros"))
    if metros is None or not 10 <= metros <= 1000:
        faltantes.append("los metros cuadrados")
        metros = None

    anios = a_numero(datos.get("anios"))
    if anios is None or not 0 <= anios <= 200:
        faltantes.append("los anios de antiguedad")
        anios = None

    if faltantes:
        return None, faltantes

    # Los nombres de las claves deben coincidir EXACTAMENTE con las columnas
    # con las que se entreno el Pipeline en Colab.
    fila = {
        "Neighborhood": sector,
        "Area_m2": metros,
        "Antiguedad": anios,
    }
    return fila, []


# ------------------------------------------------------------------
# 4. PREDICCION CON RANDOM FOREST (el unico que produce el precio)
# ------------------------------------------------------------------

def predecir_precio(modelo, fila):
    """Pasa los datos validados al Pipeline entrenado y devuelve el precio."""
    entrada = pd.DataFrame([fila])
    return float(modelo.predict(entrada)[0])


# ------------------------------------------------------------------
# 5. COMUNICACION CON TELEGRAM
# ------------------------------------------------------------------

def enviar_mensaje(token, chat_id, texto):
    """Envia un mensaje de texto al chat del usuario (metodo sendMessage)."""
    respuesta = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": texto},
        timeout=30,
    )
    respuesta.raise_for_status()


def obtener_actualizaciones(token, offset=None):
    """Recibe los mensajes nuevos mediante long polling (metodo getUpdates).

    'offset' le dice a Telegram cual fue el ultimo mensaje que ya procesamos,
    para que no nos lo vuelva a enviar.
    """
    parametros = {"timeout": 30, "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        parametros["offset"] = offset

    respuesta = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        params=parametros,
        timeout=40,
    )
    respuesta.raise_for_status()

    datos = respuesta.json()
    if not datos.get("ok"):
        raise RuntimeError(datos.get("description", "Telegram devolvio un error."))

    return datos.get("result", [])


def comprobar_token(token):
    """Verifica el token contra Telegram y devuelve el nombre del bot."""
    respuesta = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20)
    if respuesta.status_code == 401:
        raise ValueError("Telegram rechazo el token. Revisa que lo copiaste completo.")
    respuesta.raise_for_status()
    return respuesta.json()["result"]["username"]


# ------------------------------------------------------------------
# 6. FLUJO COMPLETO DE UN MENSAJE
# ------------------------------------------------------------------

MENSAJE_BIENVENIDA = (
    "Estimador de precios de viviendas de Ames, Iowa.\n\n"
    "Escribeme el sector, los metros cuadrados y los anios de antiguedad.\n"
    "Ejemplo: Tengo una casa en Old Town de 120 metros cuadrados y 25 anios."
)


def procesar_mensaje(modelo, sectores, texto):
    """Recorre el flujo completo: texto -> Mistral -> validacion -> Random Forest."""
    if texto.strip().lower() in ("/start", "/help", "/ayuda"):
        return MENSAJE_BIENVENIDA

    datos = extraer_datos_con_mistral(texto, sectores)
    print(f"   Mistral extrajo: {datos}")

    fila, faltantes = validar_datos(datos, sectores)

    if fila is None:
        aviso = (
            "Me faltan datos para calcular el precio: "
            + ", ".join(faltantes)
            + ".\nEscribelos en un mensaje y te doy la estimacion."
        )
        # Si el problema es el sector, mostramos algunos validos como ayuda.
        if "el sector" in faltantes:
            ejemplos = ", ".join(list(sectores.values())[:5])
            aviso += (
                "\n\nSolo trabajo con sectores de Ames, Iowa. "
                f"Por ejemplo: {ejemplos}."
            )
        return aviso

    precio = predecir_precio(modelo, fila)
    print(f"   Random Forest predijo: ${precio:,.0f}")

    # La guia pide que, cuando los datos esten completos,
    # el bot responda unicamente con el precio.
    return f"${precio:,.0f}"


# ------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  BOT ESTIMADOR DE PRECIOS - Ames, Iowa")
    print("  Telegram -> Mistral -> Random Forest -> Telegram")
    print("=" * 60)

    print("\n[1/4] Cargando el modelo Random Forest...")
    modelo, sectores = cargar_recursos()
    print(f"      Modelo cargado. {len(sectores)} sectores disponibles.")

    print(f"\n[2/4] Verificando Ollama y el modelo '{OLLAMA_MODELO}'...")
    verificar_ollama()
    print("      Ollama responde correctamente.")

    print("\n[3/4] Token del bot de Telegram")
    print("      (no se muestra al escribir, y no se guarda en ningun archivo)")
    token = getpass.getpass("      Pega tu token y presiona Enter: ").strip()
    if not token:
        raise ValueError("El token no puede estar vacio.")

    nombre_bot = comprobar_token(token)
    print(f"      Token valido. Bot conectado: @{nombre_bot}")

    print("\n[4/4] Bot escuchando. Escribele desde Telegram.")
    print("      Para detenerlo: Ctrl + C\n")

    # Descartamos los mensajes que llegaron antes de arrancar,
    # para que el bot no conteste conversaciones viejas al iniciar.
    pendientes = obtener_actualizaciones(token)
    offset = pendientes[-1]["update_id"] + 1 if pendientes else None

    try:
        while True:
            try:
                for actualizacion in obtener_actualizaciones(token, offset):
                    offset = actualizacion["update_id"] + 1

                    mensaje = actualizacion.get("message")
                    if not mensaje or "text" not in mensaje:
                        continue

                    chat_id = mensaje["chat"]["id"]
                    texto = mensaje["text"]
                    print(f"-> Mensaje recibido: {texto}")

                    try:
                        respuesta = procesar_mensaje(modelo, sectores, texto)
                    except RuntimeError as error:
                        print(f"   Error: {error}")
                        respuesta = (
                            "No pude interpretar tu mensaje. "
                            "Intenta escribir el sector, los metros y los anios."
                        )

                    enviar_mensaje(token, chat_id, respuesta)
                    print(f"<- Respuesta enviada: {respuesta}\n")

            except requests.RequestException as error:
                print(f"Problema de conexion: {error}. Reintentando en 3 segundos...")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\nBot detenido correctamente.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\nNo se pudo iniciar el bot:\n{error}\n")
