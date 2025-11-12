# Chat de Asistente de Documentos PDF

Aplicación web de chat que permite hacer preguntas sobre documentos PDF usando RAG (Retrieval Augmented Generation) con Ollama.

## Requisitos Previos

1. **Python 3.11+**
2. **Ollama** instalado y corriendo con el modelo `mistral`
3. Archivo `fuente.pdf` en el directorio raíz

## Instalación

Las dependencias ya están instaladas, solo asegúrate de tener Flask:

```bash
pip install flask
```

## Ejecutar la Aplicación

1. Asegúrate de que Ollama esté corriendo:
   ```bash
   ollama serve
   ```

2. Verifica que tengas el modelo mistral:
   ```bash
   ollama list
   ```

3. Ejecuta el servidor:
   ```bash
   python app.py
   ```

4. Abre tu navegador en:
   ```
   http://localhost:5000
   ```

## Estructura del Proyecto

```
.
├── app.py                 # Servidor Flask
├── test.py               # Script CLI original
├── fuente.pdf            # Documento PDF a consultar
├── templates/
│   └── index.html        # Interfaz del chat
├── static/
│   └── style.css         # Estilos del chat
└── README.md             # Este archivo
```

## Características

- ✅ Interfaz de chat moderna y responsiva
- ✅ Respuestas inteligentes usando Ollama (Mistral)
- ✅ Sistema RAG con búsqueda por similitud
- ✅ Filtrado de relevancia para evitar respuestas incorrectas
- ✅ Sin necesidad de login o autenticación
- ✅ Indicador de "escribiendo..." mientras procesa

## Uso

1. Escribe tu pregunta en el campo de texto
2. Presiona Enter o haz clic en el botón de enviar
3. Espera la respuesta del asistente
4. Continúa la conversación

## Notas

- El sistema tarda unos segundos en inicializarse al arrancar
- Las respuestas solo se basan en el contenido del PDF cargado
- Si una pregunta no está relacionada con el documento, el sistema lo indicará

