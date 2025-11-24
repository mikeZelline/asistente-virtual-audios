import os
import requests
import time
import re
import json
import logging
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde archivo .env (si existe)
load_dotenv('config.env', override=True)

# Leer directamente la API key del archivo config.env para asegurarnos de usar la correcta
def read_api_key_from_config():
    """Lee la API key directamente del archivo config.env"""
    try:
        with open('config.env', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Buscar la línea que contiene OPENAI_API_KEY= y que no sea un comentario
                if line.startswith('OPENAI_API_KEY=') and not line.startswith('#'):
                    # Extraer el valor después del =
                    key_value = line.split('=', 1)[1].strip()
                    # Remover comillas si las tiene
                    key_value = key_value.strip('"').strip("'")
                    return key_value
    except Exception as e:
        logger.warning(f"Error al leer config.env directamente: {e}")
    return None

# Leer API key directamente del archivo
OPENAI_API_KEY_FROM_FILE = read_api_key_from_config()

# Configura variables de entorno ANTES de los imports
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "mi-usuario-personalizado/0.0.1")

# Usar la key del archivo si se encontró, sino usar la de entorno (fallback)
if OPENAI_API_KEY_FROM_FILE:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_FROM_FILE
    OPENAI_API_KEY = OPENAI_API_KEY_FROM_FILE
    logger.info(f"✅ API_KEY cargada desde config.env (termina en: ...{OPENAI_API_KEY[-4:]})")
    print(f"✅ API_KEY cargada desde config.env (termina en: ...{OPENAI_API_KEY[-4:]})")
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    logger.warning(f"⚠️ API_KEY no encontrada en config.env, usando variable de entorno")
    print(f"⚠️ API_KEY no encontrada en config.env, usando variable de entorno")

os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY", "")
os.environ["ELEVENLABS_VOICE_ID"] = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Clave única para cada ejecución
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutos

# Token de sesión único por ejecución del servidor
SESSION_TOKEN = os.urandom(16).hex()

# Credenciales hardcoded
USERS = {
    'test': 'test123*'
}

# Variables globales para el sistema RAG
vector_store = None
llm = None
graph = None
def build_base_prompt():
    """Construye el prompt base con el objetivo del chat."""
    return """OBJETIVO DEL CHAT:
Actúa como un ejecutivo de ventas de alto rendimiento para IBUX y entrena mis habilidades en ventas B2B complejas en tecnología (soluciones TIC, automatización, nube, ciberseguridad, datos, inteligencia artificial y consultoría). 
Basa tus respuestas en la metodología 3C de IBUX (Crear conexión, Comprender profundamente las necesidades y Comunicar soluciones alineadas al contexto) y en el proceso de ventas de 7 etapas: 
 - prospección
 - primer contacto 
 - diagnóstico o descubrimiento 
 - diseño y elaboración de la propuesta
 - presentación de la propuesta
 - seguimiento
 - cierre y postventa. 
Ten en cuenta mis metas SMART actuales: 
 - subir mi tasa de cierre del 10% al 30%
 - generar 10 leads calificados por semana
 - alcanzar ₡3,000 millones en ventas anuales y abrir mercado en soluciones como Sophos XDR, AppGate SDP, AWS, automatización y datos. 
Considera también el perfil del cliente ideal de IBUX: instituciones públicas y municipalidades en Costa Rica que valoran precio, efectividad, relación y asesoría, pero que lidian con burocracia, falta de presupuesto y presiones normativas. 
Integra técnicas, modelos y autores de referencia como Grant Cardone, Dale Carnegie, Brian Tracy, Jordan Belfort, Neil Rackham, Daniel Pink, Chet Holmes, James Clear, Carol Dweck, así como metodologías como SPIN, BANT, MEDDIC, Challenger Sale, Consultative Selling, Gap Selling, AIDA, PAS, FAB, Hook-Story-Offer, Storytelling Selling y Value-Based Selling.

Cuando escriba "tutorial", debes mostrar inmediatamente la guía completa del proceso de ventas con ejemplos aplicados a cada etapa. Mantén un estilo claro, directo, práctico y enfocado a resultados. 
Provee frases poderosas, guiones, ejercicios, cierres, seguimientos, diagnósticos, rutinas mentales y retroalimentación según la etapa. 
Simula objeciones realistas y entrena usando escenarios del mundo real. Tu objetivo es convertirme en un vendedor consultivo, persuasivo, estratégico y centrado en el cliente."""

# Estado (siguiendo patrón de LangChain)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[dict]  # Historial de conversación

def normalize_url(url):
    """Normaliza una URL eliminando fragmentos, parámetros de tracking y normalizando el path."""
    try:
        parsed = urlparse(url)
        # Normalizar path: remover trailing slash y duplicados
        path = parsed.path.rstrip('/') or '/'
        # Remover parámetros de tracking comunes
        query_params = []
        if parsed.query:
            for param in parsed.query.split('&'):
                if param and not param.startswith(('utm_', 'ref=', 'source=', 'fbclid=', 'gclid=')):
                    query_params.append(param)
        query = '&'.join(query_params) if query_params else ''
        # Construir URL normalizada sin fragmento
        normalized = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, ''))
        return normalized
    except Exception:
        return url

# Función para recuperar documentos relevantes
def retrieve(state: State):
    """Recupera los documentos más relevantes para la pregunta."""
    if not vector_store:
        return {"context": []}
    
    # Aumentar k para recuperar más documentos y tener más opciones
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=15)
    
    # Mostrar pregunta y relevancia
    best_score = retrieved_docs_with_scores[0][1] if retrieved_docs_with_scores else None
    print(f"\nPregunta: '{state['question']}'")
    
    # Umbrales mejorados
    # Nota: En FAISS, scores más bajos = más similares
    SIMILARITY_THRESHOLD = 0.85
    MIN_SCORE_THRESHOLD = 1.5  # Score mínimo absoluto para descartar resultados irrelevantes
    BEST_DOC_THRESHOLD = 1.2
    TOP_N_THRESHOLD = 1.0
    TOP_N_COUNT = 3
    
    relevant_docs = []
    seen_content = set()  # Para evitar duplicados
    
    best_doc_accepted = False
    top_n_accepted = 0
    
    for idx, (doc, score) in enumerate(retrieved_docs_with_scores):
        # Filtrar por score mínimo absoluto
        if score > MIN_SCORE_THRESHOLD and idx > 0:
            continue
        
        # Evitar duplicados basados en contenido
        content_hash = hash(doc.page_content[:200])  # Hash de primeros 200 chars
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        # Aceptar si está por debajo del threshold normal
        if score < SIMILARITY_THRESHOLD:
            relevant_docs.append(doc)
            if idx == 0:
                best_doc_accepted = True
            if idx < TOP_N_COUNT:
                top_n_accepted += 1
        # Si es uno de los mejores documentos y tiene un score razonable, aceptarlo
        elif idx < TOP_N_COUNT and score < TOP_N_THRESHOLD and top_n_accepted < TOP_N_COUNT:
            relevant_docs.append(doc)
            if idx == 0:
                best_doc_accepted = True
            top_n_accepted += 1
        # Si es el mejor documento y no se ha aceptado nada, aceptarlo con umbral más alto
        elif idx == 0 and score < BEST_DOC_THRESHOLD and not best_doc_accepted and len(relevant_docs) == 0:
            relevant_docs.append(doc)
            best_doc_accepted = True
    
    # Mostrar relevancia
    if relevant_docs:
        relevancia_score = round(best_score, 3)
        print(f"Relevancia encontrada (índice de coincidencia): {relevancia_score}")
    else:
        print(f"Sin relevancia (mejor índice encontrado: {round(best_score, 3) if best_score else 'N/A'})")
    print()
    
    return {"context": relevant_docs}

def validate_relevance(question, context_docs, history):
    """
    Usa el LLM para determinar si la pregunta es relevante y generar la respuesta.
    
    Returns:
        tuple: (answer: str, is_relevant: bool, reason: str, can_use_history: bool, should_reject: bool)
    """
    # Construir contexto disponible
    has_rag_context = len(context_docs) > 0
    has_history = history and len(history) > 0
    
    # Debug: Mostrar pregunta que se está evaluando
    print(f"\n{'='*80}")
    print(f"🔍 [DEBUG VALIDACIÓN] Evaluando pregunta:")
    print(f"  Pregunta: '{question}'")
    if has_history:
        print(f"  Historial disponible: {len(history)} mensajes")
        if history:
            last_user_msg = next((msg for msg in reversed(history) if msg.get('role') == 'user'), None)
            if last_user_msg:
                print(f"  Último mensaje del usuario: {last_user_msg.get('content', '')[:100]}")
    print(f"  Contexto RAG: {'Sí' if has_rag_context else 'No'} ({len(context_docs)} documentos)")
    print(f"{'='*80}\n")
    
    # Construir historial completo para el prompt
    history_text = ""
    if has_history:
        history_messages = history[-10:]  # Últimos 10 mensajes
        if history_messages:
            history_text = "\n\n═══════════════════════════════════════════════════════\n"
            history_text += "HISTORIAL DE CONVERSACIÓN ANTERIOR (MUY IMPORTANTE - USA ESTO PARA ENTENDER EL CONTEXTO):\n"
            history_text += "═══════════════════════════════════════════════════════\n"
            for i, msg in enumerate(history_messages, 1):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_text += f"[{i}] Usuario: {content}\n"
                elif role == "assistant":
                    history_text += f"[{i}] Asistente: {content}\n"
            history_text += "═══════════════════════════════════════════════════════\n"
            history_text += "\nIMPORTANTE: Si la pregunta actual usa palabras como 'eso', 'ese', 'esa', 'ellos', 'eso que', 'lo que mencionaste', etc., se refiere a algo del historial arriba. DEBES usar ese contexto para entender la pregunta.\n"
    
    # Construir información del contexto RAG (si existe)
    context_rag = ""
    if has_rag_context:
        logger.info(f"📄 [GENERATE] Generando respuesta con contexto RAG: {len(context_docs)} documentos")
        docs_content = "\n\n".join(doc.page_content for doc in context_docs if doc.page_content and len(doc.page_content.strip()) > 50)
        context_limit = 2000
        context_rag = docs_content[:context_limit].strip()
        
        # Debug: Mostrar texto exacto extraído del RAG
        print(f"\n{'='*80}")
        print("🔍 [DEBUG RAG] TEXTO EXACTO EXTRAÍDO DEL RAG:")
        print(f"{'='*80}")
        print(context_rag)
        print(f"{'='*80}\n")
    else:
        logger.info(f"🔍 [GENERATE] Sin contexto RAG - Generando respuesta sin documentos")
    
    # Construir sección de contexto RAG
    context_section = ""
    if context_rag:
        context_section = f"""
INFORMACIÓN RELEVANTE DEL CONTENIDO:
{context_rag}
"""
    else:
        context_section = """
NOTA: No tengo información específica en mis documentos sobre este tema.
"""
    
    base_prompt = build_base_prompt()
    validation_prompt = f"""Eres un asistente mentor comercial especializado en ventas B2B de tecnología para IBUX.

{base_prompt}

{history_text}
{context_section}
PREGUNTA ACTUAL DEL USUARIO: {question}

INSTRUCCIONES:
1. REGLA SUPERIOR - Interpretación semántica amplia del dominio (APLICAR PRIMERO Y SIEMPRE):
   El asistente debe asumir que una pregunta es RELEVANTE siempre que pueda interpretarse razonablemente como relacionada con:
   - ventas (cualquier tipo de ventas: B2B, B2C, productos físicos, servicios, tecnología, etc.)
   - habilidades comerciales
   - técnicas de persuasión y cierre
   - procesos comerciales y estrategias de venta
   - mejora profesional y aumento de ganancias/ingresos
   - tecnología asociada a soluciones vendidas por IBUX
   - el rol del mentor (qué sabe, qué enseña, cómo puede ayudar)
   
   Esta regla aplica incluso si la pregunta:
   - es muy corta ("¿Y sobre ventas?", "¿Qué sabes hacer?", "¿Y eso qué es?")
   - es ambigua, pero puede entenderse dentro del contexto del asistente
   - no menciona explícitamente "B2B", "IBUX", "3C", ni pasos del proceso
   - es una derivación natural de preguntas ya respondidas
   - trata sobre ventas de productos o servicios que NO son tecnología
   
   Principio clave:
   - Si la pregunta puede interpretarse de forma razonable dentro del ámbito de ventas, habilidades comerciales, mejora de ganancias o tecnología, el asistente SIEMPRE debe considerarla RELEVANTE.
   - Solo se debe marcar como no relevante si el tema NO tiene NINGUNA posible relación con ventas, tecnología, el rol del mentor o el objetivo del chat.
   - ESTA ES LA REGLA MÁS IMPORTANTE: Cuando tengas dudas, SIEMPRE marca como RELEVANTE si hay alguna conexión con ventas, habilidades comerciales o mejora profesional.

2. Analiza la pregunta en el contexto del objetivo del chat y el historial disponible.

3. Una pregunta ES RELEVANTE si:
   - Trata directamente sobre ventas, ventas B2B, técnicas comerciales, metodología 3C, soluciones de IBUX, o temas relacionados
   - Hace referencia a algo mencionado en el historial anterior (por ejemplo, si antes preguntaron "¿Qué es IBUX?" y ahora preguntan "¿Pero eso es una empresa, un servicio o un método?", entonces "eso" se refiere a IBUX y es relevante)
   - Tiene información relevante en el contexto RAG (SI HAY CONTEXTO RAG, LA PREGUNTA ES AUTOMÁTICAMENTE RELEVANTE - esto se verifica antes de esta evaluación)
   - Es una continuación, aclaración o seguimiento de algo del historial que SÍ era relevante
   - Es una pregunta meta sobre cómo usar el asistente o qué puede preguntar
   - Trata sobre ventas en general, mejora de ganancias, técnicas comerciales (aplicar regla 1 primero)
   - Trata sobre herramientas tecnológicas, software, plataformas o tecnologías en el contexto de ventas, mejora profesional o aumento de ganancias

4. Una pregunta NO ES RELEVANTE (should_reject: true) SOLO si:
   - Trata sobre temas completamente ajenos al objetivo del chat (entretenimiento, deportes, cocina, política no relacionada, etc.)
   - Es irónica o sarcástica sin relación alguna con ventas, tecnología o el rol del mentor
   - Se refiere a actividades ilegales o no éticas
   - NO tiene NINGUNA posible relación con ventas, tecnología, el rol del mentor o el objetivo del chat (después de aplicar la regla 1)

5. Determina si el historial puede ayudar a entender una pregunta ambigua (can_use_history):
   - Si la pregunta usa referencias implícitas (eso, ese, esa, ellos, "eso que", "lo que dijiste", etc.) Y el historial contiene contexto relevante sobre el tema, entonces can_use_history = true
   - Si la pregunta es ambigua o corta pero el historial provee contexto relevante, entonces can_use_history = true
   - Ejemplo: Si el historial mencionó "IBUX" y ahora preguntan "¿Pero eso es una empresa?", can_use_history = true porque "eso" se refiere a IBUX del historial

6. Excepción especial para saludos:
   - Si la pregunta es un saludo o frase social básica (como "hola", "buenas", "cómo estás", "qué tal", "hey", "saludos"), SIEMPRE trátala como RELEVANTE aunque no tenga contenido comercial.
   - Para estos casos:
        "is_relevant": true
        "should_reject": false
        "can_use_history": false
   - Motivo: Los saludos deben recibir una respuesta amable e iniciar conversación, seguido de una invitación a continuar con el entrenamiento.

7. Puedes analizar si la pregunta puede ser respondida aunque el contexto no sea tan claro, es decir, si tú como inteligencia artificial puedes entender la pregunta y responderla y es pertinente con el objetivo del chat, entonces is_relevant = true.

INSTRUCCIONES PARA RESPONDER:
- PRIMERO Y MÁS IMPORTANTE: Revisa cuidadosamente el HISTORIAL DE CONVERSACIÓN ANTERIOR arriba. SIEMPRE úsalo para entender el contexto y dar continuidad a la conversación.
- Si la pregunta actual contiene palabras como "eso", "ese", "esa", "ellos", "ellas", "lo que dijiste", "eso que mencionaste", "eso que explicaste", "eso de", o cualquier referencia implícita, DEBES buscar en el historial a qué se refiere.
- Si la pregunta es una continuación, aclaración o seguimiento de algo mencionado en el historial, usa ese contexto para entender la pregunta.
- NO respondas como si fuera una pregunta nueva sin contexto. SIEMPRE verifica si hay referencias al historial antes de responder.
- Analiza si la pregunta tiene sentido en el contexto de ventas legítimas, técnicas comerciales, mejora profesional o tecnología en contexto comercial.
- Si hay información en "INFORMACIÓN RELEVANTE DEL CONTENIDO" arriba, ÚSALA como base principal de tu respuesta. Si no hay información del contenido, responde basándote en tu conocimiento como mentor comercial experto Y el contexto del historial.
- Si la pregunta está relacionada con ventas pero de forma confusa o mal formulada, intenta entender la intención usando el historial y ayuda a reformular la pregunta de forma útil.
- Responde de forma natural, conversacional y profesional - como un mentor hablando con su aprendiz.
- No menciones que consultas documentos o información externa - simplemente responde como un experto que conoce el tema.
- Mantén un tono natural y cercano, pero profesional. Usa frases como "Excelente pregunta...", "Perfecto, déjame explicarte...", "Entiendo tu situación...", etc.
- SIEMPRE mantén un enfoque ético y profesional.
- **OBLIGATORIO SI RECHAZAS UNA PREGUNTA**: Si decides rechazar una pregunta, DEBES explicarte claramente POR QUÉ la rechazaste. Explicación es OBLIGATORIA y debe aparecer SIEMPRE que rechaces una pregunta.

FORMATO DE RESPUESTA (MUY IMPORTANTE):
- Usa formato Markdown para estructurar tus respuestas de forma clara y visualmente atractiva.
- Usa títulos con ## para secciones principales, ### para subsecciones.
- Usa **texto en negrita** para resaltar conceptos importantes, nombres de productos, o términos clave.
- Usa listas con viñetas (-) para enumerar características, beneficios, pasos, o elementos.
- Usa listas numeradas (1., 2., 3.) para procesos secuenciales o pasos ordenados.
- Estructura tu respuesta de forma jerárquica y organizada, similar a un documento profesional.
- Si rechazas una pregunta, SIEMPRE da una breve una explicación e invita a que se pueda continuar la conversación preguntando sobre algo que si sea relevante.
- Mantén un tono natural y conversacional, no robótico. Sé cercano pero profesional.

Ahora genera tu respuesta directamente. Responde como un mentor comercial experto basándote en el OBJETIVO DEL CHAT y las instrucciones anteriores.
"""

    try:
        # Usar el LLM global
        if llm is None:
            response_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                timeout=30,
                api_key=OPENAI_API_KEY
            )
        else:
            response_llm = llm
        
        response = response_llm.invoke(validation_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        answer = answer.strip() if answer else ""
        
        # Determinar relevancia basándose en la respuesta
        # Si la respuesta es muy corta o contiene frases de rechazo, probablemente fue rechazada
        answer_lower = answer.lower()
        should_reject = any(phrase in answer_lower for phrase in [
            "no puedo ayudar", "no puedo asistir", "no es apropiado", 
            "no está relacionado", "fuera de mi alcance"
        ])
        
        # Si hay contexto RAG, es relevante
        is_relevant = has_rag_context or not should_reject
        can_use_history = has_history and len(question.split()) <= 8
        
        logger.info(f"Respuesta generada: is_relevant={is_relevant}, should_reject={should_reject}")
        print(f"\n{'='*80}")
        print(f"🔍 [DEBUG] Respuesta generada:")
        print(f"  - Pregunta: '{question}'")
        print(f"  - Respuesta generada: {len(answer)} caracteres")
        print(f"  - should_reject: {should_reject}")
        print(f"{'='*80}\n")
        
        return answer, is_relevant, "Respuesta generada", can_use_history, should_reject
        
    except Exception as e:
        logger.error(f"Error al generar respuesta: {e}")
        # Fallback: respuesta de error
        error_answer = "Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo."
        can_use = has_history and len(question.split()) <= 8
        return error_answer, True, f"Error en generación: {e}", can_use, False

# Función para generar respuesta
def generate(state: State):
    """Genera respuesta inteligente usando OpenAI basada en el contexto recuperado."""
    
    # Usar validate_relevance que ahora genera la respuesta directamente
    context_docs = state.get("context", [])
    history = state.get("history", [])
    answer, is_relevant, reason, can_use_history, should_reject = validate_relevance(
        state["question"], 
        context_docs, 
        history
    )
    
    return {"answer": answer}

# URLs de páginas web para extraer información
# Grupo 1: Páginas con crawling (navega por enlaces internos)
paginas_con_crawl = [
    "https://www.strategysoftware.com/es",
    "https://www.sophos.com/es-es",
    "https://www.appgate.com/",
    "https://www.sentisis.com/",
    "https://www.auraquantic.com/es/",
    "https://www.checkpoint.com/es/"
]

# Grupo 2: Páginas sin crawling (solo la página especificada)
paginas_sin_crawl = [
    "https://aws.amazon.com/es/ai/",
    "https://aws.amazon.com/es/quicksuite/",
    "https://aws.amazon.com/es/",
    "https://aws.amazon.com/es/security/",
    "https://www.motorolasolutions.com/es_xl.html",
    "https://www.motorolasolutions.com/es_xl/video-security-analytics/fixed-video-security.html",
    "https://www.motorolasolutions.com/en_xl/video-security-access-control/body-cameras-and-in-car-video.html"
]

def crawl_website(start_url, max_pages=50, max_depth=2, delay=1.5):
    """
    Crawler que navega por los enlaces internos de una página web.
    
    Args:
        start_url: URL inicial para comenzar el crawling
        max_pages: Número máximo de páginas a visitar por sitio
        max_depth: Profundidad máxima de navegación (0 = solo la página inicial)
        delay: Tiempo de espera entre requests (segundos)
    
    Returns:
        Lista de documentos LangChain con el contenido extraído
    """
    visited = set()
    normalized_visited = set()  # Para evitar URLs normalizadas duplicadas
    to_visit = [(start_url, 0)]  # (url, depth)
    documents = []
    base_domain = urlparse(start_url).netloc
    user_agent = os.getenv("USER_AGENT", "mi-usuario-personalizado/0.0.1")
    
    while to_visit and len(visited) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        # Normalizar URL antes de verificar
        normalized_url = normalize_url(current_url)
        
        # Saltar si ya fue visitada (normalizada) o excede profundidad
        if normalized_url in normalized_visited or depth > max_depth:
            continue
        
        # Verificar que sea del mismo dominio
        current_domain = urlparse(normalized_url).netloc
        if current_domain != base_domain:
            continue
        
        try:
            # Hacer request con headers apropiados
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
            }
            
            response = requests.get(current_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Verificar que sea HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                normalized_visited.add(normalized_url)
                visited.add(current_url)
                continue
            
            # Parsear HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraer texto principal (remover scripts, styles, etc.)
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Obtener texto
            text = soup.get_text(separator=' ', strip=True)
            
            # Limpiar espacios múltiples y caracteres especiales
            text = ' '.join(text.split())
            # Remover contenido muy corto o que parece basura
            text = re.sub(r'\s+', ' ', text)
            
            # Validar contenido: debe tener al menos 150 caracteres y no ser solo números/símbolos
            if text and len(text) > 150 and len(set(text)) > 10:
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': normalized_url,
                        'source_file': normalized_url,
                        'source_type': 'web',
                        'depth': depth
                    }
                )
                documents.append(doc)
            
            # Marcar como visitada (tanto original como normalizada)
            visited.add(current_url)
            normalized_visited.add(normalized_url)
            
            # Si no hemos alcanzado la profundidad máxima, extraer enlaces
            if depth < max_depth:
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if not href or href.startswith('#') or href.startswith('javascript:'):
                        continue
                    
                    # Convertir enlace relativo a absoluto
                    absolute_url = urljoin(current_url, href)
                    
                    # Normalizar URL
                    normalized_link = normalize_url(absolute_url)
                    
                    # Filtrar URLs no válidas
                    parsed = urlparse(normalized_link)
                    if parsed.scheme not in ['http', 'https']:
                        continue
                    
                    # Solo agregar si es del mismo dominio y no fue visitada (normalizada)
                    if parsed.netloc == base_domain and normalized_link not in normalized_visited:
                        if normalized_link not in [normalize_url(url) for url, _ in to_visit]:
                            to_visit.append((absolute_url, depth + 1))
            
            # Delay entre requests para evitar ser bloqueado
            if delay > 0:
                time.sleep(delay)
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout al cargar {current_url}")
            normalized_visited.add(normalized_url)
            visited.add(current_url)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error al cargar {current_url}: {e}")
            normalized_visited.add(normalized_url)
            visited.add(current_url)
        except Exception as e:
            logger.error(f"Error inesperado en {current_url}: {e}")
            normalized_visited.add(normalized_url)
            visited.add(current_url)
    
    return documents

def load_single_page(url):
    """
    Carga el contenido de una sola página web sin hacer crawling.
    
    Args:
        url: URL de la página a cargar
    
    Returns:
        Lista de documentos LangChain con el contenido extraído
    """
    documents = []
    user_agent = os.getenv("USER_AGENT", "mi-usuario-personalizado/0.0.1")
    
    try:
        normalized_url = normalize_url(url)
        
        # Hacer request con headers apropiados
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Verificar que sea HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return documents
        
        # Parsear HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer texto principal (remover scripts, styles, etc.)
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Obtener texto
        text = soup.get_text(separator=' ', strip=True)
        
        # Limpiar espacios múltiples
        text = ' '.join(text.split())
        text = re.sub(r'\s+', ' ', text)
        
        # Validar contenido: debe tener al menos 150 caracteres
        if text and len(text) > 150 and len(set(text)) > 10:
            doc = Document(
                page_content=text,
                metadata={
                    'source': normalized_url,
                    'source_file': normalized_url,
                    'source_type': 'web',
                    'depth': 0
                }
            )
            documents.append(doc)
            
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout al cargar {url}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error al cargar {url}: {e}")
    except Exception as e:
        logger.error(f"Error inesperado en {url}: {e}")
    
    return documents

# Inicializar el sistema
def initialize_system():
    global vector_store, llm, graph
    
    print("Inicializando sistema...")
    
    # Directorio para persistencia de FAISS
    faiss_index_dir = "faiss_index"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Intentar cargar índice existente
    if os.path.exists(faiss_index_dir) and os.path.exists(os.path.join(faiss_index_dir, "index.faiss")):
        try:
            print("Cargando índice FAISS existente...")
            vector_store = FAISS.load_local(faiss_index_dir, embedding_model, allow_dangerous_deserialization=True)
            print("Índice FAISS cargado exitosamente.")
            
            # Configurar OpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.8,
                timeout=30,
                api_key=OPENAI_API_KEY  # Pasar explícitamente la key del archivo
            )
            
            # Compilar grafo
            graph_builder = StateGraph(State).add_sequence([retrieve, generate])
            graph_builder.add_edge(START, "retrieve")
            graph = graph_builder.compile()
            return
        except Exception as e:
            logger.warning(f"Error al cargar índice FAISS: {e}. Reconstruyendo...")
    
    # Si no existe o falló, construir desde cero
    all_docs = []
    
    # Cargar contenido de páginas web
    total_paginas = len(paginas_con_crawl) + len(paginas_sin_crawl)
    
    if total_paginas > 0:
        # Grupo 1: Páginas con crawling (navega por enlaces internos)
        if paginas_con_crawl:
            for url in paginas_con_crawl:
                try:
                    docs = crawl_website(url, max_pages=50, max_depth=2, delay=1.5)
                    all_docs.extend(docs)
                    logger.info(f"Crawleado {url}: {len(docs)} páginas")
                except Exception as e:
                    logger.error(f"Error al crawlear {url}: {e}")
        
        # Grupo 2: Páginas sin crawling (solo la página especificada)
        if paginas_sin_crawl:
            for url in paginas_sin_crawl:
                try:
                    docs = load_single_page(url)
                    all_docs.extend(docs)
                    logger.info(f"Cargada página {url}: {len(docs)} documentos")
                except Exception as e:
                    logger.error(f"Error al cargar {url}: {e}")
    
    # Cargar todos los PDFs de la carpeta fuentes dinámicamente
    fuentes_dir = "fuentes"
    
    if not os.path.exists(fuentes_dir):
        os.makedirs(fuentes_dir, exist_ok=True)
    
    # Recorrer todos los archivos PDF en la carpeta fuentes
    pdf_files = [f for f in os.listdir(fuentes_dir) if f.lower().endswith('.pdf')]
    
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(fuentes_dir, pdf_file)
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                valid_docs = []
                # Asegurar que cada documento tenga metadata coherente y contenido válido
                for doc in docs:
                    if not hasattr(doc, 'page_content') or not doc.page_content:
                        continue
                    content = doc.page_content.strip()
                    if len(content) < 50:  # Filtrar documentos muy cortos
                        continue
                    
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['source_file'] = pdf_file
                    doc.metadata['source'] = pdf_path
                    doc.metadata['source_type'] = 'pdf'
                    valid_docs.append(doc)
                
                all_docs.extend(valid_docs)
                logger.info(f"Cargado PDF {pdf_file}: {len(valid_docs)} páginas válidas")
            except Exception as e:
                logger.warning(f"Error al cargar PDF {pdf_file}: {e}. El archivo puede estar corrupto.")
    
    if not all_docs:
        raise Exception(f"No se pudieron cargar documentos. Asegúrate de que haya al menos un archivo PDF válido en '{fuentes_dir}' o URLs válidas en los arrays 'paginas_con_crawl' o 'paginas_sin_crawl'.")
    
    # Dividir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_docs)
    
    # Filtrar splits vacíos o muy cortos
    valid_splits = [s for s in all_splits if s.page_content and len(s.page_content.strip()) > 50]
    
    # Vector store
    vector_store = FAISS.from_documents(valid_splits, embedding_model)
    
    # Guardar índice para futuras cargas
    try:
        os.makedirs(faiss_index_dir, exist_ok=True)
        vector_store.save_local(faiss_index_dir)
        logger.info(f"Índice FAISS guardado en {faiss_index_dir}")
    except Exception as e:
        logger.warning(f"No se pudo guardar el índice FAISS: {e}")
    
    # Configurar OpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,
        timeout=30,
        api_key=OPENAI_API_KEY  # Pasar explícitamente la key del archivo
    )
    
    # Compilar grafo
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

# Ruta de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Si ya está logueado, redirigir al chat
    if session.get('logged_in'):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and USERS[username] == password:
            session.clear()  # Limpiar sesión anterior
            session['logged_in'] = True
            session['username'] = username
            session['session_token'] = SESSION_TOKEN  # Token único de este servidor
            session.permanent = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Usuario o contraseña incorrectos')
    
    return render_template('login.html')

# Ruta de logout
@app.route('/logout')
def logout():
    session.clear()
    # Asegurar que el historial se limpie
    if 'history' in session:
        del session['history']
    return redirect(url_for('login'))

# Ruta principal (protegida)
@app.route('/')
def index():
    # Verificar autenticación completa incluyendo token de sesión
    if not session.get('logged_in') or \
       not session.get('username') or \
       session.get('session_token') != SESSION_TOKEN:
        session.clear()  # Limpiar cualquier sesión inválida
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

# Endpoint para procesar preguntas (protegido)
@app.route('/chat', methods=['POST'])
def chat():
    if not session.get('logged_in') or \
       not session.get('username') or \
       session.get('session_token') != SESSION_TOKEN:
        return jsonify({'error': 'No autorizado'}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Datos inválidos'}), 400
        
        question = data.get('question', '').strip()
        
        # Sanitizar input
        question = re.sub(r'[^\w\s\?¿¡!.,;:áéíóúñÁÉÍÓÚÑ\-]', '', question)
        question = question[:500]  # Limitar longitud
        
        if not question or len(question) < 2:
            return jsonify({
                'answer': "¿En qué puedo ayudarte? También puedes escribir 'tutorial' para ver ejemplos."
            })
        
        # Inicializar historial si no existe
        if 'history' not in session:
            session['history'] = []
        
        # Obtener historial (últimos 12 mensajes)
        history = session['history'][-12:] if session['history'] else []
        
        # Manejar la palabra "tutorial" de forma especial (sin buscar en documentos)
        if question.lower() == 'tutorial':
            # Construir historial para el tutorial
            history_text = ""
            if history:
                history_text = "\n\nHISTORIAL DE CONVERSACIÓN:\n"
                for msg in history[-10:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        history_text += f"Usuario: {content}\n"
                    elif role == "assistant":
                        history_text += f"Asistente: {content}\n"
            
            base_prompt = build_base_prompt()
            tutorial_prompt = f"""{base_prompt}

{history_text}

El usuario ha escrito la palabra "tutorial". 

INSTRUCCIONES:
- Responde de forma natural, conversacional y profesional - como un mentor hablando con su aprendiz.
- Comienza con un saludo natural (ej: "Excelente pregunta" o "Perfecto, aquí tienes una guía completa...").
- Debes mostrar la guía completa de ejemplos de uso según lo que requiera en su proceso de ventas.
- Incluye ejemplos prácticos para cada una de las 7 etapas del proceso de ventas.
- Proporciona ejemplos de preguntas que el usuario puede hacer en cada etapa.
- Responde directamente con el tutorial completo, sin mencionar que consultas documentos.
- Sé claro, directo, práctico y natural. Usa un tono conversacional, no robótico.

FORMATO DE RESPUESTA (MUY IMPORTANTE):
- Usa formato Markdown para estructurar el tutorial de forma clara y visualmente atractiva.
- Usa ## para el título principal del tutorial o para cada etapa del proceso de ventas (ej: ## 1. PROSPECCIÓN INTELIGENTE).
- Usa ### para subsecciones dentro de cada etapa (ej: ### ¿Qué puedes pedirme?).
- Usa **texto en negrita** para resaltar conceptos importantes, nombres de metodologías, o términos clave.
- Usa listas con viñetas (-) para enumerar ejemplos, herramientas, preguntas, etc.
- Estructura el tutorial de forma jerárquica y profesional, pero mantén un tono natural y conversacional.
- Ejemplo de tono: "Excelente pregunta. Te puedo ayudar con..." o "Perfecto, aquí tienes tu guía práctica..."""
            
            try:
                response = llm.invoke(tutorial_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                answer = answer.strip() if answer else ""
                
                # Si la respuesta está vacía, intentar regenerar
                if not answer:
                    logger.warning("Respuesta del tutorial vacía, regenerando...")
                    base_prompt = build_base_prompt()
                    simple_prompt = f"""{base_prompt}

{history_text}

El usuario escribió "tutorial". Genera una guía completa de ejemplos de uso según lo que requiera en su proceso de ventas."""
                    response = llm.invoke(simple_prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    answer = answer.strip() if answer else ""
                
                # Guardar en historial
                session['history'].append({"role": "user", "content": question})
                session['history'].append({"role": "assistant", "content": answer})
                session['history'] = session['history'][-12:]  # Mantener últimos 12
                session.modified = True
                
                return jsonify({
                    'answer': answer
                })
            except Exception as e:
                logger.error(f"Error al generar tutorial: {e}")
                # Intentar generar respuesta de error con el LLM
                try:
                    base_prompt = build_base_prompt()
                    error_prompt = f"""{base_prompt}

{history_text}

El usuario escribió "tutorial" pero hubo un error técnico al generar la respuesta. Genera un mensaje amable explicando que hubo un error y que puede intentar de nuevo."""
                    response = llm.invoke(error_prompt)
                    error_answer = response.content if hasattr(response, 'content') else str(response)
                    return jsonify({
                        'answer': error_answer.strip() if error_answer else ""
                    }), 500
                except:
                    # Solo como último recurso, respuesta mínima
                    return jsonify({
                        'answer': ''
                    }), 500
        
        # Para otras preguntas, usar el sistema RAG normal
        # Agregar pregunta al historial
        session['history'].append({"role": "user", "content": question})
        
        # Invocar el grafo con historial
        initial_state = State(
            question=question,
            context=[],
            answer="",
            history=history
        )
        
        try:
            final_state = graph.invoke(initial_state)
            answer = final_state.get('answer', '')
            
            # Validar respuesta - si está vacía, dejar que el LLM maneje el error en la siguiente iteración
            # No usar respuestas hardcodeadas
            
            # Guardar respuesta en historial
            session['history'].append({"role": "assistant", "content": answer})
            session['history'] = session['history'][-12:]  # Mantener últimos 12
            session.modified = True
            
            return jsonify({
                'answer': answer
            })
        except Exception as e:
            logger.error(f"Error en el grafo: {e}")
            # Remover pregunta del historial si falló
            if session['history'] and session['history'][-1].get('role') == 'user':
                session['history'].pop()
            return jsonify({
                'answer': 'Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo.'
            }), 500
    
    except Exception as e:
        logger.error(f"Error en endpoint /chat: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

# Endpoint para generar audio con Eleven Labs TTS
@app.route('/tts', methods=['POST'])
def text_to_speech():
    if not session.get('logged_in') or \
       not session.get('username') or \
       session.get('session_token') != SESSION_TOKEN:
        return jsonify({'error': 'No autorizado'}), 401
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Texto vacío'}), 400
        
        # Obtener configuración de Eleven Labs
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        if not api_key:
            return jsonify({'error': 'API key de Eleven Labs no configurada'}), 500
        
        # Llamar a la API de Eleven Labs
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # Modelo gratuito multilingüe
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            # Retornar el audio como respuesta
            return Response(
                response.content,
                mimetype='audio/mpeg',
                headers={
                    'Content-Disposition': 'inline; filename=audio.mp3'
                }
            )
        else:
            error_msg = response.json().get('detail', {}).get('message', 'Error desconocido') if response.headers.get('content-type', '').startswith('application/json') else 'Error al generar audio'
            return jsonify({'error': f'Error de Eleven Labs: {error_msg}'}), response.status_code
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_system()
    
    # Obtener configuración desde variables de entorno
    debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
    port = int(os.getenv("PORT", "5001"))
    
    print("\n" + "="*60)
    print(f"Servidor web iniciado en http://localhost:{port}")
    print(f"Modo Debug: {debug_mode}")
    print(f"Token de sesión: {SESSION_TOKEN[:8]}...")
    print("="*60)
    print("IMPORTANTE: Si tenías el navegador abierto, ciérralo")
    print("completamente y vuelve a abrirlo para limpiar las cookies.")
    print("="*60 + "\n")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

