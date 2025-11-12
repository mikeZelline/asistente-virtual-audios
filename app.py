import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env (si existe)
load_dotenv('config.env')

# Configura variables de entorno ANTES de los imports
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "mi-usuario-personalizado/0.0.1")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")



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
prompt_context = """Eres un asistente experto en contratación pública colombiana. 
Tu trabajo es responder preguntas basándote ÚNICAMENTE en la información que te proporciono.
Usa un tono amable y profesional, explicando en términos sencillos.
IMPORTANTE: Si el contexto contiene información relevante, DEBES responder con esa información.
No menciones que la información se está extrayendo de un documento pdf."""

# Estado (siguiendo patrón de LangChain)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Función para recuperar documentos relevantes
def retrieve(state: State):
    """Recupera los documentos más relevantes para la pregunta."""
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=3)
    
    # Debug para ver scores
    print(f"\n{'='*60}")
    print(f"[DEBUG] Pregunta: '{state['question']}'")
    print(f"[DEBUG] Scores obtenidos: {[round(score, 3) for _, score in retrieved_docs_with_scores]}")
    
    # Umbral más estricto para evitar respuestas irrelevantes
    SIMILARITY_THRESHOLD = 0.8
    
    relevant_docs = []
    for doc, score in retrieved_docs_with_scores:
        print(f"[DEBUG] Evaluando doc con score {round(score, 3)} vs threshold {SIMILARITY_THRESHOLD}")
        if score < SIMILARITY_THRESHOLD:
            relevant_docs.append(doc)
    
    # Si no hay documentos relevantes, no forzar respuesta
    if not relevant_docs:
        print(f"[DEBUG] Ningún documento relevante encontrado - rechazando pregunta")
    
    print(f"[DEBUG] Documentos relevantes FINALES: {len(relevant_docs)}")
    print(f"{'='*60}\n")
    
    return {"context": relevant_docs}

# Función para generar respuesta
def generate(state: State):
    """Genera respuesta inteligente usando OpenAI basada en el contexto recuperado."""
    
    print(f"[DEBUG GENERATE] Cantidad de documentos en contexto: {len(state['context'])}")
    
    # Usar prompt_context cuando no hay documentos relevantes
    if not state["context"]:
        print(f"[DEBUG GENERATE] Sin documentos - enviando mensaje de rechazo")
        no_results_prompt = f"""Eres un asistente experto en contratación pública colombiana y el SECOP. 

        Responde de forma amable y breve que solo puedes ayudar con preguntas sobre contratación pública y el SECOP, 
        Invita al usuario a hacer preguntas relacionadas con contratación pública colombiana y el SECOP.
        No digas hola, responde directamente."""
        
        try:
            response = llm.invoke(no_results_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            print(f"[DEBUG GENERATE] Respuesta de rechazo: {answer[:100]}...")
            return {"answer": answer.strip()}
        except Exception as e:
            print(f"[DEBUG GENERATE] Error en rechazo: {e}")
            return {"answer": "Soy un asistente especializado en contratación pública colombiana y el SECOP. Solo puedo ayudarte con preguntas relacionadas con estos temas. ¿En qué puedo asistirte sobre contratación pública?"}
    
    # Si hay contexto relevante, generar respuesta basada en el documento
    print(f"[DEBUG GENERATE] CON documentos - generando respuesta basada en contexto")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    context_limit = 2000
    context = docs_content[:context_limit].strip()
    
    print(f"[DEBUG GENERATE] Longitud del contexto: {len(context)} caracteres")
    print(f"[DEBUG GENERATE] Primeros 100 chars del contexto: {context[:100]}...")
    
    prompt = f"""{prompt_context}

INFORMACIÓN RELEVANTE:
{context}

PREGUNTA: {state['question']}

Responde la pregunta de forma clara, precisa y profesional. No menciones que consultas documentos o información externa - simplemente responde como un experto que conoce el tema:"""
    
    print(f"[DEBUG GENERATE] Invocando OpenAI...")
    try:
        response = llm.invoke(prompt)
        print(f"[DEBUG GENERATE] Respuesta recibida de OpenAI")
        print(f"[DEBUG GENERATE] Tipo de respuesta: {type(response)}")
        
        answer = response.content if hasattr(response, 'content') else str(response)
        answer = answer.strip()
        
        print(f"[DEBUG GENERATE] Respuesta extraída: {answer[:150]}...")
        
        # Limitar longitud si es muy larga
        if len(answer) > 3000:
            answer = answer[:3000]
            if '.' in answer:
                answer = answer[:answer.rfind('.')+1]
        
        print(f"[DEBUG GENERATE] Respuesta final (longitud: {len(answer)})")
    
    except Exception as e:
        print(f"[DEBUG GENERATE] ERROR al invocar OpenAI: {e}")
        answer = f"Error al generar respuesta: {str(e)}"
    
    return {"answer": answer}

# Inicializar el sistema
def initialize_system():
    global vector_store, llm, graph
    
    print("Inicializando sistema...")
    
    # Cargar PDF
    loader = PyPDFLoader("fuente.pdf")
    docs = loader.load()
    
    # Dividir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store
    vector_store = FAISS.from_documents(all_splits, embedding_model)
    
    # Configurar OpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Modelo rápido y económico
        temperature=0.8,
    )
    
    # Compilar grafo
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    print(f"Sistema inicializado: {len(docs)} páginas, {len(all_splits)} fragmentos.")

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
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Pregunta vacía'}), 400
        
        # Invocar el grafo
        initial_state = State(question=question, context=[], answer="")
        final_state = graph.invoke(initial_state)
        
        return jsonify({
            'answer': final_state['answer']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_system()
    
    # Obtener configuración desde variables de entorno
    debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
    port = int(os.getenv("PORT", "5000"))
    
    print("\n" + "="*60)
    print(f"Servidor web iniciado en http://localhost:{port}")
    print(f"Modo Debug: {debug_mode}")
    print(f"Token de sesión: {SESSION_TOKEN[:8]}...")
    print("="*60)
    print("IMPORTANTE: Si tenías el navegador abierto, ciérralo")
    print("completamente y vuelve a abrirlo para limpiar las cookies.")
    print("="*60 + "\n")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

