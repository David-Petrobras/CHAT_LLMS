import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from configparser import ConfigParser, ExtendedInterpolation
import httpx
import os
from io import StringIO
import sys
import uuid
from streamlit_option_menu import option_menu

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Dados Inteligente",
    page_icon="C:\\Users\\FUDO\\Documents\\CHATs\\imagens\\petrobras_icone.png",  # ATENÇÃO: Caminho fixo
    layout="wide"
)

# --- Função de Carregamento de Configurações do App ---
def load_app_config():
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_path = 'C:\\Users\\FUDO\\Documents\\CHATs\\code\\config 1.ini'  # ATENÇÃO: Caminho fixo
    if not os.path.exists(config_file_path):
        st.error(f"Arquivo '{config_file_path}' não encontrado!")
        st.stop()
    config.read(config_file_path, 'UTF-8')
    if 'OPENAI' not in config:
        st.error(f"Seção [OPENAI] não encontrada em '{config_file_path}'.")
        st.stop()
    return config

# Carrega config de app uma única vez na sessão
if 'app_config' not in st.session_state:
    st.session_state.app_config = load_app_config()
app_config = st.session_state.app_config

# --- Cliente Azure OpenAI ---
@st.cache_resource
def get_openai_client_cached():
    try:
        api_key = app_config["OPENAI"]["OPENAI_API_KEY"]
        base_url = app_config["OPENAI"]["OPENAI_API_BASE"]
        api_version = app_config["OPENAI"]["OPENAI_API_VERSION"]
        cert_path = "C:/Users/FUDO/Documents/CHATs/petrobras-ca-root.pem"  # ATENÇÃO: Caminho fixo
        http_client_instance = None
        if os.path.exists(cert_path):
            http_client_instance = httpx.Client(verify=cert_path)
        else:
            st.warning(f"Arquivo PEM '{cert_path}' não encontrado, prosseguindo sem verificação de cliente SSL personalizada.")
        client = AzureOpenAI(api_key=api_key, api_version=api_version, base_url=base_url, http_client=http_client_instance)
        return client
    except Exception as e:
        st.error(f"Erro ao inicializar cliente Azure OpenAI: {e}")
        return None

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = get_openai_client_cached()
openai_client = st.session_state.openai_client

MODEL_DEPLOYMENT_NAME = app_config.get('OPENAI', 'CHATGPT_MODEL', fallback=None) if app_config else None
if openai_client and not MODEL_DEPLOYMENT_NAME:
    st.error("CHATGPT_MODEL não configurado em [OPENAI] no config.ini.")
    openai_client = None

# --- Funções Auxiliares ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext == 'csv':
                return pd.read_csv(uploaded_file)
            if ext in ['xls', 'xlsx']:
                return pd.read_excel(uploaded_file)
            st.error("Formato não suportado.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    return None

def generate_pandas_code_with_llm(client, deployment_name, df_schema_str, question):
    system_message = f"""
Você é um programador Python especialista em análise de dados com a biblioteca Pandas.
Um DataFrame Pandas chamado `df` já está carregado e disponível para você usar.

Aqui está o esquema e uma amostra do DataFrame `df`:
{df_schema_str}
(O DataFrame `df` completo tem {st.session_state.get('df_rows', 0)} linhas e {st.session_state.get('df_cols', 0)} colunas)

Pergunta do usuário: "{question}"

Sua tarefa é escrever um script Python curto que use o DataFrame `df` para responder à pergunta.
Seu script DEVE:
1. Realizar as operações Pandas necessárias em `df`.
2. Calcular ou extrair a informação final que responde à pergunta.
3. **Imprimir (`print()`) este resultado final para a saída padrão.** A saída impressa deve ser concisa:
    - Se o resultado for um DataFrame e a pergunta do usuário sugerir ou pedir explicitamente uma tabela, use `print(nome_do_dataframe.to_markdown(index=False))` para gerar uma tabela em formato Markdown. Certifique-se de que `nome_do_dataframe` seja o DataFrame final com o resultado. Limite a um número razoável de linhas se o resultado for extenso, a menos que a pergunta implique o contrário.
    - Para outros resumos de DataFrame (que não são explicitamente pedidos como tabela), você pode usar `print(nome_do_dataframe.to_string())`.
    - Para outros tipos de dados (string, número, lista), imprima diretamente o valor.
4. **Seu código deve produzir APENAS o resultado final via `print()`. Não inclua `print()` para etapas intermediárias.**
5. **Responda APENAS com o bloco de código Python. Não inclua explicações, introduções, ou markdown como ```python ... ```.**
6. Use apenas operações seguras e de leitura no DataFrame `df`. Não modifique `df` permanentemente (criar novas variáveis é permitido). Não tente acessar arquivos ou rede.
7. Se a pergunta não puder ser respondida com os dados de `df` ou for muito ambígua para gerar um código preciso, imprima uma mensagem informativa curta explicando o motivo (ex: `print("Não é possível determinar X com os dados fornecidos.")`).
"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    response_kwargs = {
        "model": deployment_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 800
    }
    try:
        response = client.chat.completions.create(**response_kwargs)
        generated_code = response.choices[0].message.content.strip()
        if generated_code.startswith("```python"):
            generated_code = generated_code[len("```python"):].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-len("```")].strip()
        elif generated_code.startswith("```") and generated_code.endswith("```"):
            generated_code = generated_code[len("```"): -len("```")].strip()

        if not generated_code or ("print(" not in generated_code and ".to_markdown(" not in generated_code):
            return None, "Código gerado pela IA inválido (sem print ou to_markdown)."
        return generated_code, None
    except Exception as e:
        return None, f"Erro na API OpenAI ao gerar código: {e}"

def execute_generated_pandas_code(code_string, dataframe_full):
    output_capture = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture

    safe_builtins = {
        'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'True': True, 'False': False, 'None': None, 'range': range,
        'abs': abs, 'round': round, 'max': max, 'min': min, 'sum': sum,
        'sorted': sorted, 'zip': zip, 'enumerate': enumerate,
        '__import__': __import__,
    }
    allowed_globals = {"df": dataframe_full, "pd": pd, "__builtins__": safe_builtins}

    script_output_val = None
    error_message = None
    try:
        exec(code_string, allowed_globals)
        script_output_val = output_capture.getvalue().strip()
    except Exception as e:
        error_message = f"Erro ao executar código gerado pela IA: {type(e).__name__}: {e}"
    finally:
        sys.stdout = original_stdout

    if error_message:
        return None, f"{error_message}\n\n--- Código que falhou ---\n```python\n{code_string}\n```"
    if not script_output_val and not error_message:
        return "O código gerado pela IA não produziu uma saída para exibir (sem print).", None
    return script_output_val, None

def get_final_answer_from_llm(client, deployment_name, question, script_output, generated_code_for_prompt):
    system_message = (
        "Você é um assistente de IA que explica resultados de análises de dados de forma clara. "
        "O usuário fez uma pergunta. Um script Python/Pandas foi gerado e executado no dataset completo para respondê-la. "
        "A saída direta desse script é fornecida abaixo."
    )
    user_prompt = (
        f"Pergunta original do usuário: \"{question}\"\n\n"
        f"O seguinte código Pandas foi executado no dataset completo:\n```python\n{generated_code_for_prompt}\n```\n\n"
        f"A saída/resultado direto desse código foi:\n---\n{script_output}\n---\n\n"
        "Com base nesta saída EXATA do código, formule uma resposta concisa para o usuário.\n"
        "- Se a pergunta original do usuário pedia explicitamente por uma tabela E a 'saída direta desse código' representa dados tabulares (por exemplo, já está em formato Markdown de tabela ou é uma representação de string de um DataFrame que pode ser convertida em tabela), sua resposta DEVE incluir esses dados formatados como uma tabela Markdown. Certifique-se de que a tabela Markdown esteja corretamente formatada (cabeçalhos separados por `|`, linha de separadores `|---|---|`, e linhas de dados também usando `|`).\n"
        "- Caso contrário, ou se a saída não for tabular, explique os resultados em linguagem natural clara.\n"
        "- Se a saída indicar que a pergunta não pôde ser respondida ou se for um erro (incluindo erros de execução do código fornecido), explique isso de forma útil.\n"
        "- Não refaça os cálculos, apenas interprete e formate o resultado fornecido."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Resultado da análise (saída direta do script): {script_output}. (Erro na API OpenAI ao formatar resposta final: {e})."

# --- Inicialização do Estado da Sessão ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_rows' not in st.session_state:
    st.session_state.df_rows = 0
if 'df_cols' not in st.session_state:
    st.session_state.df_cols = 0
if 'uploaded_file_id_key' not in st.session_state:
    st.session_state.uploaded_file_id_key = None
if "all_conversations" not in st.session_state:
    st.session_state.all_conversations = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "new_chat_pending" not in st.session_state:
    st.session_state.new_chat_pending = True
if "selected_view" not in st.session_state:
    st.session_state.selected_view = "Dados"

# --- Funções de Gerenciamento de Conversa ---
def get_current_chat_exchanges():
    if st.session_state.current_conversation_id and st.session_state.all_conversations:
        for conv in st.session_state.all_conversations:
            if conv['id'] == st.session_state.current_conversation_id:
                return conv['exchanges']
    return []

def add_exchange_to_conversation(exchange_item):
    if st.session_state.new_chat_pending:
        new_conv_id = str(uuid.uuid4())
        title = exchange_item["user_query"][:40].strip() + "..." if len(exchange_item["user_query"]) > 40 else exchange_item["user_query"].strip()
        if not title:
            title = f"Conversa {len(st.session_state.all_conversations) + 1}"
        base_title = title
        counter = 1
        existing_titles = {conv['title'] for conv in st.session_state.all_conversations}
        while title in existing_titles:
            title = f"{base_title} ({counter})"
            counter += 1
        new_conversation = {"id": new_conv_id, "title": title, "exchanges": [exchange_item]}
        st.session_state.all_conversations.insert(0, new_conversation)
        st.session_state.current_conversation_id = new_conv_id
        st.session_state.new_chat_pending = False
    else:
        found = False
        for conv in st.session_state.all_conversations:
            if conv['id'] == st.session_state.current_conversation_id:
                conv['exchanges'].append(exchange_item)
                found = True
                break
        if not found and st.session_state.current_conversation_id:
            st.warning("Não foi possível adicionar à conversa atual. Iniciando uma nova.")
            st.session_state.new_chat_pending = True
            add_exchange_to_conversation(exchange_item)

# --- Sidebar ---
st.sidebar.title("Assistente de Análise de Dados para Orçamentação Referencial")
st.sidebar.markdown("Faça upload do seu arquivo de dados, visualize e converse com a IA para análises!")
st.sidebar.markdown("---")

# Label do uploader encurtado
uploaded_file = st.sidebar.file_uploader("Upload (CSV, XLS, XLSX):", type=['csv', 'xls', 'xlsx'])

with st.sidebar:
    options_list = ["Visualize seus Dados", "Converse com à IA"]
    icons_list = ['clipboard-data', 'robot']

    try:
        default_nav_index = options_list.index(st.session_state.selected_view)
    except ValueError:
        default_nav_index = 0
        st.session_state.selected_view = options_list[0]

    selected_nav_option = option_menu(
        menu_title=None,  # Título do menu removido
        options=options_list,
        icons=icons_list,
        menu_icon=None,
        default_index=default_nav_index,
        orientation="vertical",  # ALTERADO PARA VERTICAL
        key="sidebar_nav_menu_vertical",
        styles={  # Estilos ajustados para layout vertical
            "container": {"padding": "0px !important", "background-color": "transparent", "margin-bottom": "10px"},
            "icon": {"font-size": "15px", "margin-right": "10px", "vertical-align": "middle"},
            "nav-link": {
                "font-size": "14px",
                "padding": "10px 15px",  # Padding para botões verticais
                "margin": "3px 0px",    # Margem vertical entre botões
                "border-radius": "5px",
                "border": "1px solid #ddd",  # Borda leve
                "background-color": "#f9f9f9",
                "color": "#333",
                "text-align": "left",   # Alinhar texto à esquerda
                "width": "100%"          # Fazer o botão ocupar a largura da sidebar
            },
            "nav-link-selected": {
                "background-color": "#156d24",
                "color": "white",
                "border": "1px solid #007bff",
                "font-weight": "500"
            },
        }
    )
    if selected_nav_option != st.session_state.selected_view:
        st.session_state.selected_view = selected_nav_option
        st.rerun()

if st.sidebar.button("➕ Nova Conversa", key="new_chat_button_sidebar", use_container_width=True):
    st.session_state.new_chat_pending = True
    st.session_state.current_conversation_id = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### Histórico de Conversas")

if not st.session_state.all_conversations:
    st.sidebar.caption("Nenhuma conversa registrada.")
else:
    for conv_idx, conv in enumerate(st.session_state.all_conversations):
        button_label = f"💬 {conv['title']}"
        is_current = (conv['id'] == st.session_state.current_conversation_id and not st.session_state.new_chat_pending)
        button_type = "primary" if is_current else "secondary"
        if st.sidebar.button(button_label, key=f"conv_select_{conv['id']}", type=button_type, use_container_width=True):
            st.session_state.current_conversation_id = conv['id']
            st.session_state.new_chat_pending = False
            st.rerun()

#st.sidebar.markdown("---")
#st.sidebar.info("Desenvolvido para análise interativa de dados com IA.")

# --- Lógica Principal do Aplicativo ---
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.uploaded_file_id_key != current_file_id:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.uploaded_file_id_key = current_file_id
        if st.session_state.df is not None:
            st.session_state.df_rows = len(st.session_state.df)
            st.session_state.df_cols = len(st.session_state.df.columns)
            st.success(f"Arquivo '{uploaded_file.name}' carregado: {st.session_state.df_rows} linhas, {st.session_state.df_cols} colunas.")
            st.session_state.all_conversations = []
            st.session_state.current_conversation_id = None
            st.session_state.new_chat_pending = True
        else:
            st.session_state.df = None
elif 'df' not in st.session_state or st.session_state.df is None:
    st.info("👈 Por favor, faça o upload de um arquivo Excel ou CSV na barra lateral para começar.")

if 'df' in st.session_state and st.session_state.df is not None:
    df_display = st.session_state.df

    if st.session_state.selected_view == "Dados":
        st.subheader("Visualização dos Dados Carregados")
        st.dataframe(df_display, height=500, use_container_width=True)

    elif st.session_state.selected_view == "IA":
        current_chat_exchanges = get_current_chat_exchanges()

        if openai_client and MODEL_DEPLOYMENT_NAME:
            user_question = st.chat_input("Digite sua pergunta sobre os dados...", key="chat_input_main_sidebar_nav_vertical")
            if user_question:
                with st.spinner("IA está processando sua solicitação... Por favor, aguarde. 🧠"):
                    schema_buffer = StringIO()
                    df_display.info(buf=schema_buffer)
                    df_info_str = schema_buffer.getvalue()
                    try:
                        df_head_str = df_display.head().to_string(max_rows=3, max_cols=7)
                    except Exception:
                        df_head_str = df_display.head().to_string()

                    df_schema_for_llm = (
                        f"Informações Gerais (df.info()):\n{df_info_str}\n\n"
                        f"Primeiras linhas (df.head().to_string(max_rows=3, max_cols=7)):\n{df_head_str}\n\n"
                        f"Nomes exatos das Colunas: {', '.join(df_display.columns)}\n"
                        f"Tipos de dados das colunas (dtypes):\n{df_display.dtypes.to_string()}"
                    )
                    MAX_SCHEMA_LEN = 7000
                    if len(df_schema_for_llm) > MAX_SCHEMA_LEN:
                        df_schema_for_llm = (
                            f"Primeiras linhas (df.head().to_string(max_rows=2, max_cols=5)):\n{df_display.head(2).to_string(max_cols=5)}\n\n"
                            f"Nomes exatos das Colunas: {', '.join(df_display.columns)}\n"
                            f"Tipos de dados das colunas (dtypes):\n{df_display.dtypes.iloc[:5].to_string()}..."
                        )

                    assistant_response_data = {}
                    final_answer_text = "Ocorreu um problema ao processar sua solicitação."
                    generated_code_val, script_output_val, exec_error_info, error_gen_code = None, None, None, None

                    generated_code_val, error_gen_code = generate_pandas_code_with_llm(
                        openai_client, MODEL_DEPLOYMENT_NAME, df_schema_for_llm, user_question
                    )

                    if generated_code_val:
                        assistant_response_data["generated_code"] = generated_code_val
                        script_output_val, exec_error_info = execute_generated_pandas_code(generated_code_val, df_display)

                        llm_input_script_result = ""
                        if exec_error_info:
                            llm_input_script_result = exec_error_info
                            assistant_response_data["script_output"] = exec_error_info
                        elif script_output_val is not None:
                            llm_input_script_result = script_output_val
                            assistant_response_data["script_output"] = script_output_val
                        else:
                            llm_input_script_result = "Nenhuma saída ou erro detalhado do script."
                            assistant_response_data["script_output"] = llm_input_script_result

                        final_answer_text = get_final_answer_from_llm(
                            openai_client, MODEL_DEPLOYMENT_NAME, user_question, llm_input_script_result, generated_code_val
                        )
                    else:
                        final_answer_text = f"A IA não conseguiu gerar o código Pandas para esta pergunta. Motivo: {error_gen_code if error_gen_code else 'Não especificado.'}"

                assistant_response_data["answer"] = final_answer_text
                new_exchange = {"user_query": user_question, "assistant_response": assistant_response_data}
                add_exchange_to_conversation(new_exchange)
                st.rerun()

        elif not openai_client or not MODEL_DEPLOYMENT_NAME:
            st.error("Cliente OpenAI ou nome do modelo não configurado. Verifique o arquivo config.ini e recarregue a página se necessário.")

        st.markdown("---")

        if not current_chat_exchanges and st.session_state.current_conversation_id:
            if openai_client and MODEL_DEPLOYMENT_NAME:
                st.caption("ℹ️ Esta conversa está vazia. Faça uma pergunta ou selecione outra conversa.")
        elif not current_chat_exchanges and not st.session_state.current_conversation_id and not st.session_state.new_chat_pending:
            if openai_client and MODEL_DEPLOYMENT_NAME:
                st.caption("ℹ️ Nenhuma conversa selecionada. Inicie uma nova ou escolha uma do histórico.")

        for exchange in reversed(current_chat_exchanges):
            with st.chat_message("user"):
                st.markdown(exchange["user_query"])
            if exchange.get("assistant_response"):
                with st.chat_message("assistant"):
                    assistant_content = exchange["assistant_response"]
                    st.markdown(assistant_content.get("answer", "Não foi possível obter uma resposta."))
                    if "generated_code" in assistant_content and assistant_content["generated_code"]:
                        with st.expander("Ver código Pandas gerado", expanded=False):
                            st.code(assistant_content["generated_code"], language="python", line_numbers=True)
                    if "script_output" in assistant_content and assistant_content["script_output"]:
                        with st.expander("Ver saída direta do script", expanded=False):
                            st.code(assistant_content["script_output"], language="text")
