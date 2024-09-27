from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.exc import SQLAlchemyError
from mysql.connector.errors import ProgrammingError
from asr import initialize_whisper, transcribe_audio
import streamlit as st

db = None

llm = ChatOllama(model="llama3")

def connectDatabase(username, password, host, port, database):
    global db
    mysql_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
    st.session_state.db = SQLDatabase.from_uri(mysql_uri)

def runQuery(query):
    if st.session_state.db:
        try:
            return st.session_state.db.run(query)
        except ProgrammingError as pe:  # Catch specific programming errors
            error_message = str(pe)
            print(f"SQL query execution failed: {error_message}")
            
            # Detect if it's a missing column issue
            if 'Unknown column' in error_message:
                missing_column = error_message.split("'")[1]  # Extract the missing column name
                print(f"The specified column '{missing_column}' does not exist in the table. Please verify your column names.")
            # Detect if it's a missing table issue
            elif 'Table' in error_message:
                missing_table = error_message.split("'")[1]  # Extract the missing table name
                print(f"The specified table '{missing_table}' does not exist in the database. Please check your query and try again.")
            else:
                print("There was an issue with your query. Please review the syntax and try again.")
            
            return None
        except SQLAlchemyError as e:  # Catch general SQLAlchemy errors
            print(f"SQL query execution failed: {str(e)}")
            return None
    else:
        st.error("Database not connected")
        return None

def getDatabaseSchema():
    return st.session_state.db.get_table_info() if st.session_state.db else print("Database not connected")
    
def getQuery(question):
    template = """below is the schema of MYSQL database, read the schema carefully about the table and column names. Also take care of table or column name case sensitivity.
    Finally answer user's question in the form of SQL query.

    {schema}

    please only provide the SQL query and nothing else

    for example:
    question: how many albums we have in database
    SQL query: SELECT COUNT(*) FROM album
    question: how many customers are from Brazil in the database ?
    SQL query: SELECT COUNT(*) FROM customer WHERE country=Brazil

    your turn :
    question: {question}
    SQL query :
    please only provide the SQL query and nothing else
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({
        "schema": getDatabaseSchema(),
        "question": question,
    })
    return response.content

def getResponse(question, query, result):
    template2 = """
    Look into the question and the result of the SQL query.
    Finally write a response in natural language by looking into the conversation and result.

    Here are some example for you:
    question: how many albums we have in database?
    SQL query: SELECT COUNT(*) FROM album;
    Result : [(34,)]
    Response: There are 34 albums in the database.

    question: how many users we have in database?
    SQL query: SELECT COUNT(*) FROM customer;
    Result : [(59,)]
    Response: There are 59 amazing users in the database.

    question: how many users above are from india we have in database?
    SQL query: SELECT COUNT(*) FROM customer WHERE country=india;
    Result : [(4,)]
    Response: There are 4 amazing users in the database.

    question: Name of all customer who live in hyderabad?
    SQL query: SELECT name FROM customer WHERE city=hyderabad;
    Result : []
    Response: There are no customers who live in Hyderabad.

    If the question is irrevelant like hi, hello, who are you anything other than sql question then you can ignore the question.

    your turn to write response in natural language from the given result :
    question: {question}
    Result : {result}
    Response:
    """

    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = prompt2 | llm

    response = chain2.invoke({
        "question": question,
        "result": result
    })

    return response.content


st.set_page_config(
    page_icon="🤖",
    page_title="Chat with MYSQL DB",
    layout="centered"
)

st.title("Voice driven MYSQL Database Chatbot 🎙️")

with st.sidebar:
    st.title('Connect to database')
    st.text_input(label="Host", key="host", value="localhost")
    st.text_input(label="Port", key="port", value="3306")
    st.text_input(label="Username", key="username", value="root")
    st.text_input(label="Password", key="password", value="", type="password")
    st.text_input(label="Database", key="database", value="")
    connectBtn = st.button("Connect")
    micBtn = st.button("MIC")

########## ASR part below

if micBtn:
    processor, model = initialize_whisper()
    with st.spinner("Recording for 5 seconds..."):
        question = transcribe_audio(processor, model)
    st.success("Recording complete!")
    st.session_state.chat.append({
        "role": "user",
        "content": question
    })
    query = getQuery(question)
    result = runQuery(query)
    response = getResponse(question, query, result)
    st.session_state.chat.append({
    "role": "assistant",
    "content": response
    })

########## ASR part ended

if connectBtn:
    connectDatabase(
        username=st.session_state.username,
        password=st.session_state.password,
        host=st.session_state.host,
        port=st.session_state.port,
        database=st.session_state.database,
    )
    st.success("Database connected")

question = st.chat_input('Chat with your mysql database')

if "chat" not in st.session_state:
    st.session_state.chat = []

if question:
    if "db" not in st.session_state:
        st.error('Please connect database first.')
    else:
        st.session_state.chat.append({
            "role": "user",
            "content": question
        })

        query = getQuery(question)
        print(query)
        result = runQuery(query)
        response = getResponse(question, query, result)
        st.session_state.chat.append({
            "role": "assistant",
            "content": response
        })


for chat in st.session_state.chat:
    st.chat_message(chat['role']).markdown(chat['content'])