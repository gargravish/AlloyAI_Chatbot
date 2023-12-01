import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import vertexai
from langchain.agents import AgentType
from langchain.llms import VertexAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.memory import ConversationBufferMemory

# Set the required variables:
PROJECT_ID = {project_id}
LOCATION = {location_id}
DB_USER = {username}
DB_PASSWORD = {password}
DB_HOST = {hostname}
DB_NAME = {db_name}
DB_PORT = {db_port}

#Initialise the vertexai environment
vertexai.init(project=PROJECT_ID, location=LOCATION)
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

#Connecting to AlloyDB
db = SQLDatabase.from_uri(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",)

#Streamlit UI Parameters
st.title(":violet[Allo - AI]")
colored_header(label='', description='', color_name='gray-30')
def on_btn_click():
    del st.session_state.user_responses[:]
    del st.session_state.bot_responses[:]
    st.session_state['user_responses'] = ["Hey"]
    st.session_state['bot_responses'] = ["Hey there! How may I help you today?"]

# Initialize session state variables
if 'user_responses' not in st.session_state:
    st.session_state['user_responses'] = ["Hey"]
if 'bot_responses' not in st.session_state:
    st.session_state['bot_responses'] = ["Hey there! How may I help you today?"]

input_container = st.container()
response_container = st.container()

if 'msg' not in st.session_state:
    st.session_state.msg = ''

def clearoff():
    st.session_state.msg = st.session_state.widget
    st.session_state.widget = ''

# Capture user input and display bot responses
st.text_input("You: ", key='widget',on_change=clearoff)
user_input = st.session_state.msg
print(user_input)

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.user_responses.append(user_input)
        st.session_state.bot_responses.append(response)
        
    if st.session_state['bot_responses']:
        for i in range(len(st.session_state['bot_responses'])):
            message(st.session_state['user_responses'][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Kavita")
            message(st.session_state['bot_responses'][i], key=str(i), avatar_style="initials", seed="AI",)

with input_container:
    st.button("Clear message", on_click=on_btn_click)
    display_input = ""

prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.
If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If someone asks for the database version do a select version().

Use the tables customer and nation to find the number of customers who live in a country by joining the 2 tables together.

Use the table g_columnar_columns with relation_name to see if a table exists in the columnar store.

When asked What columns from the table are in the columnar store, Use the table g_columnar_columns with relation_name as the lookup for the table when asked what columns from this table are in the columnar store.
Use the table g_columnar_columns with relation_name as the table name to get the list of columns in the columnar store.
Use the table g_columnar_columns with last_accessed_time when asking for last accessed for relation.
Use the table g_columnar_columns to get the list of columns in the columnar store.
Use the table Customer with n_name as the key to get the list when asking for country 
Use the table position with id as the primary key to get the position.

If the question does not seem related to the database, just return "I don't know" as the answer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
suffix = """Begin!

{chat_history}
Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""

memory = ConversationBufferMemory(memory_key="chat_history", input_key = "input")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    input_variables = ["input", "agent_scratchpad","chat_history"],
    verbose=True,
    memory=memory,
    return_intermediate_steps=True,
    prefix = prefix,
    suffix = suffix, # must have history as variable,
    agent_executor_kwargs = {'memory':memory}
)
def generate_response(prompt):
    question = prompt
    output = agent_executor.run(question)
    response = translate_text(target,output)
    return response

lang_select = st.selectbox('Select the Target Language: ',('English','German','Spanish','French','Hindi','Swedish','Norwegian'))
if lang_select == "English":
    lang_option='en'
if lang_select == "German":
    lang_option='de'
elif lang_select == "Spanish":
    lang_option='es'
elif lang_select == "French":
    lang_option='fr'
elif lang_select == "Hindi":
    lang_option='hi'
elif lang_select == "Swedish":
    lang_option='sv'
elif lang_select == "Norwegian":
    lang_option='no'

sourcelanguage = 'English'
target = lang_option

def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)
    converted=format(result["translatedText"])
    return converted
