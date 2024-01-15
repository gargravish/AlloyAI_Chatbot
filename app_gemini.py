#!/usr/bin/env python
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit_lottie import st_lottie
from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import VertexAI
import ast
import re
import pandas as pd
import requests
import json


barchart = None
table_data = None
chat = ChatVertexAI()
db = SQLDatabase.from_uri(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",)
system = """
        Given an input question, create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        If someone asks for the database version do a select version().

        For the following query, if it requires drawing a table, reply as follows:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        If the query requires creating a bar chart, reply as follows:
        {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        If the query requires creating a line chart, reply as follows:
        {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        There can only be two types of chart, "bar" and "line".

        If it is just asking a question that requires neither, reply as follows:
        {"answer": "The response from the database is: 'answer'"}
        Example:
        {"answer": "The title with the highest rating is 'Gilead'"}

        If you do not know the answer, reply as follows:
        {"answer": "I do not know."}

        Return all output as a string.

        All strings in "columns" list and data list, should be in double quotes,

        For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

        Lets think step by step.

        Below is the query.
        Query: 
        """


llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, convert_system_message_to_human=True,google_api_key="{GOOGLE_API_KEY}")

toolkit = SQLDatabaseToolkit(db=db,llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    handle_parsing_errors=True,
)

def load_lottieurl(url:str):
    r = requests.get(url)
    url_json = dict()
    if r.status_code == 200:
        url_json = r.json()
        return url_json
    else:
        None

def gen_response(human):
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system), HumanMessage(content=human)])
    messages = prompt.format_messages()
    response = agent_executor.run(messages)
    print("\n\n")
    print(response)
    print("\n\n Stripping the JSON from response....\n\n")
    #strip_resp = re.findall(r'\{.*?\}',response.content)
    strip_resp = response
    if strip_resp:
        print(strip_resp)
        decoded_response = decode_response(strip_resp)
        result,data_type = write_response(decoded_response)
        print(result)
        print(type(result))
        return result,data_type
    else:
        return response

def decode_response(response: str) -> dict:
    json_conv = None
    json_conv = ast.literal_eval(response)
    print("Response in JSON: \n",json_conv)
    return json_conv

def write_response(response_dict: dict):
    df = None
    # Check if the response is an answer.
    if "answer" in response_dict:
        emstrng = None
        return response_dict["answer"],emstrng

    if "table" in response_dict:
        table_data = "table"
        data = response_dict["table"]
        print(data)
        df = pd.DataFrame(data["data"], columns=data["columns"])
        print(df)
        return df,table_data

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        barchart ="bar"
        data = dict(response_dict["bar"])
        print("Dict: \n",response_dict["bar"])
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df.set_index(df.columns[0],inplace=True)
        print("\n\nBar Graph dataframe: \n",df)
        return df,barchart

load_bot = load_lottieurl("https://lottie.host/02dc354a-6772-4696-9a72-5de200583aaa/LnRctYz4gl.json")
col1,col2 = st.columns(2)
with col1:
    st.markdown('Managed by: <a href="mailto:kirantailor@google.com">Kiran Tailor</a>, <a href="mailto:ravishgarg@google.com">Ravish Garg</a>',unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')
    st.title(":violet[Alloy - AI Bot]")
with col2:
    st_lottie(load_bot,height=200,width=200,speed=1,loop=True,quality='high',key='bot')


# Set a default model
if "vertexai_model" not in st.session_state:
    st.session_state["vertexai_model"] = "chat-bison"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if human := st.chat_input("Which database version is in use?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": human})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(human)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Processing..."):
            full_response,data_type = gen_response(human)
        if isinstance(full_response,pd.DataFrame):
            if data_type == 'table':
                st.table(full_response)
            elif data_type == 'bar':
                st.bar_chart(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
