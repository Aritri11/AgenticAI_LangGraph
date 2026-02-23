from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_classic import hub
from dotenv import load_dotenv
import os

load_dotenv()
#setting up project name inside the code (method 2)
os.environ['LANGCHAIN_PROJECT'] = 'Agents'

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=6f4fc701f46b9b40232531d7faa4bed7&query={city}'

  response = requests.get(url)

  return response.json()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react").template  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_agent(
    model=model,
    tools=[search_tool,get_weather_data],
    system_prompt=prompt
)

# Step 4: Invoke
response = agent.invoke({"input": "What is the current temperature of gurgaon ?"})
print(response)

print(response["messages"][-1].content)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.



