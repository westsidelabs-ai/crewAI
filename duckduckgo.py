import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# You can choose to use a local model through Ollama for example.
#
# from langchain.llms import Ollama
# ollama_llm = Ollama(model="openhermes")

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting
  actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
  # You can pass an optional llm attribute specifying what mode you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic of others (https://python.langchain.com/docs/integrations/llms/)
  #
  # Examples:
  # llm=ollama_llm # was defined above in the file
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
)

writer = Agent(
  role='Venture Capitalist Strategist',
  goal='Craft assessments in tech advancements, particularly for AI in Retail.',
  backstory="""You are a renowned VC Strategist, known for
  your insightful and engaging assessments.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  # (optional) llm=ollama_llm
)
ceo_agent = Agent(
  role='CEO Strategist',
  goal='Define the team needed to form a AI software startup in Retail.',
  backstory="""You are a seasoned CEO Strategist, known for
  your insightful and engaging assessments of personnel, especially related to Software companies.
  You have traits like Steve Jobs, and Sam Altman. Some of the traits are Visionary Thinking, passion and conviction.
  Innoative Mindset, intuition for talent and risk-taking.""",
  verbose=True,
  allow_delegation=True,
  # (optional) llm=ollama_llm
)
finance_agent = Agent(
  role='Financial Guru',
  goal='Define the financial goals for the newly formed AI software startup in Retail.',
  backstory="""You are a seasoned Financial Strategist, known for
  your insightful visionary assessments of financial KPIs such as ROI and TAM. Especially related to Software companies.
  You have traits like Satya Nadella, and Tim Cook. their ability to foresee market trends, adapt to 
  changing technology landscapes, and make strategic decisions that have not only benefited their 
  respective companies but also set trends in the software industry.""",
  verbose=True,
  allow_delegation=True,
  # (optional) llm=ollama_llm
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of what team is needed to create a company that 
  the will identify the latest niche opportunities in retail for AI in 2024. Prepare financial ROI, and KPIs.
  Identify key roles, and the traits of the roles. The team will lead breakthrough in technologies, and 
  potential have AI industry impacts.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, add to the report highlights the 
  team and their roles for the significant AI  niche opportunities in Retail for AI in 2024.
  Your report should be informative yet accessible, catering to a tech-savvy audience.
  Avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full report of at least 2 paragraphs.""",
  agent=writer
)
task3 = Task(
  description="""Using the insights provided, add to the report highlights the 
  team and their roles for the significant AI  niche opportunities in Retail for AI in 2024.
  Your additions should be insightful visionary assessments of financial KPIs such as ROI and TAM. Especially related to Software companies.
  You have traits like Satya Nadella, and Tim Cook.Your report should be informative yet accessible, catering to a tech-savvy audience.
  Avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full report of at least 4 paragraphs.""",
  agent=finance_agent
)
task4 = Task(
  description="""Using the insights provided, add to the report highlights the 
  team and their roles for the significant AI  niche opportunities in Retail for AI in 2024.
  Your additions should be insightful visionary assessments of how the roles this AI startup team should have. 
  You have traits like Steve Jobs, and Sam Altman. Your report should be informative yet accessible, catering to a tech-savvy audience.
  Avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full report of at least 6 paragraphs.""",
  agent=ceo_agent
)
# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer, finance_agent, ceo_agent],
  tasks=[task1, task2, task3, task4],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)