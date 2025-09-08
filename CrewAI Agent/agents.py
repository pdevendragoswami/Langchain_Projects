from crewai import Agent
from tools import yt_tool
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

#create a senior blog vontent researcher agent

blog_researcher_agent = Agent(
    role = "Senior Blog Researcher from Youtube Videos",
    goal = "get the relevant video content for the topic {topic} from Youtube Channel",
    verbose = True,
    memory = True,
    backstory = (
        "Expert in understanding videos in AI Data Science, Machine  Learning and Generative AI and providing suggestion."
    ),
    tools = [yt_tool],
    allow_delegation = True  # transfering the data to next one to pass the data to blog writer
)


# create a senior blog writer agent with YT tools

blog_writer_agent = Agent(
    role = "Senior Blog writer",
    goal = "Narrate compelling tech stories about the video {topic}",
    verbose = True,
    memory = True,
    backstory = (
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in accessible manner"
    ),
    tools = [yt_tool],
    allow_delegation = False   # as we will not transfer the data furthermore
)
