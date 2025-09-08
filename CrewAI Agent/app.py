from crewai import Crew, Process
from agents import blog_writer_agent, blog_researcher_agent
from tools import yt_tool
from tasks import research_task, write_task


# forming the tech-focused crew with some enhanced configuration
crew = Crew(
    agents = [blog_researcher_agent, blog_writer_agent],
    tasks = [research_task, write_task],
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm = 100,
    share_crew = True
)



# start the task execution process with enhanced feedback
result = crew.kickoff({"topic":"AI vs ML vs DL vs Data Science"})
print(result)