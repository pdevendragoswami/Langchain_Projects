from crewai import Task
from tools import yt_tool
from agents import blog_researcher_agent, blog_writer_agent


# defining the task  - Research task
research_task = Task(
    description = (
        "Identify the video {topic}."
        "Get detailed information about the video from the channel."
    ),
    expected_output = "A comprehensive 3 paragraphs long report based on the topic{topic} of the video content",
    tools = [yt_tool],
    agent = blog_researcher_agent
)


# writing task with language model configuration
write_task = Task(
    description = (
        "get the info from youtube channel on the topic {topic}."
    ),
    expected_output = "Summarize the info from the youtube channel video on the topic {topic}",
    tools = [yt_tool],
    agent = blog_writer_agent,
    async_execution = False,     # as our task will be sequential
    output_file  = "{topic}.md"
)