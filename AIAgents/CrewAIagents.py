

# Crew AI uses openAI if no tools are specified
# Define Agents

from crewai import Agent,Task, Crew, Process



researcher = Agent(
    role="Senior Research Analyst",
    goal = "Research cutting-edge AI advancements",
    backstory="You work at a tech think tank, analyzing AI trends ",
    verbose=True
)

writer = Agent(
    role="Content Strategist",
    goal = "Create engaging content from research",
    backstory="You are a skilled writer transforming research into engaging articles ",
    verbose=True
)

#Create Tasks
task1 = Task(
    description=(
        "Analyse AI advancements in 2024 and provide a detailed report. "
        "Synthesize from your knowledge (no web browsing). Include: overview, key developments, "
        "notable models/frameworks, pros/cons, and suggested further reading topics."
    ),
    expected_output = "Research report in bullet points",
    agent=researcher
)
task2 = Task(
    description="Write a blog post based on the research report",
    expected_output = "Full blog post(at least 4 paragraphs)",
    agent=writer
)

#Define Crew and Process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)

#Run the process
result = crew.kickoff()
print("Final Output: ")
print(result)



