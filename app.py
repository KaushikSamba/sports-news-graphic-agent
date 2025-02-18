from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool

from Gradio_UI import GradioUI


@tool
def get_latest_sports_news_tool(sport: str)-> str: 
    """A tool that retrieves the latest sports news headlines.
    Args:
        sport: the sport to get news about. Allowable options: 'football/nfl' for NFL, 'tennis/wta' for Women's Tennis, 'tennis/atp' for Men's Tennis.
    """
    
    # Allowable endpoints are:
    # "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
    # "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/news"
    # "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/news"
    # Can I count on the LLM to adhere to the instructions in my documentation? 
    endpoint = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/news"
    response = requests.get(endpoint)
    response = response.json()
    return response.get("articles", [])


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        image_generation_tool, 
        get_current_time_in_timezone,
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        get_latest_sports_news_tool,
        final_answer,
    ], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()