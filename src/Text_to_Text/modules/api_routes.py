import time
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from .response_generator import ResponseGenerator

app = FastAPI()
response_generator = ResponseGenerator()
mode = "text"

class UserInput(BaseModel):
    input: str

@app.post("/toggle_mode")
async def toggle_mode():
    """Switch between text and voice modes."""
    global mode
    mode = "voice" if mode == "text" else "text"
    return {"mode": mode}

@app.post("/respond")
async def respond_to_input(data: UserInput):
    """Processes user input and returns AI-generated response."""
    user_input = data.input
    ai_response = response_generator.generate_response(user_input)
    return {"response": ai_response}

@app.post("/set_task")
async def set_task(data: UserInput, background_tasks: BackgroundTasks):
    """Schedules tasks asynchronously."""
    task_description = data.input
    background_tasks.add_task(schedule_task, task_description)
    return {"status": "Task scheduled!"}

def schedule_task(task_description):
    """Executes the scheduled task asynchronously."""
    time.sleep(5)  # Simulating delay before reminder
    print(f"🔔 Reminder: {task_description}") 