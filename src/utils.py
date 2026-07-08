import json
from pathlib import Path

def getInputJson():
    # Read input/tasks.json (located at repository root `input/tasks.json`)
    tasks_file = Path(__file__).resolve().parents[1] / "input" / "tasks.json"
    with open(tasks_file, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    print(tasks)
    return tasks

def addToOutputJson(output):
    # Write output to output/output.json (located at repository root `output/output.json`)
    output_file = Path(__file__).resolve().parents[1] / "output" / "results.json"
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=4)

def formatOutput(taskId, result):
    return {"task_id": taskId, "result": result}