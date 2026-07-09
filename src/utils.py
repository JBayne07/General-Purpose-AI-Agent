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
  # Append output to output/results.json (located at repository root `output/results.json`)
  output_file = Path(__file__).resolve().parents[1] / "output" / "results.json"
  output_file.parent.mkdir(parents=True, exist_ok=True)

  if output_file.exists():
      with open(output_file, "r", encoding="utf-8") as fh:
          try:
              existing_output = json.load(fh)
          except json.JSONDecodeError:
              existing_output = []

      if isinstance(existing_output, list):
          existing_output.append(output)
      else:
          existing_output = [existing_output, output]
  else:
      existing_output = [output]

  with open(output_file, "w", encoding="utf-8") as fh:
      json.dump(existing_output, fh, indent=4)

def formatOutput(taskId, result):
    return {"task_id": taskId, "result": result}