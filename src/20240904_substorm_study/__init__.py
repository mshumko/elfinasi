from pathlib import Path

# Define the directories to be created
directories = ["data", "plots"]

# Create the directories if they do not exist
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)