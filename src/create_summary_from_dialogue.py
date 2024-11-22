import os
import json
import re
import shutil

import matplotlib.pyplot as plt
import pickle

add_system_prompt = True
user_input_count = 1
# Function to sort naturally (e.g., "dialogue43_2" before "dialogue43_10")
def natural_sort_key(text):
    return [int(num) if num.isdigit() else num for num in re.split(r'(\d+)', text)]

# Function to extract speech input from user content
def extract_speech_input(content):
    match = re.search(r"Speech input:\s*(.*?)\s*Gaze history", content)
    return match.group(1) if match else "Speech input not found"

def extract_gaze_history(content):
    match = re.search(r"Gaze history \(in seconds\):\s*(\[.*\])", content)
    return match.group(1) if match else "Gaze history not found"

def generate_summary(root_dir, new_dialogue_number, new_dir):
    
    # Check if a folder exists
    dialogue_dir = os.path.join(new_dir, f"dialogue{new_dialogue_number}")
    if os.path.exists(dialogue_dir):
        print(f"Error Figures directory dialogue{dialogue_dir} already exists!!!")
        return
    else:
        os.makedirs(dialogue_dir, exist_ok=True)
    
    
    markdown_summary = f"# Dialogue {new_dialogue_number}:\n\n"
    previous_interaction_responses = 0
    sorted_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))], key=natural_sort_key)
    # Loop through each dialogue folder
    count_interactions = 0
    


        
    for folder in sorted_folders:
        print("folder: ", folder)
        count_interactions +=1
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Load JSON data from interaction_data.json
            json_path = os.path.join(folder_path, "interaction_data.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)

                
                # Add folder name as a section in Markdown
                markdown_summary += f"## Dialogue {new_dialogue_number}.{count_interactions}\n\n"
                
                print(data)
                

                # Parse interaction responses
                for index,entry in enumerate(data.get("interaction_responses", [])):
                    if index>=previous_interaction_responses:
                        
                        role = entry.get("role")
                        content = entry.get("content")
                        tool_calls = entry.get("tool_calls")

                        global add_system_prompt
                        if add_system_prompt and role == "system":
                            markdown_summary += f"🤖 SYSTEM PROMPT: {content}\n"
                            add_system_prompt = False
                        if role == 'user':
                   
                            speech_input = extract_speech_input(content)
                            markdown_summary += f"🧑‍🎙️  SPEECH INPUT: {speech_input}\n"
                            gaze_input = extract_gaze_history(content)
                            markdown_summary += f"🧑👀 GAZE INPUT: {gaze_input}\n"
                        
                        if tool_calls:
                            for tool_call in tool_calls:
                                if role == 'assistant':
                                    function = tool_call.get("function")
                                    markdown_summary += f"🤖🔧 GPT response is function call: {(function.get('name'))}{(function.get('arguments'))}\n"
                        if role == 'tool':
                            markdown_summary += f"🔧 Function result is: {content}\n"       
                

              
                # Add plot.png image to summary (assuming it exists)
                plot_path = os.path.join(folder_path, "plot.png")
                fig_path = os.path.join(folder_path, "figure.fig")
                with open(fig_path, 'rb') as f:
                    fig = pickle.load(f)
                fig.show()
                new_plot_filename = f"dialogue{new_dialogue_number}_{count_interactions}.png"
                plot_dest_path = os.path.join(dialogue_dir, new_plot_filename)
                shutil.copyfile(plot_path, plot_dest_path)
                if os.path.exists(plot_path):
                    markdown_summary += f"![[{new_plot_filename}]]\n"
                previous_interaction_responses = len(data.get("interaction_responses", []))-1
                print(previous_interaction_responses)
            else:
                markdown_summary += f"## {folder}\n\n- No interaction_data.json found.\n"
                
    return markdown_summary, dialogue_dir

# Usage example
root_dir = "/hri/storage/user/emenende/interaction_recordings/22_11_2024/dialogue19"  # Replace with the path to your folders
new_dialogue_number = 10
new_dir = "/hri/storage/user/emenende/myNotes/dialogues/22_11_2024"
summary, dialogue_dir = generate_summary(root_dir, new_dialogue_number, new_dir)
if summary:
    # Save summary to a Markdown file
    with open(os.path.join(dialogue_dir,"interaction_summary.md"), "w") as f:
        f.write(summary)