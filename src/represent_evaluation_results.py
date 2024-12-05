import matplotlib.pyplot as plt

# Data for the table
data_breakfast = [
    ["Task 1", 186, 166],
    ["Task 2", 216, 208],
    ["Task 3", 120, 90]
    ]
data_drink = [
    ["Task 1", 180, 178],
    ["Task 2", 216, 205],
    ["Task 3", 170, 117],
]

number_executions = 216
columns = ["Task", "Async Gaze + Speech + Scene", "Async Gaze + Speech"]
row_labels = ["Breakfast", None, None, "Drink", None, None]  # Merge simulation

import matplotlib.pyplot as plt
import numpy as np

tasks = ("Task 1", "Task 2", "Task 3")

representions_percentage = {
    'Async Gaze + Speech + Scene': (round(data_breakfast[0][1]/number_executions*100.0,2), round(data_breakfast[1][1]/number_executions*100.0,2), round(data_breakfast[2][1]/number_executions*100.0,2)),
    'Async Gaze + Speech': (round(data_breakfast[0][2]/number_executions*100.0,2), round(data_breakfast[1][2]/number_executions*100.0,2), round(data_breakfast[2][2]/number_executions*100.0,2)),
}
x = np.arange(len(tasks))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in representions_percentage.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('% Correct')
ax.set_title('Breakfast scenario', pad=30)
ax.set_xticks(x + width, tasks)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)



tasks = ("Task 1", "Task 2", "Task 3")
representions_percentage = {
    'Async Gaze + Speech + Scene': (round(data_drink[0][1]/number_executions*100.0,2), round(data_drink[1][1]/number_executions*100.0,2), round(data_drink[2][1]/number_executions*100.0,2)),
    'Async Gaze + Speech': (round(data_drink[0][2]/number_executions*100.0,2), round(data_drink[1][2]/number_executions*100.0,2), round(data_drink[2][2]/number_executions*100.0,2)),
}

x = np.arange(len(tasks))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in representions_percentage.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('% Correct')
ax.set_title('Drinks scenario', pad=30)
ax.set_xticks(x + width, tasks)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)

plt.show()
