import matplotlib.pyplot as plt
import random
def compute_gaze_history_closest_object(gaze_data, start_time, gaze_velocity_threshold=20.0, angle_diff_threshold=15.0,
                         angle_diff_xz_threshold=5.0, excluded_objects=[], off_target_velocity_threshold=5.0,
                         off_target_duration_threshold=0.5):
    """
    Computes the gaze history by determining the closest object the user is looking at based on angle differences.

    :param gaze_data: List of gaze data entries, each entry contains the time,head direction velocity, and a list of objects with angle differences.
    :param start_time: The time when the gaze tracking started.
    :param gaze_velocity_threshold: Maximum allowed gaze velocity to consider the gaze stable (default is 20.0).
    :param angle_diff_threshold: The maximum allowed angle difference in 3D (default is 15.0 degrees).
    :param angle_diff_xz_threshold: The maximum allowed angle difference in the vertical plane (default is 5.0 degrees).
    :param excluded_objects: List of objects that should be ignored in the gaze history.
    :param off_target_velocity_threshold: Gaze velocity threshold to determine off-target fixation (default is 5.0).
    :param off_target_duration_threshold: Minimum duration to consider an off-target gaze (default is 0.5 seconds).

    :return: A tuple of two lists - gaze history and object timestamps. 
             Gaze history contains tuples of (object_name, gaze_duration), and object timestamps store (object_name, start_time, end_time).
    """
    # List to store gaze history and object timestamps
    gaze_history = []
    objects_timestamps = []  
    
    # Variables to track the current object and gaze start time of the current object
    current_object = None
    gaze_start_time = None

    # Loop through the gaze data entries
    for entry in gaze_data:
        # Calculate the current time relative to the start time
        current_time = entry['time'] - start_time
        if current_time < 0:
            continue  # Skip entries before the start time
        
        # If gaze veloity is high, exclude the object (unstable gaze)
        if entry['gaze_velocity'] > gaze_velocity_threshold:
            # print(f"High gaze velocity: {entry['gaze_velocity']}, so we exclude object: %s" % entry['objects'][0]['name'])
            
            # If we were previously gazing at an object, end the gaze segment
            if gaze_start_time is not None:
                gaze_duration = current_time - gaze_start_time
                
                if current_object == 'off-target gaze' and gaze_duration < off_target_duration_threshold:
                    # Ignore short off-target gaze segments
                    pass
                elif current_object is None:
                    pass
                else:
                    # Store in the object history and timestamps
                    gaze_history.append((current_object, gaze_duration))
                    objects_timestamps.append(
                        (current_object, gaze_start_time, current_time))
                
                # Reset current object since we are no longer tracking a stable gaze
                current_object = None
            continue

        # print(f"Time: {current_time}")
        # print("Objects: ", entry['objects'])

        # Find the closest object that is not in the excluded list based on angle differences
        
        object_name = None  # Default
        for obj in entry['objects']:
            if obj['name'] in excluded_objects:
                print(f"Excluded object: {obj['name']}")
                continue  # Skip excluded objects
            # print(f"Object: {obj['name']}, Angle Diff: {obj['angleDiff']}, Angle Diff XZ: {obj['angleDiffXZ']}, velocity: {entry['gaze_velocity']}")
            # Check if the object satisfies the angle difference thresholds in both 3D and vertical plane
            
            if obj['angleDiff'] < angle_diff_threshold and obj['angleDiffXZ'] < angle_diff_xz_threshold and obj['name'] not in excluded_objects:
                object_name = obj['name']
                break  # Found a valid object, stop searching
        # print(f"Object: {object_name}")
        
        if object_name == 'camera':
            object_name = 'Johnnie'

        # If no valid object was chosen, but the gaze velocity is below the off-target threshold, mark it as off-target
        if object_name is None and entry['gaze_velocity'] < off_target_velocity_threshold:
            object_name = 'off-target gaze'

        # Initialize the first object and gaze start time
        if current_object is None and gaze_start_time is None:
            current_object = object_name
            gaze_start_time = current_time
            continue  # Skip to the next iteration since we just initialized

        # If the current object being gazed at changes, finalize the previous gaze segment
        if object_name != current_object:
            # Compute the time spent on the previous object
            gaze_duration = current_time - gaze_start_time

            if current_object == 'off-target gaze' and gaze_duration < off_target_duration_threshold:
                # Ignore short off-target gaze segments
                pass
            elif current_object is None:
                pass
            else:
                # Store in the object history and timestamps
                gaze_history.append((current_object, gaze_duration))
                objects_timestamps.append(
                    (current_object, gaze_start_time, current_time))

            # Switch to the new object and update the start time
            current_object = object_name
            gaze_start_time = current_time  

    # Handle the last object gazed at (after the loop ends)
    if current_object is not None and gaze_start_time is not None:
        gaze_duration = current_time - gaze_start_time
        gaze_history.append((current_object, gaze_duration))
        objects_timestamps.append(
            (current_object, gaze_start_time, current_time))

    # Return the gaze history and object timestamps
    return gaze_history, objects_timestamps


def compute_multi_object_gaze_history(gaze_data, start_time, threshold_angle=60.0, max_average_angle_diff=45.0):
    gaze_history = []
    current_object = None
    current_segment_start = None
    current_angle_diffs = {}

    for entry in gaze_data:
        current_time = entry['time'] - start_time
        if current_time < 0:
            continue  # Skip entries before the start time

        # Collect angle diffs for all available objects in this gaze entry
        angle_diffs = {obj['name']: obj['angleDiff'] for obj in entry['objects']}
        all_objects_in_frame = set(angle_diffs.keys())

        # Fill missing objects with threshold_angle
        all_objects = set(obj['name'] for gaze_entry in gaze_data for obj in gaze_entry['objects'])
        for obj in all_objects:
            if obj not in angle_diffs:
                angle_diffs[obj] = threshold_angle  # Assign threshold angle for missing objects

        # Get the first object (smallest angleDiff) in this frame
        if entry['objects']:
            main_object = entry['objects'][0]['name']
            main_angle_diff = entry['objects'][0]['angleDiff']
        else:
            main_object = 'off-target gaze'
            main_angle_diff = None

        # Start a new segment if the object changes
        if main_object != current_object:
            if current_object is not None:
                # Compute segment duration and filter objects with average angleDiff > max_average_angle_diff
                segment_duration = current_time - current_segment_start
                filtered_angle_diffs = {obj: avg_angle for obj, avg_angle in current_angle_diffs.items() if avg_angle <= max_average_angle_diff and obj != current_object}
                
                if filtered_angle_diffs:
                    gaze_history.append((segment_duration, {current_object: current_angle_diffs[current_object]}, filtered_angle_diffs))
            
            # Start new segment
            current_object = main_object
            current_segment_start = current_time
            current_angle_diffs = angle_diffs
        else:
            # Update angle diffs if the object remains the same
            current_angle_diffs = {obj: (current_angle_diffs[obj] + angle_diffs[obj]) / 2 for obj in current_angle_diffs}

    # Handle the final segment
    if current_object is not None:
        segment_duration = gaze_data[-1]['time'] - current_segment_start
        filtered_angle_diffs = {obj: avg_angle for obj, avg_angle in current_angle_diffs.items() if avg_angle <= max_average_angle_diff and obj != current_object}
        
        if filtered_angle_diffs:
            gaze_history.append((segment_duration, {current_object: current_angle_diffs[current_object]}, filtered_angle_diffs))

    return gaze_history



def filter_multi_object_gaze_history(gaze_history, excluded_objects):
    filtered_gaze_history = []

    for entry in gaze_history:
        time_spent, main_object_dict, filtered_objects = entry

        # Check if main_object is not one of the robot hands
        main_object_name = list(main_object_dict.keys())[0]
        if main_object_name in excluded_objects:
            continue  # Skip this entry

        # Filter out robot hands from filtered_objects
        filtered_objects = {obj: angle for obj, angle in filtered_objects.items() if obj not in excluded_objects}
        
        # Only add the entry if valid objects remain
        if filtered_objects or main_object_name not in excluded_objects:
            filtered_gaze_history.append((time_spent, main_object_dict, filtered_objects))

    return filtered_gaze_history


def plot_angle_diff_over_time(gaze_data, start_time=0, end_time=None, angle_diff_mode='3D'):
    # Extract all unique objects from gaze data
    unique_objects = set()
    for entry in gaze_data:
        for obj in entry['objects']:
            unique_objects.add(obj['name'])
    
    
    if angle_diff_mode == '3D':
        selected_angle_diff = 'angleDiff'
        selected_angle_diffs = 'angleDiffs'
    elif angle_diff_mode == 'XZ':
        selected_angle_diff = 'angleDiffXZ'
        selected_angle_diffs = 'angleDiffsXZ'
    elif angle_diff_mode == 'XY':
        selected_angle_diff = 'angleDiffXY'
        selected_angle_diffs = 'angleDiffsXY'
    else:
        raise ValueError(f"Invalid angle_diff_mode: {angle_diff_mode}. Position must be one of '3D', 'XZ', or 'XY'")
    # Initialize data dictionary for each object
    object_data = {obj: {'times': [], selected_angle_diffs: []} for obj in unique_objects}
    
    # Loop through the gaze data and fill in time and angleDiff values for each object
    for entry in gaze_data:
        current_time = entry['time']
        if current_time >= start_time and current_time <= end_time:
            objects_in_frame = {obj_data['name']: obj_data[selected_angle_diff] for obj_data in entry['objects']}
            
            for obj in unique_objects:
                if obj in objects_in_frame:
                    object_data[obj]['times'].append(current_time-start_time)
                    object_data[obj][selected_angle_diffs].append(objects_in_frame[obj])
                else:
                    # If the object is not in the frame, add None to leave a gap
                    object_data[obj]['times'].append(current_time-start_time)
                    object_data[obj][selected_angle_diffs].append(None)

    # Plot all objects' angleDiffs on the same graph with different colors
    plt.figure(figsize=(10, 6))
    for obj, data in object_data.items():
        plt.plot(data['times'], data[selected_angle_diffs], label=obj, marker='o')

    # Customize plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle Difference '+ angle_diff_mode+' (degrees)')
    plt.title('Angle Difference '+ angle_diff_mode+ ' Over Time for Objects')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_gaze_velocity_over_time(gaze_data, start_time=0, end_time=None):
    # Loop through the gaze data and fill in time and angleDiff values for each object
    gaze_times = []
    gaze_velocities = []
    for entry in gaze_data:
        current_time = entry['time']
        if current_time >= start_time and current_time <= end_time:
            current_gaze_velocity = entry['gaze_velocity']
            gaze_times.append(current_time-start_time)
            gaze_velocities.append(current_gaze_velocity)

    # Plot all objects' angleDiffs on the same graph with different colors
    plt.figure(figsize=(10, 6))
    plt.plot(gaze_times, gaze_velocities, marker='o')

    # Customize plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle velocities')
    plt.title('Gaze angle velocities Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
def plot_gaze_and_speech(gazed_objects_timestamps, words_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Two subplots, sharing the same x-axis

    # Extract unique objects for the y-axis
    unique_objects = list(set([obj for obj, start, end in gazed_objects_timestamps]))
    object_mapping = {obj: i for i, obj in enumerate(unique_objects)}

    # Plot Gaze Data (Top subplot)
    for obj, start_time, end_time in gazed_objects_timestamps:
        ax1.hlines(y=object_mapping[obj], xmin=start_time, xmax=end_time, label=obj, linewidth=5)

    ax1.set_yticks(range(len(unique_objects)))
    ax1.set_yticklabels(unique_objects)
    ax1.set_ylabel('Gaze Objects')
    ax1.set_title('Gaze Data Over Time')
    ax1.grid(True)
    # Plot Speech Data (Bottom subplot)

    for i, (word, start_time, end_time) in enumerate(words_data):
        print(f"Word: {word}, Start: {start_time}, End: {end_time}")
        ax2.hlines(y=0, xmin=start_time, xmax=end_time, color=(random.random(), random.random(), random.random()), linewidth=6)
        mid_time = (start_time + end_time) / 2
        ax2.text(mid_time, 0, word, ha='center', va='center', fontsize=12, color='black')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speech')
    ax2.set_title('Speech Word-level Timestamps')
    ax2.grid(True)

    plt.tight_layout()
