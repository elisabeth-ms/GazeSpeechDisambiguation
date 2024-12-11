import matplotlib.pyplot as plt
import random

from typing import List, Tuple, Optional, Dict, Any


def merge_gaze_word_intervals(
        gaze_data: List[Tuple[List[str], float, float]],
        word_data: List[Tuple[str, float, float]]
    ) -> Dict[str, Any]:
    """Merges gaze data and word data into a single structure with aligned time intervals.
    Parameters:
        gaze_data (List[Tuple[List[str], float, float]]): List of gaze data entries, each containing a list of objects, start time, and end time.
        word_data (List[Tuple[str, float, float]]): List of word data entries, each containing a word, start time, and end time.
    Returns:
        Dict[str, Any]: A dictionary containing the merged rows with time intervals, current word, and current gaze objects.
    """

    boundaries = sorted(set(
        [start for _, start, _ in gaze_data] +
        [end for _, _, end in gaze_data] +
        [start for _, start, _ in word_data] +
        [end for _, _, end in word_data]
    ))

    merged_rows = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        time_str = f"{start:.3f}-{end:.3f}"

        current_word = None
        for word, word_start, word_end in word_data:
            if word_start <= start < word_end:
                current_word = word
                break 
        
        current_gaze_objects = []
        for gaze_objects, gaze_start, gaze_end in gaze_data:
            if gaze_start <= start < gaze_end:
                current_gaze_objects = gaze_objects
                break  

        merged_rows.append([time_str, current_word, [current_gaze_objects] if current_gaze_objects else []])
    
    
    result_rows = []
    start_time_segment = None
    gazed_objects_segment = []
    new_segment = True
    for i in range(len(merged_rows)):
        if i < len(merged_rows) - 1:
            if merged_rows[i][1] == merged_rows[i+1][1] and merged_rows[i][1] is not None:
                time_range = merged_rows[i][0]
                start_str, end_str = time_range.split('-')
                start_time = float(start_str)
                end_time = float(end_str)
                
                if new_segment:
                    start_time_segment = start_time
                    new_segment = False
                gazed_objects_segment.append(f"{merged_rows[i][2]}, ({start_time:.3f}-{end_time:.3f})")

            else:
                if gazed_objects_segment:
                    time_range = merged_rows[i][0]
                    start_str, end_str = time_range.split('-')
                    start_time = float(start_str)
                    end_time = float(end_str)
                    gazed_objects_segment.append(f"{merged_rows[i][2]}, ({start_time:.3f}-{end_time:.3f})")

                    result_rows.append([f"Time: {start_time_segment:.3f}-{end_time:.3f}", f"Word: {merged_rows[i][1]}", f"Gazed objects: {gazed_objects_segment[0]}"])
                    gazed_objects_segment = []
                    new_segment = True
                else:
                    if merged_rows[i][1] is not None or merged_rows[i][2] != []:
                        time_range = merged_rows[i][0]
                        start_str, end_str = time_range.split('-')
                        start_time = float(start_str)
                        end_time = float(end_str)
                        result_rows.append([f"Time: {start_time:.3f}-{end_time:.3f}", f"Word: {merged_rows[i][1]}", f"Gazed objects: {merged_rows[i][2]}"])
                    new_segment = True
                    gazed_objects_segment = []
        else:
            
            if gazed_objects_segment:
                time_range = merged_rows[i][0]
                start_str, end_str = time_range.split('-')
                start_time = float(start_str)
                end_time = float(end_str)
                result_rows.append([f"Time: {start_time_segment:.3f}-{end_time:.3f}", f"Word: {merged_rows[i][1]}", f"Gazed objects: {gazed_objects_segment[0]}"])
            else:
                time_range = merged_rows[i][0]
                start_str, end_str = time_range.split('-')
                start_time = float(start_str)
                end_time = float(end_str)
                result_rows.append([f"Time: {start_time:.3f}-{end_time:.3f}", f"Word: {merged_rows[i][1]}", f"Gazed objects: {merged_rows[i][2]}"])

        
    return {
        "rows": result_rows
    }


def compute_gaze_history_closest_object(gaze_data, start_time, gaze_velocity_threshold=20.0, angle_diff_threshold=15.0,
                         angle_diff_xz_threshold=5.0, excluded_objects=[], off_target_velocity_threshold=5.0,
                         off_target_duration_threshold=0.5, minimum_fixation_duration=0.08):
    """
    Computes the gaze history by determining the closest object the user is looking at based on angle differences.
    Parameters:
    
        gaze_data (list): List of gaze data entries, each entry contains the time,head direction velocity, and a list of objects with angle differences.
        start_time (float): The time when the gaze tracking started.
        gaze_velocity_threshold (float): Maximum allowed gaze velocity to consider the gaze stable (default is 20.0).
        angle_diff_threshold (float): The maximum allowed angle difference in 3D (default is 15.0 degrees).
        angle_diff_xz_threshold (float): The maximum allowed angle difference in the vertical plane (default is 5.0 degrees).
        excluded_objects (list): List of objects that should be ignored in the gaze history.
        off_target_velocity_threshold (float): Gaze velocity threshold to determine off-target fixation (default is 5.0).
        off_target_duration_threshold (float): Minimum duration to consider an off-target gaze (default is 0.5 seconds).
        minimum_fixation_duration (float): Minimum duration to consider a gaze fixation (default is 0.08 seconds).

    Returns:
        Tuple[list, list]: Gaze history and Object timestamps. 
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
                    if gaze_duration > minimum_fixation_duration:
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
            
            if obj['angle_diff'] < angle_diff_threshold and obj['angle_diffXZ'] < angle_diff_xz_threshold and obj['name'] not in excluded_objects:
                object_name = obj['name']
                break  # Found a valid object, stop searching
        # print(f"Object: {object_name}")
        
        if object_name == 'camera':
            object_name = 'Johnnie'

        """ Remove off-target gaze because it generates a lot of confusion for the gpt agent"""
        # If no valid object was chosen, but the gaze velocity is below the off-target threshold, mark it as off-target
        # if object_name is None and entry['gaze_velocity'] < off_target_velocity_threshold:
        #     object_name = 'off-target gaze'

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
                if gaze_duration > minimum_fixation_duration:
                    gaze_history.append((current_object, gaze_duration))
                    objects_timestamps.append(
                        (current_object, gaze_start_time, current_time))

            # Switch to the new object and update the start time
            current_object = object_name
            gaze_start_time = current_time  

    # Handle the last object gazed at (after the loop ends)
    if current_object is not None and gaze_start_time is not None:
        gaze_duration = current_time - gaze_start_time
        if gaze_duration > minimum_fixation_duration:
            gaze_history.append((current_object, gaze_duration))
            objects_timestamps.append(
                (current_object, gaze_start_time, current_time))

    
    # Return the gaze history and object timestamps
    return gaze_history, objects_timestamps


def compute_list_closest_objects_gaze_history(gaze_data, start_time, gaze_velocity_threshold=20.0, angle_diff_threshold=15.0,
                         angle_diff_xz_threshold=5.0, excluded_objects=[], off_target_velocity_threshold=5.0,
                         off_target_duration_threshold=0.5, minimum_fixation_duration = 0.05, end_time=None):
    
    """
    Computes a detailed gaze history, including a list of closest objects at each segment.

    Parameters:
        gaze_data (list): List of gaze entries with time, velocity, and objects.
        start_time (float): Starting time of gaze tracking.
        end_time (float): End time to stop processing data.
        gaze_velocity_threshold (float): Maximum allowed gaze velocity to consider the gaze stable (default is 20.0).
        angle_diff_threshold (float): The maximum allowed angle difference in 3D (default is 15.0 degrees).
        angle_diff_xz_threshold (float): The maximum allowed angle difference in the vertical plane (default is 5.0 degrees).
        excluded_objects (list): List of objects that should be ignored in the gaze history.
        off_target_velocity_threshold (float): Gaze velocity threshold to determine off-target fixation (default is 5.0).
        off_target_duration_threshold (float): Minimum duration to consider an off-target gaze (default is 0.5 seconds).
        minimum_fixation_duration (float): Minimum duration to consider a gaze fixation (default is 0.08 seconds).
    
    Returns:
        Tuple[list, list]: Filtered gaze history and updated object timestamps.
    """
    # List to store gaze history and object timestamps
    
    gaze_history = []
    objects_timestamps = []  
    
    # Variables to track the current object and gaze start time of the current object
    current_objects = []
    gaze_start_time = None
    angle_diff_data = {}  # Dictionary to store angle differences for each object in a segment

    # Loop through the gaze data entries
    for entry in gaze_data:
        
        if end_time:
            if entry['time'] > end_time:
                continue
        
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
                
                if current_objects == ['off-target gaze'] and gaze_duration < off_target_duration_threshold:
                    # Ignore short off-target gaze segments
                    pass
                elif not current_objects:
                    pass
                else:
                    if current_objects == ['off-target gaze']:
                        sorted_objects = current_objects
                    else:
                        
                        # Order by average angle diff 
                        sorted_objects = sorted(current_objects, key=lambda obj: sum(angle_diff_data[obj]) / len(angle_diff_data[obj]))
                    #Store in history and timestamps
                    # if gaze_duration > minimum_fixation_duration:
                    gaze_history.append((sorted_objects.copy(), round(gaze_duration,3)))
                    objects_timestamps.append(
                            (sorted_objects.copy(), gaze_start_time, current_time))
                    
                # Reset current object since we are no longer tracking a stable gaze
                current_objects = []
                angle_diff_data = {}
            continue


        
        valid_objects = []
        for obj in entry['objects']:
            if obj['name'] in excluded_objects:
                continue  # Skip excluded objects

            
            if obj['angle_diff'] < angle_diff_threshold and obj['angle_diffXZ'] < angle_diff_xz_threshold and obj['name'] not in excluded_objects:
                valid_objects.append(obj['name'])
                
                if obj['name'] not in angle_diff_data:
                    angle_diff_data[obj['name']] = []
                angle_diff_data[obj['name']].append(obj['angle_diff'])

        # If no valid object was chosen, but the gaze velocity is below the off-target threshold, mark it as off-target
        # if not valid_objects and entry['gaze_velocity'] < off_target_velocity_threshold:
        #     valid_objects = ['off-target gaze']

        # Initialize the first object and gaze start time
        if not current_objects and not gaze_start_time:
            current_objects = valid_objects.copy()
            gaze_start_time = current_time
            continue  # Skip to the next iteration since we just initialized

        # If the current object being gazed at changes, finalize the previous gaze segment
        if set(valid_objects) != set(current_objects):
            # print(f"Current Objects: {current_objects}, Valid Objects: {valid_objects}")
            # Compute the time spent on the previous object
            gaze_duration = current_time - gaze_start_time

            if current_objects == ['off-target gaze'] and gaze_duration < off_target_duration_threshold:
                # Ignore short off-target gaze segments
                pass
            elif not current_objects:
                pass
            else:
                if current_objects == ['off-target gaze']:
                    sorted_objects = current_objects
                else:
                    # Order by average angle diff 
                    sorted_objects = sorted(current_objects, key=lambda obj: sum(angle_diff_data[obj]) / len(angle_diff_data[obj]))
                # Store in history and timestamps
                #if gaze_duration > minimum_fixation_duration:
                gaze_history.append((sorted_objects.copy(), round(gaze_duration, 3)))
                objects_timestamps.append((sorted_objects.copy(), gaze_start_time, current_time))

            # Switch to the new object and update the start time
            current_objects = valid_objects.copy()
            gaze_start_time = current_time
            angle_diff_data = {obj: [entry['objects'][i]['angle_diff']] for i, obj in enumerate(valid_objects)}

    # Handle the last object gazed at (after the loop ends)
    if current_objects:
        gaze_duration = current_time - gaze_start_time
        if current_objects == ['off-target gaze']:
            sorted_objects = current_objects
        else:
            sorted_objects = sorted(current_objects, key=lambda obj: sum(angle_diff_data[obj]) / len(angle_diff_data[obj]))
            
        #if gaze_duration > minimum_fixation_duration:
        gaze_history.append((sorted_objects.copy(), round(gaze_duration, 3)))
        objects_timestamps.append((sorted_objects.copy(), gaze_start_time, current_time))
    
    # Now lets check tje gaze history and objects timestamps if we have concurrent gazed objects with short duration we should joint them. 
    # If they arent the same group of objects and the duration is less than minimum_fixation_duration we should remove them
    
    new_objects_timestamps = []
    new_gaze_history = []
    segment_start = True
    segment_start_time = None
    for i in range(len(objects_timestamps)):
        if i < len(objects_timestamps) - 1:
            current_objects, current_start, current_end = objects_timestamps[i]
            next_objects, next_start, next_end = objects_timestamps[i+1]
            if segment_start:
                segment_start = False
                segment_start_time = current_start
            if current_objects == next_objects:
                if next_start - current_end < minimum_fixation_duration:
                    continue
                else:
                    for i in range(len(current_objects)):
                        if current_objects[i] == 'camera':
                            current_objects[i] = 'Johnnie'
                    new_objects_timestamps.append((current_objects, segment_start_time, current_end))
                    new_gaze_history.append((current_objects, round(current_end - segment_start_time, 3)))
                    segment_start = True
            else:
                if current_end - segment_start_time < minimum_fixation_duration:
                    segment_start = True
                else:
                    for i in range(len(current_objects)):
                        if current_objects[i] == 'camera':
                            current_objects[i] = 'Johnnie'
                    new_objects_timestamps.append((current_objects, segment_start_time, current_end))
                    new_gaze_history.append((current_objects, round(current_end - segment_start_time,3)))
                    segment_start = True
        else:
            current_objects, current_start, current_end = objects_timestamps[i]
            if segment_start:
                segment_start = False
                segment_start_time = current_start
            if current_end - segment_start_time < minimum_fixation_duration:
                continue
            else:
                for i in range(len(current_objects)):
                    if current_objects[i] == 'camera':
                        current_objects[i] = 'Johnnie'
                new_objects_timestamps.append((current_objects, segment_start_time, current_end))
                new_gaze_history.append((current_objects, round(current_end - segment_start_time,3)))
    

    # Return the gaze history and object timestamps
    return new_gaze_history, new_objects_timestamps


def compute_multi_object_gaze_history(gaze_data, start_time, threshold_angle=60.0, max_average_angle_diff=45.0):
    
    """
    Computes gaze history for multiple objects based on angle differences over time.

    Parameters:
        gaze_data (list): List of gaze entries with objects and angle differences.
        start_time (float): Starting time for gaze tracking.
        threshold_angle (float): Default angle assigned to missing objects.
        max_average_angle_diff (float): Maximum average angle difference to include an object in the result.

    Returns:
        list: Gaze history with durations, main objects, and filtered angle differences.
    """
    
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
    """
    Filters out specific objects from the gaze history.

    Parameters:
        gaze_history (list): List of gaze history entries with time spent and objects.
        excluded_objects (list): List of object names to exclude.

    Returns:
        list: Filtered gaze history.
    """
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


def plot_angle_diff_over_time(gaze_data, start_time=0.0, end_time=None, angle_diff_mode='3D'):
    """
    Plots angle differences over time for objects in the gaze data.

    Parameters:
        gaze_data (list): Gaze data with angle differences for objects.
        start_time (float): Start time for the plot.
        end_time (Optional[float]): End time for the plot.
        angle_diff_mode (str): Angle difference mode ('3D', 'XZ', 'XY').

    Returns:
        None
    """
    # Extract all unique objects from gaze data
    unique_objects = set()
    # print(gaze_data)
    for entry in gaze_data:
        for obj in entry['objects']:
            unique_objects.add(obj['name'])
    
    
    if angle_diff_mode == '3D':
        selected_angle_diff = 'angle_diff'
        selected_angle_diffs = 'angle_diffs'
    elif angle_diff_mode == 'XZ':
        selected_angle_diff = 'angle_diffXZ'
        selected_angle_diffs = 'angle_diffsXZ'
    elif angle_diff_mode == 'XY':
        selected_angle_diff = 'angle_diffXY'
        selected_angle_diffs = 'angle_diffsXY'
    else:
        raise ValueError(f"Invalid angle_diff_mode: {angle_diff_mode}. Position must be one of '3D', 'XZ', or 'XY'")
    # Initialize data dictionary for each object
    object_data = {obj: {'times': [], selected_angle_diffs: []} for obj in unique_objects}
    # print(object_data)
    # Loop through the gaze data and fill in time and angleDiff values for each object
    for entry in gaze_data:
        current_time = entry['time']
        # print(current_time)
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
    # print(object_data)

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
    """
    Plots gaze velocity over time.

    Parameters:
        gaze_data (list): Gaze data with velocity and time information.
        start_time (float): Start time for the plot.
        end_time (float): End time for the plot.

    Returns:
        None
    """
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
    """
    Plots gaze and speech data on separate subplots.

    Parameters:
        gazed_objects_timestamps (list): Timestamps of gazed objects.
        words_data (list): Timestamps of spoken words.

    Returns:
        None
    """
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
    
def plot_multi_gaze_and_speech(gazed_objects_timestamps, words_data, start_time_limit=None, end_time_limit=None):
    """
    Plots gaze data for multiple objects and speech data on the same timeline.

    Parameters:
        gazed_objects_timestamps (list): Timestamps for multiple gazed objects.
        words_data (list): Timestamps of spoken words.
        start_time_limit (float): Start time for the plot.
        end_time_limit (float): End time for the plot.

    Returns:
        None
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Two subplots, sharing the same x-axis

    # print(gazed_objects_timestamps)
    # print(words_data)
    # Extract unique objects for the y-axis
    unique_objects = list(set([obj for objs, _, _ in gazed_objects_timestamps for obj in objs]))
    object_mapping = {obj: i for i, obj in enumerate(unique_objects)}

    # Plot Gaze Data (Top subplot)
    for objs, start_time, end_time in gazed_objects_timestamps:
        for obj in objs:  # Now we loop over multiple objects in each segment
            ax1.hlines(y=object_mapping[obj], xmin=start_time, xmax=end_time, label=obj, linewidth=5)

    # Set x-axis limits if specified
    if start_time_limit is not None and end_time_limit is not None:
        ax1.set_xlim(start_time_limit, end_time_limit)

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

    if start_time_limit is not None and end_time_limit is not None:
        ax2.set_xlim(start_time_limit, end_time_limit)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speech')
    ax2.set_title('Speech Word-level Timestamps')
    ax2.grid(True)

    plt.tight_layout()

