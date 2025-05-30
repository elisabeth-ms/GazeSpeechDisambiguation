#!/usr/bin/env python3
#
# Copyright (c) 2024, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import platform
import sys
import yaml

from pathlib import Path
import extra_tools_tiago
from action_tools_llm.msg import ObjectManipulationAction, ObjectManipulationGoal
import actionlib
import rospy
from geometry_msgs.msg import Point
from pal_interaction_msgs.msg import TtsGoal, TtsAction
# System setup

with open(Path(__file__).parent.resolve() / "config.yaml", "r") as config:
    config_data = yaml.safe_load(config)
    SMILE_WS_PATH = Path(__file__).parents[1].resolve() / config_data["install"]
    print(f"{SMILE_WS_PATH=}")

CFG_ROOT_DIR = str(SMILE_WS_PATH / "config")
CFG_DIR = str(SMILE_WS_PATH / "config/xml/examples")
if platform.system() == "Linux":
    sys.path.append(str(SMILE_WS_PATH / "lib"))
elif platform.system() in ("WindowsLocal", "Windows"):
    sys.path.append(str(SMILE_WS_PATH / "bin"))
else:
    sys.exit(platform.system() + " not supported")


from pyAffaction import (
    LlmSim,
    addResourcePath,
    setLogLevel,
)

sceneTransformationDataRecorderEnabled = False
sceneTransformationDataPlayerEnabled = False
addResourcePath(CFG_ROOT_DIR)
addResourcePath(CFG_DIR)
print(f"{CFG_DIR=}")
setLogLevel(-1)

SIMULATION = LlmSim()
SIMULATION.noTextGui = True
SIMULATION.unittest = False
SIMULATION.speedUp = 1
SIMULATION.noLimits = False
SIMULATION.verbose = False
# SIMULATION.usersGazeComponentEnabled = True
# SIMULATION.sceneTransformationDataPlayerEnabled = sceneTransformationDataPlayerEnabled
# SIMULATION.sceneTransformationDataRecorderEnabled = sceneTransformationDataRecorderEnabled
SIMULATION.xmlFileName = "g_example_breakfast_scenario_tiago.xml"
SIMULATION.addLandmarkZmq(camera_name="camera_0", withArucoTracking=True, withSkeletonTracking=True)
SIMULATION.addTTS("native")
SIMULATION.init(True)

# Tools
ARG1 = True

rospy.init_node('tools_tiago')

client = actionlib.SimpleActionClient('object_manipulation', ObjectManipulationAction)
        
client.wait_for_server()

client_tts = actionlib.SimpleActionClient('tts',TtsAction)

## client_tts.wait_for_server()

print("Both clients are ready!")

def query_objects() -> str:
    """
    Query all objects that are available in the scene. You can see all these objects.

    :return: Result message.
    """
    result = SIMULATION.get_objects()
    if not result:
        return "No objects were observed."
    
    objects_no_camera = [obj for obj in result["objects"] if obj != "camera"]
    return "Following objects were observed: " + ", ".join(objects_no_camera) + "."


def query_agents() -> str:
    """
    Query all agents that are available in the scene, including yourself. You can see all these agents.

    :return: Result message.
    """
    result = SIMULATION.get_agents()
    if not result:
        return "No agents were observed."
    return "The following agents, including yourself, were observed: " + ", ".join(result["agents"]) + "."

# def query_all_close_objects(threshold: float = 0.5) -> str:
#     """
#     Query all pairs of objects that are spatially close to each other.

#     :param threshold: The maximum distance to consider objects as "close." Defaults to 0.5 meters.
#     :return: Result message.
#     """
    
#     # Initialize the list of objects to skip
#     agents = SIMULATION.get_agents()["agents"]
#     skip_objects = ["camera", "table"] + agents 
    
#     distances_json = SIMULATION.get_pairwise_distances()

#     if not distances_json:
#         return "No objects were observed in the scene."    rospy.init_node('test_object_manipulation_client')


#     # Find close object pairs
#     close_pairs = []
#     for obj_a, distances in distances_json.items():
#         if obj_a in skip_objects or obj_a.startswith("Hand") or obj_a.startswith("hand"):
#             continue
#         for obj_b, distance in distances.items():
#             # Avoid duplicates (A -> B and B -> A)
#             if obj_b in skip_objects or obj_b.startswith("Hand") or obj_b.startswith("hand"):
#                 continue
#             if obj_a < obj_b and distance <= threshold:
#                 close_pairs.append((obj_a, obj_b))

#     # Format the result
#     if not close_pairs:
#         return f"No objects were found to be within {threshold} meters of each other."

#     result = "The following objects are close to each other:\n"
#     result += "\n".join(f"{pair[0]} is close to {pair[1]}" for pair in close_pairs)
#     return result


def speak(person_name: str, text: str) -> str:
    """
    You speak out the given text.
    You provide a reason for the action you are about to take.

    :param person_name: The name of the person to speak to. The person must be available in the scene. Give "All" if you want to speak to everyone.
    :param text: The text to speak.
    :return: Result message.
    """
    agents = SIMULATION.get_agents()["agents"]
    if person_name not in agents and person_name != "All":
        return f"There is no agent with the name {person_name} in the scene. Did you mean one of these: {agents}?"



    # SIMULATION.execute(f"speak {text}")
    # 
    
    goal = TtsGoal()
    goal.rawtext.text = text
    goal.rawtext.lang_id = "en_GB"
    
    client_tts.send_goal(goal)
    
    # Waits for the server to finish performing the action
    client_tts.wait_for_result()

    # Prints out the result of executing the action
    result = client_tts.get_result()
    
    return f"You said to {person_name}: {text}"
    

    
    

def reasoning(reason: str) -> str:
    """
    You provide a reason for the action you are about to take.

    :param reason: The reason for the action.
    :return: Result message.
    """
    return f"You are about to take the following action: {reason}."

def required_objects(required_objects: str) -> str:
    """
    Yor provide the name of the objects the user requires. 
    :param required_objects: Object names
    :return: Result message.
    """
    return f"The objects required by the user are: {required_objects}"



def hand_object_over_to_person(object_name: str, person_name: str) -> str:
    """
    You get an object and hand it over to a person.

    :param object_name: The name of the object to hand over. The object must be available in the scene.
    :param person_name: The name of the person to hand over the object to. The person must be available in the scene.
    :return: Result message.
    """
    
    objects_names, objects_poses, objects_shapes = extra_tools_tiago.get_objects_names_shapes_and_poses(SIMULATION)
    
    goal = ObjectManipulationGoal()
    
    goal.task = "hand_over_to_person"
    goal.object_for_task_name1 = object_name
    goal.object_for_task_name2 = person_name
    goal.object_names = objects_names
    goal.object_poses = objects_poses
    goal.object_shapes = objects_shapes
    
    goal.zone_place = []
    p1 = Point()
    p1.x = -0.5
    p1.y = -0.3
    p1.z = 0.85
    goal.zone_place.append(p1)

    p2 = Point()
    p2.x = 0.5
    p2.y = -0.3
    p2.z = 0.85
    goal.zone_place.append(p2)

    p3 = Point()
    p3.x = 0.5
    p3.y = 0.3
    p3.z = 0.85
    goal.zone_place.append(p3)

    p4 = Point()
    p4.x = -0.5
    p4.y = 0.3
    p4.z = 0.85
        
    
    goal.zone_place.append(p4)
            
    client.send_goal(goal)
    
    # Waits for the server to finish performing the action
    client.wait_for_result()

    # Prints out the result of executing the action
    result = client.get_result()
    if result.success:
        return f"You moved {object_name} to {person_name}."
      
    return f"You were not able to hand {object_name} over to {person_name}."

def pour_into(source_container_name: str, target_container_name: str) -> str:
    """
    You get a source container, pour it into a target container, and put it back on the table.

    :param source_container_name: The name of the container to pour from.
    :param target_container_name: The name of the container to pour into.
    :return: Result message.
    """
    
    objects_names, objects_poses, objects_shapes = extra_tools_tiago.get_objects_names_shapes_and_poses(SIMULATION)
    
    goal = ObjectManipulationGoal()
    
    goal.task = "pour_into"
    goal.object_for_task_name1 = source_container_name
    goal.object_for_task_name2 = target_container_name
    goal.object_names = objects_names
    goal.object_poses = objects_poses
    goal.object_shapes = objects_shapes
    
    goal.zone_place = []
    
    client.send_goal(goal)
    
    client.wait_for_result()
    result = client.get_result()


    if result.success:
        return f"You poured {source_container_name} into {target_container_name}."
    return f"You were not able to pour {source_container_name} into {target_container_name}."

# def is_person_busy_or_idle(person_name: str) -> str:
#     """
#     Check if the person is busy or idle. If the person is busy, it would be hindered from helping.

#     :param person_name: The name of the person to check. The person must be available in the scene.
#     :return: Result message.
#     """
#     agents = SIMULATION.get_agents()["agents"]
#     if person_name not in agents:
#         return f"There is no agent with the name {person_name} in the scene. Did you mean one of these: {agents}?"

#     busy = SIMULATION.isBusy(person_name)
#     if busy is None:
#         return f"It could not be determined if {person_name} is busy. There were technical problems."
#     return f"{person_name} is {'busy' if busy else 'idle'}."


# def check_hindering_reasons(person_name: str, object_name: str) -> str:
#     """
#     Checks all hindering reasons for a person (busy or idle), and in combination with an object (if person can see and reach object).
#     If the person cannot see or cannot reach the object, it would be hindered from helping with the object.
#     If the person is busy, it would be hindered from helping.

#     :param person_name: The name of the person to check. The person must be available in the scene.
#     :param object_name: The name of the object to check. The object must be available in the scene.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["objects"]
#     if object_name not in objects:
#         return f"There is no object with the name {object_name} in the scene. Did you mean one of these: {objects}?"
#     agents = SIMULATION.get_agents()["agents"]
#     if person_name not in agents:
#         return f"There is no agent with the name {person_name} in the scene. Did you mean one of these: {agents}?"

#     # visibility
#     occluded_by_ = SIMULATION.isOccludedBy(person_name, object_name)["occluded_by"]
#     occluded_by = [e["name"] for e in occluded_by_]
#     if not occluded_by:
#         visible_text = f"{person_name} can see {object_name}."
#     else:
#         visible_text = f"{person_name} cannot see {object_name}, it is occluded by {' and '.join(occluded_by)}."

#     # reachability
#     if SIMULATION.isReachable(person_name, object_name) is True:
#         reachable_text = f"{person_name} can reach {object_name}."
#     else:
#         reachable_text = f"{person_name} cannot reach {object_name}."

#     result_str = visible_text + " "
#     result_str += reachable_text + " "
#     result_str += is_person_busy_or_idle(person_name)
#     return result_str


# def check_reach_object_for_robot(object_name: str) -> str:
#     """
#     Check if you (the_robot) can get the object.

#     :param object_name: The name of the object to check. The object must be available in the scene.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["obhttps://www.google.com/search?q=ttsaction+tiago&client=ubuntu-sn&sca_esv=342fb0b783af65fb&channel=fs&udm=2&biw=1415&bih=742&ei=v6igZ5bkAYnhkdUPpenMqQg&ved=0ahUKEwjWv_O2s6eLAxWJcKQEHaU0M4UQ4dUDCBI&uact=5&oq=ttsaction+tiago&gs_lp=EgNpbWciD3R0c2FjdGlvbiB0aWFnb0iQFVCkA1ipE3ABeACQAQCYAWCgAZMDqgEBNrgBA8gBAPgBAZgCAKACAJgDAIgGAZIHAKAHjgI&sclient=imgjects"]
#     if object_name not in objects:
#         return f"There is no object with the name {object_name} in the scene. Did you mean one of these: {objects}?"

#     reachable = SIMULATION.isReachable("robot", object_name)
#     if reachable:
#         return f"You can get {object_name}."
#     return f"You cannot get {object_name}. "



# This is just to text if the robot could infere that the user has done the requested action

# def open_object(object_name: str) -> str:
#     """
#     Open an object.

#     :param object_name: The name of the object to open. The object must be available in the scene.
#     :return: Result message.
#     """

#     return f"You were not able to open {object_name}."


# def pour_into(source_container_name: str, target_container_name: str) -> str:
#     """
#     You get a source container, pour it into a target container, and put it back on the table.

#     :param source_container_name: The name of the container to pour from.
#     :param target_container_name: The name of the container to pour into.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["objects"]
#     if source_container_name not in objects:
#         return f"There is no object with the name {source_container_name} in the scene. Did you mean one of these: {objects}?"
#     if target_container_name not in objects:
#         return f"There is no object with the name {target_container_name} in the scene. Did you mean one of these: {objects}?"

#     res = SIMULATION.plan_fb(
#         (
#             f"get {source_container_name} duration 8;"
#             f"pour {source_container_name} {target_container_name};"
#             f"put {source_container_name} table duration 7;"
#             "pose default duration 4"
#         ),
#     )
#     if res.startswith("SUCCESS"):
#         return f"You poured {source_container_name} into {target_container_name}."
#     return f"You were not able to pour {source_container_name} into {target_container_name}."




# def hand_object_over_to_person(object_name: str, person_name: str) -> str:
#     """
#     You get an object and hand it over to a person.

#     :param object_name: The name of the object to hand over. The object must be available in the scene.
#     :param person_name: The name of the person to hand over the object to. The person must be available in the scene.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["objects"]
#     if object_name not in objects:
#         return f"There is no object with the name {object_name} in the scene. Did you mean one of these: {objects}?"
#     agents = SIMULATION.get_agents()["agents"]
#     if person_name not in agents:
#         return f"There is no agent with the name {person_name} in the scene. Did you mean one of these: {agents}?"

#     res = SIMULATION.plan_fb(
#         (
#             f"get {object_name} duration 8;"
#             f"pass {object_name} {person_name};"
#             "pose default duration 4"
#         ),
#     )
#     if res.startswith("SUCCESS"):
#         return f"Passed {object_name} to {person_name}"
#     else:
#         res = SIMULATION.plan_fb(
#             (
#                 f"get {object_name} duration 8;"
#                 f"put {object_name};
#                 f"get {object_name} duration 8;"
#                 f"pass {object_name} {person_name};"
#                 "pose default duration 4"
#             ),
#         )
#     if res.startswith("SUCCESS"):
#         return f"You moved {object_name} to {person_name}."
#     return f"You were not able to hand {object_name} over to {person_name}."


# def move_object_to_person(object_name: str, person_name: str) -> str:
#     """
#     You get an object and move it to a person.

#     :param object_name: The name of the object to move. The object must be available in the scene.
#     :param person_name: The name of the person to move the object to. The person must be available in the scene.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["objects"]
#     if object_name not in objects:
#         return f"There is no object with the name {object_name} in the scene. Did you mean one of these: {objects}?"
#     agents = SIMULATION.get_agents()["agents"]
#     if person_name not in agents:
#         return f"There is no agent with the name {person_name} in the scene. Did you mean one of these: {agents}?"

#     res = SIMULATION.plan_fb(
#         (
#             f"get {object_name};"
#             f"put {object_name} near {person_name};"
#             "pose default,default_up,default_high"
#         ),
#     )
#     if res.startswith("SUCCESS"):
#         return f"You moved {object_name} to {person_name}."
#     else:
#         res = SIMULATION.plan_fb(
#             (
#                 f"get {object_name};"
#                 f"put {object_name};"
#                 f"get {object_name};"
#                 f"put {object_name} near {person_name};"
#                 "pose default,default_up,default_high"
#             ),
#         )
#     if res.startswith("SUCCESS"):
#         return f"You moved {object_name} to {person_name}."
#     return f"You were not able to move {object_name} to {person_name}."


# def move_object_away_from_person(object_name: str, away_from: str) -> str:
#     """
#     You get an object and move it away from a person.

#     :param object_name: The name of the object to move. The object must be available in the scene.
#     :param away_from: The name of the person or object to move the object away from. It must be available in the scene.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["objects"]
#     if object_name not in objects:
#         return f"There is no object with the name {object_name} in the scene. Did you mean one of these: {objects}?"
#     agents = SIMULATION.get_agents()["agents"]
#     if away_from not in agents and away_from not in objects:
#         return (
#             f"There is no agent or object with the name {away_from} in the scene. "
#             f"Did you mean one of these: {agents} or {objects}?"
#         )

#     holdingHand = SIMULATION.is_held_by(
#         (
#             f"{object_name}"
#         ),
#     )
    
#     getCommand = ""
#     if not holdingHand:
#         getCommand = f"get {object_name};"
    
#     res = SIMULATION.plan_fb(
#         (
#             getCommand +
#             f"put {object_name} far {away_from};"
#             "pose default,default_up,default_high"
#         ),
#     )
#     if res.startswith("SUCCESS"):
#         return f"You moved the {object_name} away from the {away_from}."
#     return f"You couldn't move the {object_name} away from the {away_from}: {res}."


# def point_at_object_or_agent(name: str) -> str:
#     """
#     Point at an object or a person. 

#     :param name: The name of the object or person you want to point at.
#     :return: Result message.
#     """
#     objects = SIMULATION.get_objects()["objects"]
#     agents = SIMULATION.get_agents()["agents"]
#     if name not in agents and name not in objects:
#         return (
#             f"There is no agent or object with the name {name} in the scene. "
#             f"Did you mean one of these: {agents} or {objects}?"
#         )

#     res = SIMULATION.plan_fb(
#         (
#             f"point {name};"
#             "pose default,default_up,default_high"
#         ),
#     )
#     if res.startswith("SUCCESS"):
#         return f"You pointed at the {name}."
#     return f"You couldn't point at the {name}: {res}."
