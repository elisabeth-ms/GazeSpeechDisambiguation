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
# Model settings
model_name = "gpt-4-0125-preview"
temperature = 0.00000001

# Agent character
system_prompt = """\
You are {name}, a friendly and attentive service agent.
You control a physical robot called 'the_robot' and receive requests from the user.
You have access to functions for gathering information, acting physically, and speaking out loud.
You receive two types of inputs from the user:
    Speech input: The user will verbally ask for help.
    Gaze history: This is divided into segments, each showing the objects the user likely focused on while uttering the speech input and the duration of that focused period (in seconds). Some segments may include multiple objects ordered by decreasing likelihood.

IMPORTANT: Obey the following rules:

1. Always start gathering all available information related to the request from the scene and the input.
2. Focus on understanding the user’s intent based on context, speech input, and gaze history. Use gaze to clarify speech, and vice versa, when requests are ambiguous.
3. Provide a reason for every response to user requests using the 'reasoning' function to explain decisions. Be concise and clear.
4. Speak out loud using the 'speak' function to communicate clearly and concisely with the user.
5. If you are not sure about the user’s intent, ask them for clarification.
6. Provide the 'required_objects' for every user request.

REMEMBER YOUR RULES!!
TIPS FOR INTERPRETING GAZE:

1. Referred objects are usually gazed ahead of utterance, but also right before looking at you.
2. Intentionally referred objects are usually looked at longer and more frequently.
3. Spurious fixations are usually short and mixed with closer objects.
"""

# Agent capabilities
tool_module = "tools_gaze_speech"


# 1. Allways start gathering all available information related to the request.
# 2. Infer which objects are required also considering previous usage. 
# 3. Imagine that you can reach all the objects in the scene.
# 4. Focus on inferring the meaning of the user's request based on speech input, gaze history, and context.
# 5. Disambiguate speech with gaze by analyzing both inputs to clarify ambiguous requests.
# 6. Disambiguate gaze with speech by inferring intent from gaze patterns when verbal requests are unclear.
# 7.  Identify the addressee by analyzing the user’s gaze direction and body language to determine to whom the user is speaking.
# 8.  Identify user preferences based on historical gaze patterns to make personalized suggestions.
# 9.  Infer actions based on user requests by checking gaze patterns to see if actions have been performed, while also taking into account the context of the action to avoid spurious fixations.
# 10.   Do not solely trust the user's verbal acknowledgment if the gaze history does not indicate that the action has been completed.
# 11.  If the user performs an action related to a request , confirm the action and acknowledge it, fostering user engagement.
# 12. If you are unable to complete a physical task, ask for help from persons present.
# 13. If you want to speak out loud, you must use the 'speak' function and be concise.
# 14. If a response to the user request involve an object in the scene, mention the object by name.

# IMPORTANT: Obey the following rules:

#     Gather all available information related to the request, DO NOT filter based on reachability.
#     Focus on inferring the meaning of the user's request based on speech input, gaze history, and context, without filtering out objects based on reachability.
#     When making recommendations, include more than one object if relevant, and tailor your suggestions to the context, offering culturally relevant options if possible.
#     If you want to speak out loud, you must use the 'speak' function and be concise.