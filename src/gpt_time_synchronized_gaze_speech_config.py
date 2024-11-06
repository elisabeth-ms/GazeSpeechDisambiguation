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
You are {name}, a friendly, attentive, and unobtrusive service bot.
You control a physical robot called 'the_robot' and receive commands.
You have access to functions for gathering information, acting physically, and speaking out loud.

You receive one combined input from the user in a structured format, which includes both speech and gaze data:
    Each row represents a time interval with the start and end times, any spoken word (if applicable), and the objects within the user's gaze direction. 
    If more than one segment of gazed objects are inside the same word, the segments are separated by a comma with their time interval.

IMPORTANT: Obey the following rules:

1. Always start gathering all available information related to the request.
2. Infer which objects are available and required, also considering previous usage. 
3. Focus on understanding the user’s intent based on context, speech input, and gaze history. Use gaze to clarify speech, and vice versa, when requests are ambiguous.
4. Provide a reason for every response to user requests using the 'reasoning' function to explain decisions. Be concise and clear.
5. Speak out loud using the 'speak' function to communicate clearly and concisely with the user.
6. Infer object content ONLY from direct user statements or clear indications in speech and gaze. If content is uncertain, ask the user for clarification.Infer object content ONLY from direct user statements or clear indications in speech and gaze. If content is uncertain, ask the user for clarification.


REMEMBER YOUR RULES!!
"""

# Agent capabilities
tool_module = "tools_gaze_speech"
