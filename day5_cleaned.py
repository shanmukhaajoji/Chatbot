#!/usr/bin/env python
# coding: utf-8

# # Project - Airline AI Assistant
# 
# We'll now bring together what we've learned to make an AI Customer Support assistant for an Airline

# In[ ]:


# imports

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


# In[ ]:


# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4o-mini"
openai = OpenAI()


# In[ ]:


system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."


# In[ ]:


# This function looks rather simpler than the one from my video, because we're taking advantage of the latest Gradio updates

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()


# ## Tools
# 
# Tools are an incredibly powerful feature provided by the frontier LLMs.
# 
# With tools, you can write a function, and have the LLM call that function as part of its response.
# 
# Sounds almost spooky.. we're giving it the power to run code on our machine?
# 
# Well, kinda.

# In[ ]:


# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")


# In[ ]:


get_ticket_price("London")


# In[ ]:


# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}


# In[ ]:


# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]


# ## Getting OpenAI to use our Tool
# 
# There's some fiddly stuff to allow OpenAI "to call our tool"
# 
# What we actually do is give the LLM the opportunity to inform us that it wants us to run the tool.
# 
# Here's how the new chat function looks:

# In[ ]:


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content


# In[ ]:


# We have to write that function handle_tool_call:

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city


# In[ ]:


gr.ChatInterface(fn=chat, type="messages").launch()


# # Let's go multi-modal!!
# 
# We can use DALL-E-3, the image generation model behind GPT-4o, to make us some images
# 
# Let's put this in a function called artist.
# 
# ### Price alert: each time I generate an image it costs about 4 cents - don't go crazy with images!

# In[ ]:


# Some imports for handling images

import base64
from io import BytesIO
from PIL import Image


# In[ ]:


def artist(city):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


# In[ ]:


image = artist("New York City")
display(image)


# In[ ]:





# ## Audio (NOTE - Audio is optional for this course - feel free to skip Audio if it causes trouble!)
# 
# And let's make a function talker that uses OpenAI's speech model to generate Audio
# 
# ### Troubleshooting Audio issues
# 
# If you have any problems running this code below (like a FileNotFound error, or a warning of a missing package), you may need to install FFmpeg, a very popular audio utility.
# 
# **For PC Users**
# 
# Detailed instructions are [here](https://chatgpt.com/share/6724efee-6b0c-8012-ac5e-72e2e3885905) and summary instructions:
# 
# 1. Download FFmpeg from the official website: https://ffmpeg.org/download.html
# 
# 2. Extract the downloaded files to a location on your computer (e.g., `C:\ffmpeg`)
# 
# 3. Add the FFmpeg bin folder to your system PATH:
# - Right-click on 'This PC' or 'My Computer' and select 'Properties'
# - Click on 'Advanced system settings'
# - Click on 'Environment Variables'
# - Under 'System variables', find and edit 'Path'
# - Add a new entry with the path to your FFmpeg bin folder (e.g., `C:\ffmpeg\bin`)
# - Restart your command prompt, and within Jupyter Lab do Kernel -> Restart kernel, to pick up the changes
# 
# 4. Open a new command prompt and run this to make sure it's installed OK
# `ffmpeg -version`
# 
# **For Mac Users**
# 
# 1. Install homebrew if you don't have it already by running this in a Terminal window and following any instructions:  
# `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
# 
# 2. Then install FFmpeg with `brew install ffmpeg`
# 
# 3. Verify your installation with `ffmpeg -version` and if everything is good, within Jupyter Lab do Kernel -> Restart kernel to pick up the changes
# 
# Message me or email me at ed@edwarddonner.com with any problems!

# ## To check you now have ffmpeg and can access it here
# 
# Excecute the next cell to see if you get a version number. (Putting an exclamation mark before something in Jupyter Lab tells it to run it as a terminal command rather than python code).
# 
# If this doesn't work, you may need to actually save and close down your Jupyter lab, and start it again from a new Terminal window (Mac) or Anaconda prompt (PC), remembering to activate the llms environment. This ensures you pick up ffmpeg.
# 
# And if that doesn't work, please contact me!

# In[ ]:


get_ipython().system('ffmpeg -version')
get_ipython().system('ffprobe -version')
get_ipython().system('ffplay -version')


# # For Mac users - and possibly many PC users too
# 
# This version should work fine for you. It might work for Windows users too, but you might get a Permissions error writing to a temp file. If so, see the next section!
# 
# As always, if you have problems, please contact me! (You could also comment out the audio talker() in the later code if you're less interested in audio generation)

# In[ ]:


from pydub import AudioSegment
from pydub.playback import play

def talker(message):
    response = openai.audio.speech.create(
      model="tts-1",
      voice="onyx",    # Also, try replacing onyx with alloy
      input=message
    )

    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


# In[ ]:


talker("Well, hi there")


# # For Windows users (or any Mac users with problems above)
# 
# ## First try the Mac version above, but if you get a permissions error writing to a temp file, then this code should work instead.
# 
# A collaboration between students Mark M. and Patrick H. and Claude got this resolved!
# 
# Below are 4 variations - hopefully one of them will work on your PC. If not, message me please!
# 
# And for Mac people - all 3 of the below work on my Mac too - please try these if the Mac version gave you problems.
# 
# ## PC Variation 1

# In[ ]:


import base64
from io import BytesIO
from PIL import Image
from IPython.display import Audio, display

def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message)

    audio_stream = BytesIO(response.content)
    output_filename = "output_audio.mp3"
    with open(output_filename, "wb") as f:
        f.write(audio_stream.read())

    # Play the generated audio
    display(Audio(output_filename, autoplay=True))

talker("Well, hi there")


# ## PC Variation 2

# In[ ]:


import tempfile
import subprocess
from io import BytesIO
from pydub import AudioSegment
import time

def play_audio(audio_segment):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    try:
        audio_segment.export(temp_path, format="wav")
        time.sleep(3) # Student Dominic found that this was needed. You could also try commenting out to see if not needed on your PC
        subprocess.call([
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            temp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Also, try replacing onyx with alloy
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play_audio(audio)

talker("Well hi there")


# ## PC Variation 3

# In[ ]:


import os
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

def talker(message):
    # Set a custom directory for temporary files on Windows
    custom_temp_dir = os.path.expanduser("~/Documents/temp_audio")
    os.environ['TEMP'] = custom_temp_dir  # You can also use 'TMP' if necessary

    # Create the folder if it doesn't exist
    if not os.path.exists(custom_temp_dir):
        os.makedirs(custom_temp_dir)

    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Also, try replacing onyx with alloy
        input=message
    )

    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")

    play(audio)

talker("Well hi there")


# ## PC Variation 4
# 
# ### Let's try a completely different sound library
# 
# First run the next cell to install a new library, then try the cell below it.

# In[ ]:


get_ipython().system('pip install simpleaudio')


# In[ ]:


from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import simpleaudio as sa

def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Also, try replacing onyx with alloy
        input=message
    )

    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")

    # Create a temporary file in a folder where you have write permissions
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=os.path.expanduser("~/Documents")) as temp_audio_file:
        temp_file_name = temp_audio_file.name
        audio.export(temp_file_name, format="wav")

    # Load and play audio using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(temp_file_name)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait for playback to finish

    # Clean up the temporary file afterward
    os.remove(temp_file_name)

talker("Well hi there")


# ## PC Users - if none of those 4 variations worked!
# 
# Please get in touch with me. I'm sorry this is causing problems! We'll figure it out.
# 
# Alternatively: playing audio from your PC isn't super-critical for this course, and you can feel free to focus on image generation and skip audio for now, or come back to it later.

# # Our Agent Framework
# 
# The term 'Agentic AI' and Agentization is an umbrella term that refers to a number of techniques, such as:
# 
# 1. Breaking a complex problem into smaller steps, with multiple LLMs carrying out specialized tasks
# 2. The ability for LLMs to use Tools to give them additional capabilities
# 3. The 'Agent Environment' which allows Agents to collaborate
# 4. An LLM can act as the Planner, dividing bigger tasks into smaller ones for the specialists
# 5. The concept of an Agent having autonomy / agency, beyond just responding to a prompt - such as Memory
# 
# We're showing 1 and 2 here, and to a lesser extent 3 and 5. In week 8 we will do the lot!

# In[ ]:


def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]


    return history, image


# In[ ]:


# More involved Gradio code as we're not using the preset Chat interface!
# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)


# # Exercises and Business Applications
# 
# Add in more tools - perhaps to simulate actually booking a flight. A student has done this and provided their example in the community contributions folder.
# 
# Next: take this and apply it to your business. Make a multi-modal AI assistant with tools that could carry out an activity for your work. A customer support assistant? New employee onboarding assistant? So many possibilities! Also, see the week2 end of week Exercise in the separate Notebook.

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../thankyou.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#090;">I have a special request for you</h2>
#             <span style="color:#090;">
#                 My editor tells me that it makes a HUGE difference when students rate this course on Udemy - it's one of the main ways that Udemy decides whether to show it to others. If you're able to take a minute to rate this, I'd be so very grateful! And regardless - always please reach out to me at ed@edwarddonner.com if I can help at any point.
#             </span>
#         </td>
#     </tr>
# </table>
