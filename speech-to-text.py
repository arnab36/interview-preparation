# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:51:50 2022

@author: 01927Z744

 The followng program will convert an audo to text.
 
     pip install pipwin
     pipwin install pyaudio
 
 
"""

import speech_recognition as sr
import pyttsx3
 
# Initialize the recognizer
r = sr.Recognizer()
 
# Function to convert text to
# speech
def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
     
     
# Loop infinitely for user to
# speak

def append_to_file(txt):
    with open("test.txt", "a") as myfile:
        myfile.write(" "+txt)

def run_voice_recognition():
    while(1):   
         
        # Exception handling to handle
        # exceptions at the runtime
        try:
             
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                 
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)
                 
                #listens for the user's input
                audio2 = r.listen(source2)
                 
                # Using google to recognize audio
                my_text = r.recognize_google(audio2)
                my_text = my_text.lower()
                append_to_file(my_text)
                print(my_text)
                if my_text == "stop it":
                    break
                SpeakText(my_text)
                 
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
             
        except sr.UnknownValueError:
            print("unknown error occured")