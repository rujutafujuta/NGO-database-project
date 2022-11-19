import cv2
import os
import numpy as np
from PIL import Image
from tkinter import *
from face_detection import *
from face_recognition import *

#gui

root=Tk()
root.geometry("")
frame= Frame(root, borderwidth=6,)
root.geometry("500x400")
root.title("ngo database")

frame = Frame(root, bg = "grey", borderwidth = 6, relief = RIDGE)
frame.pack(side = LEFT, anchor = 'nw', fill = X)

b1 = Button(frame, fg = "blue", text = "check blacklist", command =lambda :face_recognition())
b1.pack(side = LEFT)

b2 = Button(frame, fg = "blue", text = "add person to blacklist", command = lambda : face_detection())
b2.pack(side = LEFT)

root.mainloop()