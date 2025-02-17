#pip install tensorflow pillow numpy

import tensorflow as tf
import numpy as np 
import tkinter as tk
from PIL import Image, ImageDraw

#Create the main application window
draw_digit_inter = tk.Tk()
draw_digit_inter.title("Digit Recognition")
draw_digit_inter.geometry("700x580")

#Add a label
label = tk.Label(draw_digit_inter, text="Draw a digit from 0-9!", font=("Arial", 20))
label.pack(pady=20)  #Add padding to separate widgets

#Create canvas
canvas = tk.Canvas(draw_digit_inter, bg="white", width=400, height=400)
canvas.pack()

#Create a new image to draw on
img = Image.new("L", (400, 400), color=255)
draw = ImageDraw.Draw(img)

def draw_on_canvas(event):
    #x and y coordinates
    x, y = event.x, event.y
    #x - , y - : Defines the top-left corner from the cursor
    #x + , y + : Defines the bottom-right corner from the cursor
    size = 15
    canvas.create_oval(x - size, y - size, x + size, y + size, fill="black")
    draw.ellipse([x - size, y - size, x + size, y + size], fill="black")
                       
#Bind mouse motion to the drawing function
canvas.bind("<B1-Motion>", draw_on_canvas)  #Left mouse button drag to draw

def clear_canvas():
    global img, draw
    #Clear the canvas
    canvas.delete("all")

    label.config(text="Draw a digit from 0-9!")
    
    #Reset the image to a blank state
    img = Image.new("L", (400, 400), color=255)
    draw = ImageDraw.Draw(img)

#Load model
model = tf.keras.models.load_model('HandwrittenDigitRecognition.keras')

def test_image():
    img_resize  = img.resize((28, 28))  #Resize the image to 28x28
    img_resize = np.invert(np.array([img_resize]))
    
    #Predict using the model
    prediction = model.predict(img_resize)
    #Print the predicted digit
    predicted_digit = np.argmax(prediction)
    label.config(text=f"This digit is probably a {predicted_digit}")

#Create Submit and Clear Buttons
button_frame = tk.Frame(draw_digit_inter)
button_frame.pack(pady=20, anchor="center")

submit_button = tk.Button(button_frame, text="Submit", font=("Arial", 20), command=test_image)
submit_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(button_frame, text="Clear", font=("Arial", 20), command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=10)

#Run the application
draw_digit_inter.mainloop()