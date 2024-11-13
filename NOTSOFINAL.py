import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Import Image and ImageTk from PIL
from ultralytics import YOLO
import datetime  # Import datetime for timestamp

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection")
        
        self.model = YOLO("ReefRouverYolo11.pt")
        self.video_source = 0
        self.is_recording = False
        self.out = None
        
        # Create a canvas to display the video
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        # Create a record button
        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack()
        
        # Start video capture
        self.vid = cv2.VideoCapture(self.video_source)
        self.update()
    
    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            if self.out:
                self.out.release()
                self.out = None
            messagebox.showinfo("Recording", "Recording stopped.")
        else:
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            
            # Generate a unique filename using the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'output_{timestamp}.avi'  # Example filename: output_20231001_123456.avi
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))  # Set to 30 FPS
            messagebox.showinfo("Recording", f"Recording started. Saving to {filename}.")
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Perform prediction
            results = self.model.predict(source=frame, conf=0.6, show_conf=False)
            annotated_frame = results[0].plot()  # Get the annotated frame
            
            # If recording, write the frame to the video file
            if self.is_recording and self.out is not None:
                self.out.write(annotated_frame)
            
            # Convert the frame to a format suitable for Tkinter
            self.photo = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.photo = Image.fromarray(self.photo)  # Use PIL's Image.fromarray
            self.photo = ImageTk.PhotoImage(self.photo)  # Convert to PhotoImage
            
            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Call this function again after 10 ms
        self.root.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        if self.out:
            self.out.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()