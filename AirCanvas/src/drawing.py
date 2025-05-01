import cv2
import numpy as np
from enum import Enum
from colours import Colours


class Tools(Enum):
    PEN = 1
    ERASER = 2


class DrawingCanvas:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Current drawing settings
        self.current_colour_name = Colours.RED.name
        self.current_colour = Colours.RED
        self.thickness = 15
        self.eraser_thickness = 125
        self.current_tool = Tools.PEN
        self.drawing = False
        self.start_point = None

    def draw(self, point):
        if not self.drawing or self.start_point is None:
            return

        if self.current_tool == Tools.PEN:
            # Get the actual BGR color to use
            bgr_colour = self.current_colour.value
            # print(f"Drawing with colour: {self.current_colour_name}, BGR: {bgr_colour}")
            cv2.line(self.canvas, self.start_point, point, bgr_colour, self.thickness)
            self.start_point = point
            return

        if self.current_tool == Tools.ERASER:
            cv2.line(self.canvas, self.start_point, point, (0, 0, 0), self.eraser_thickness)
            self.start_point = point

    def start_drawing(self, point):
        self.drawing = True
        self.start_point = point
        print(f"Started drawing with: {self.current_colour_name}")

    def stop_drawing(self):
        self.drawing = False
        self.start_point = None

    def set_colour(self, colour: str):
        self.current_colour_name = colour
        self.current_colour = Colours[colour]
        print(f"Canvas colour set to: {colour}, BGR: {self.current_colour}")

    def set_tool(self, tool: Tools):
        self.current_tool = tool

    def get_display(self):
        return self.canvas.copy()

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
