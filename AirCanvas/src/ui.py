import cv2
from colours import Colours


class UIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.box_size = 120
        self.margin = 20
        self.selected_colour = Colours.RED.name

        # Create colour boxes
        self.colour_boxes = {}
        y_pos = self.margin
        x_pos = self.width - self.box_size - self.margin

        for colour in Colours:
            self.colour_boxes[colour.name] = (
                x_pos,
                y_pos,
                self.box_size,
                self.box_size,
            )
            y_pos += self.box_size + self.margin

    def draw_box(self, frame, colour_name, x, y, w, h):
        cv2.rectangle(frame, (x, y), (x + w, y + h), Colours[colour_name].value, -1)

    def draw_text(
        self, frame, text, x, y, font_scale=1, color=(255, 255, 255), thickness=2
    ):
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
            False,
        )

    def draw_selected_colour(self, frame, box):
        (x, y, w, h) = box
        cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 255, 255), 2)

    def draw(self, frame, last_audio_command: str):
        # Draw colour boxes
        for colour_name, (x, y, w, h) in self.colour_boxes.items():
            self.draw_box(frame, colour_name, x, y, w, h)

        self.draw_selected_colour(frame, self.colour_boxes[self.selected_colour])

        # Draw current colour indicator
        cv2.rectangle(
            frame, (10, 10), (50, 50), Colours[self.selected_colour].value, -1
        )
        cv2.rectangle(frame, (10, 10), (50, 50), (255, 255, 255), 2)

    def handle_selection(self, point):
        for colour_name, (x, y, w, h) in self.colour_boxes.items():
            if (x <= point[0] <= x + w) and (y <= point[1] <= y + h):
                self.selected_colour = colour_name
                return True, colour_name
        return False, None

    def set_colour(self, colour_name):
        self.selected_colour = colour_name
