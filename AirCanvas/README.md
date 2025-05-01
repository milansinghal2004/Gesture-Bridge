AirCanvas is a computer vision project that lets you draw using hand gestures. Built with OpenCV and MediaPipe, it tracks your hand movements to simulate basic painting—like MS Paint, but without a mouse.

## About

AirCanvas transforms your webcam into an interactive digital canvas, allowing you to create art using just your hand gestures in the air. The application uses:

- **MediaPipe** for real-time hand tracking and gesture recognition
- **OpenCV** for image processing and drawing operations
- **Computer Vision** techniques to interpret hand movements as drawing commands

The project demonstrates how computer vision can be used to create natural human-computer interfaces, enabling creative expression without physical input devices.

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/milansinghal2004/Gesture-Bridge
   cd aircanvas
   ```

2. Create a virtual environment (recommended):
   ```
   # Using conda
   conda create -n aircanvas python=3.8
   conda activate aircanvas
   
   # Or using venv
   python -m venv aircanvas
   source aircanvas/bin/activate  # On Windows: aircanvas\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python src/main.py
   ```
## Usage

AirCanvas recognises different hand gestures to control drawing functions:

### Drawing Gesture

Make a pinch gesture with your index finger and thumb to draw. Move your hand while maintaining the pinch to create lines and shapes.

![Demonstration of Pinch Gesture](images/pinch.png)

### Select Gesture

Point with your index finger to select colors from the palette on the right side of the screen.

![Demonstration of Select Gesture](images/select.png)


### Erase Gesture

Show an open palm to erase. A circle will appear around your index finger showing the eraser size.

![Demonstration of Erase Gesture](images/erase.png)


### Controls and Features
- **Colour Palette**: Located on the right side of the screen
- **Current Colour**: Displayed in the top-left corner
- **Exit**: Press 'q' to quit the application

### Audio Recognition 

AirCanvas includes an audio recognition feature that allows you to control the application using voice commands. This feature leverages the `speech_recognition` library to interpret spoken words and execute corresponding actions within the application.

### Supported Commands
- **"Clear"**: Clears the canvas.
- **"Exit"**: Closes the application.
- **"Blue", "Red", "Green", "Yellow", "White"**: Changes the drawing color to the specified color.

The audio recognition runs in a separate thread, allowing it to listen for commands continuously while you draw.

## Implementation Progress

### Phase 1: Setup ✅
- [x] Project structure setup
- [x] Dependencies installation
- [x] Camera integration
- [x] Basic configuration

### Phase 2: Hand Tracking ✅
- [x] MediaPipe integration
- [x] Landmark detection
- [x] Coordinate system
- [x] Real-time tracking

### Phase 3: Gesture System ✅
- [x] Basic gesture detection
- [x] Gesture states
- [x] Action mapping
- [x] Gesture refinement

### Phase 4: Canvas System ✅
- [x] Drawing implementation
- [x] Color system

### Phase 5: Features 🔄
- [ ] Tool panel
- [ ] Save/Load system
- [ ] Shapes e.g. Rectangle, Circle
- [ ] Pen thickness

### Legend
✅ Complete
🔄 In Progress
⏳ Planned
