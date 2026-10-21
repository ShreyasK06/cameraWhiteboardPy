# Camera Whiteboard

**Draw in the air.** A hand-gesture whiteboard that turns your webcam into a drawing surface: pinch to draw, two fingers to erase, no mouse or marker required. Built with MediaPipe hand tracking and OpenCV.

## Features

- **Gesture control.** Touch your thumb and index finger together to draw; touch index and middle finger together to erase.
- **Real-time tracking.** One-hand tracking with the lightweight MediaPipe model, a 1280x720 capture and a 1-frame buffer to keep latency low.
- **Eight-colour palette and adjustable brush**, with a glow cursor that shows the current mode.
- **Dark canvas UI** with a status panel (current gesture, colour, brush size, FPS) and on-screen colour palette.
- **Save your work.** Export a screenshot and the canvas as images.

## Controls

| Input | Action |
|---|---|
| Thumb + index finger together | Draw |
| Index + middle finger together | Erase |
| `1` to `8` | Choose a colour |
| `+` / `-` | Increase / decrease brush size |
| `s` | Save a screenshot and the canvas |
| `c` | Clear the canvas |
| `q` | Quit |

## Web app (Streamlit)

**Live demo: https://shreyask06-camerawhiteboard.streamlit.app** (press Start and allow camera access).

`streamlit_app.py` runs the same tracking and drawing code in the browser. The visitor's webcam is streamed to the server over WebRTC ([streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)); each frame goes through `FingerWhiteboard.process_frame`, and the annotated video comes back. Colour, brush size, clear and a PNG download are in the sidebar. Frames are processed in memory and are not stored.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io): point it at this repo with `streamlit_app.py` as the main file, pick **Python 3.11** under Advanced settings (the pinned `mediapipe==0.10.21` has no wheels for newer Pythons), and set the app URL. `packages.txt` installs the system libraries OpenCV needs. On a shared CPU the stream runs at a lower frame rate than the desktop app, and some networks block WebRTC.

## Run it on the desktop

Requires Python 3.9+ and a webcam.

```bash
pip install -r requirements.txt     # opencv-python, mediapipe, numpy, streamlit, streamlit-webrtc
python whiteboard.py
```

Keep your hand clearly visible and well lit. The app prints its controls on launch.

## How it works

1. OpenCV grabs frames from the camera and mirrors them.
2. MediaPipe Hands returns 21 landmarks for the detected hand.
3. Distances between fingertip landmarks (in MediaPipe's normalized image coordinates) are compared against fixed thresholds to classify the gesture as draw, erase, or plain tracking.
4. The index fingertip position drives the cursor, and strokes are drawn onto a separate canvas that is composited under the UI.

## Tech stack

Python, OpenCV, MediaPipe, NumPy, Streamlit with streamlit-webrtc for the browser version.

## Known limitations

- Gesture thresholds are fixed values tuned by hand, so results vary with lighting, camera distance and hand size.
- Only one hand is tracked at a time.
- There is no automated test suite; behaviour was checked by hand against a live camera.
