"""Browser front end for the gesture whiteboard.

The visitor's webcam is streamed to this server over WebRTC; each frame goes through
FingerWhiteboard.process_frame (MediaPipe hand tracking + drawing) and comes back
as video. Frames are processed in memory and are not stored.

Run locally:  streamlit run streamlit_app.py
"""
import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from whiteboard import FingerWhiteboard

st.set_page_config(page_title="AI Whiteboard", page_icon="✍️", layout="wide")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
COLOR_NAMES = ["Purple", "Emerald", "Red", "Amber", "Pink", "Sky", "White", "Gray"]


class WhiteboardProcessor(VideoProcessorBase):
    def __init__(self):
        # lite_ui skips the full-frame blends, which dominate the per-frame cost on a shared CPU.
        self.board = FingerWhiteboard(open_camera=False, size=(800, 450), keyboard_hints=False, lite_ui=True)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        out = self.board.process_frame(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(out, format="bgr24")


st.title("AI Whiteboard")
st.write(
    "Draw in the air. Press **Start**, allow camera access, and hold your hand up. "
    "**Touch thumb and index finger** to draw, **touch index and middle finger** to erase."
)

with st.sidebar:
    st.header("Brush")
    color_name = st.radio("Color", COLOR_NAMES, index=1)
    brush = st.slider("Brush size", 5, 30, 15, step=2)
    clear = st.button("Clear canvas")
    st.caption(
        "Frames are processed in memory to track your hand and are never saved. "
        "If the video does not connect, your network may block WebRTC."
    )

ctx = webrtc_streamer(
    key="whiteboard",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=WhiteboardProcessor,
    # Ask for a small, capped-rate camera stream: less to encode, send and process means less lag.
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 360}, "frameRate": {"ideal": 24, "max": 30}},
        "audio": False,
    },
    # The stream is 800x450; stretch the player to the page width so the canvas is large.
    video_html_attrs={"style": {"width": "100%", "margin": "0 auto"}, "controls": False, "autoPlay": True},
    async_processing=True,
)

if ctx.video_processor:
    board = ctx.video_processor.board
    idx = COLOR_NAMES.index(color_name)
    board.current_color_index = idx
    board.draw_color = board.colors[idx]
    board.brush_thickness = brush
    if clear:
        board.reset_canvas()

    ok, png = cv2.imencode(".png", board.canvas)
    if ok:
        st.download_button("Download drawing (PNG)", png.tobytes(), "whiteboard.png", "image/png")
