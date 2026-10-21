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

st.set_page_config(page_title="AI Whiteboard", page_icon="✍️", layout="wide", initial_sidebar_state="collapsed")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Same order as FingerWhiteboard.color_names; the heart emoji is only the picker label.
COLOR_LABELS = ["💜 Purple", "💚 Emerald", "❤️ Red", "💛 Amber", "🩷 Pink", "🩵 Sky", "🤍 White", "🩶 Gray"]
DEFAULT_COLOR = COLOR_LABELS[1]

st.markdown(
    """
    <style>
      .block-container { max-width: 1100px; padding-top: 2rem; }
      .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: -0.02em; }
      .hero p { margin: .25rem 0 1rem; opacity: .7; }
      .status { display: inline-flex; align-items: center; gap: .4rem; padding: .15rem .7rem;
                border-radius: 999px; font-size: .85rem; border: 1px solid rgba(128,128,128,.35); }
      .dot { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }
      .card h4 { margin: 0 0 .25rem; }
      .card p { margin: 0; opacity: .75; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


class WhiteboardProcessor(VideoProcessorBase):
    def __init__(self):
        # lite_ui skips the full-frame blends, which dominate the per-frame cost on a shared CPU.
        # show_overlay=False keeps the video clean: colours and instructions live in the page instead.
        self.board = FingerWhiteboard(
            open_camera=False, size=(800, 450), keyboard_hints=False, lite_ui=True, show_overlay=False
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        out = self.board.process_frame(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(out, format="bgr24")


st.markdown(
    '<div class="hero"><h1>✍️ AI Whiteboard</h1>'
    "<p>Draw in the air with your hand. No mouse, no marker.</p></div>",
    unsafe_allow_html=True,
)

# ---- Toolbar -------------------------------------------------------------------------------
with st.container(border=True):
    c_color, c_brush, c_clear, c_capture = st.columns([5, 3, 1.4, 1.6], vertical_alignment="bottom")
    with c_color:
        picked = st.pills("Colour", COLOR_LABELS, default=DEFAULT_COLOR, selection_mode="single")
    with c_brush:
        brush = st.slider("Brush size", 5, 30, 15, step=2)
    with c_clear:
        clear = st.button("Clear", icon=":material/delete:", width="stretch")
    with c_capture:
        capture = st.button("Capture", icon=":material/photo_camera:", type="primary", width="stretch")

# Pills can be toggled off; keep the last colour instead of leaving the pen undefined.
if picked is None:
    picked = st.session_state.get("last_color", DEFAULT_COLOR)
st.session_state["last_color"] = picked

# ---- Canvas --------------------------------------------------------------------------------
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

live = bool(ctx.state.playing)
st.markdown(
    f'<span class="status"><span class="dot" style="background:{"#22c55e" if live else "#9ca3af"}"></span>'
    f'{"Live: hold your hand up to the camera" if live else "Press Start and allow camera access"}</span>',
    unsafe_allow_html=True,
)

if ctx.video_processor:
    board = ctx.video_processor.board
    idx = COLOR_LABELS.index(picked)
    board.current_color_index = idx
    board.draw_color = board.colors[idx]
    board.brush_thickness = brush
    if clear:
        board.reset_canvas()
    if capture:
        ok, png = cv2.imencode(".png", board.canvas)
        if ok:
            st.session_state["snapshot"] = png.tobytes()

# ---- How to use ----------------------------------------------------------------------------
st.write("")
a, b, c = st.columns(3)
with a.container(border=True):
    st.markdown(
        '<div class="card"><h4>👍 Draw</h4><p>Touch your thumb and index finger together, '
        "then move your index fingertip.</p></div>",
        unsafe_allow_html=True,
    )
with b.container(border=True):
    st.markdown(
        '<div class="card"><h4>✌️ Erase</h4><p>Touch your index and middle finger together '
        "and sweep over what you want gone.</p></div>",
        unsafe_allow_html=True,
    )
with c.container(border=True):
    st.markdown(
        '<div class="card"><h4>💡 Tips</h4><p>Good light, one hand, palm towards the camera, '
        "about an arm's length away.</p></div>",
        unsafe_allow_html=True,
    )

# ---- Captured drawing ----------------------------------------------------------------------
snapshot = st.session_state.get("snapshot")
if snapshot:
    st.subheader("Captured drawing")
    st.image(snapshot, width=480)
    st.download_button("Download PNG", snapshot, "whiteboard.png", "image/png", icon=":material/download:")

with st.expander("Trouble connecting?"):
    st.markdown(
        "- Allow camera access when your browser asks, and close other apps using the camera.\n"
        "- The video goes over WebRTC, which some school, work or VPN networks block. "
        "Try another network or browser.\n"
        "- Frames are processed in memory to track your hand and are never saved."
    )
