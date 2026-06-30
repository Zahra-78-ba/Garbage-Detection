import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Garbage Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

* { box-sizing: border-box; }

.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #04080f !important;
    color: #e2f0ff;
}

[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: 0; z-index: -1;
    background:
        radial-gradient(ellipse 60% 50% at 0% 0%,   rgba(0,200,180,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 100% 100%, rgba(255,90,31,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 80% 80% at 50% 50%,  rgba(4,8,15,1) 0%, transparent 100%),
        #04080f;
}

[data-testid="stSidebar"] {
    background: #060d18 !important;
    border-right: 1px solid rgba(0,200,180,0.15) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #a8d5d0 !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #00c8b4 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

.hero-banner {
    width: 100%;
    background: linear-gradient(135deg, #060d18 0%, #091520 40%, #060d18 100%);
    border-bottom: 1px solid rgba(0,200,180,0.12);
    padding: 28px 40px 28px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 0%, #00c8b4 30%, #ff5a1f 70%, transparent 100%);
}
.hero-banner::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 100% at 50% 0%, rgba(0,200,180,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 5.5rem; font-weight: 800;
    line-height: 1; letter-spacing: -3px;
    color: #ffffff;
    margin: 0 0 12px;
    text-shadow: 0 0 80px rgba(0,200,180,0.3);
}
.hero-title span {
    background: linear-gradient(90deg, #00c8b4, #00e8d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem; font-weight: 400;
    color: rgba(168,213,208,0.55);
    max-width: 560px; margin: 0 auto; line-height: 1.8;
    font-style: italic;
}

[data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] {
    background: rgba(6,13,24,0.85);
    border: 1px solid rgba(0,200,180,0.15);
    border-radius: 16px;
    padding: 28px !important;
    position: relative;
    transition: border-color 0.3s, box-shadow 0.3s;
    min-height: 420px;
}
[data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"]:hover {
    border-color: rgba(0,200,180,0.3);
    box-shadow: 0 0 40px rgba(0,200,180,0.06);
}

.step-tag {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.64rem; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: #ff5a1f;
    margin-bottom: 6px;
}
.step-tag::before {
    content: "";
    width: 18px; height: 1.5px;
    background: #ff5a1f;
    display: inline-block;
}
.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    color: #e8f4f2;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
}
.card-title::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,200,180,0.2), transparent);
}

.empty-box {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 300px; gap: 16px;
}
.empty-ring {
    width: 80px; height: 80px;
    border: 1.5px dashed rgba(0,200,180,0.2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
}
.empty-msg {
    font-size: 0.82rem; color: rgba(168,213,208,0.3);
    letter-spacing: 0.3px; text-align: center;
}

.stButton > button {
    background: linear-gradient(135deg, #00a090 0%, #00c8b4 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 0.93rem !important;
    letter-spacing: 1.2px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 10px !important;
    padding: 14px 28px !important; width: 100% !important;
    margin-top: 16px !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 24px rgba(0,200,180,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(0,200,180,0.45) !important;
    background: linear-gradient(135deg, #00b8a4 0%, #00dfc8 100%) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,200,180,0.03) !important;
    border: 2px dashed rgba(0,200,180,0.22) !important;
    border-radius: 12px !important;
    transition: all 0.25s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0,200,180,0.5) !important;
    background: rgba(0,200,180,0.06) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #00c8b4 !important; }
[data-testid="stFileUploader"] label {
    color: #5ee8d8 !important;
    font-size: 0.88rem !important; font-weight: 500 !important;
}

.det-row {
    display: flex; align-items: center; justify-content: space-between;
    border-left: 3px solid;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 7px 0;
    transition: transform 0.18s, opacity 0.18s;
    border-top: 1px solid rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.det-row:hover { transform: translateX(5px); opacity: 0.9; }
.det-name {
    font-size: 0.96rem; font-weight: 600;
    color: #e2f0ff; display: flex; align-items: center; gap: 8px;
}
.det-dot {
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
}
.det-badge {
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem; font-weight: 700;
    padding: 3px 12px; border-radius: 20px;
    border: 1px solid;
}

.stats-row { display: flex; gap: 10px; margin: 0 0 18px; }
.stat-b {
    flex: 1;
    background: rgba(0,200,180,0.06);
    border: 1px solid rgba(0,200,180,0.15);
    border-radius: 10px; padding: 14px 12px; text-align: center;
}
.stat-n {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem; font-weight: 800;
    color: #00c8b4; line-height: 1;
}
.stat-l {
    font-size: 0.62rem; letter-spacing: 2px; text-transform: uppercase;
    color: rgba(168,213,208,0.38); margin-top: 6px;
}

.sb-quote {
    background: rgba(0,200,180,0.05);
    border-left: 2px solid rgba(0,200,180,0.3);
    border-radius: 0 6px 6px 0;
    padding: 11px 14px; font-style: italic;
    font-size: 0.82rem; line-height: 1.72;
    color: rgba(168,213,208,0.55) !important; margin: 12px 0;
}

[data-testid="stImage"] { width: 100% !important; }
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid rgba(0,200,180,0.15) !important;
    width: 100% !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #04080f; }
::-webkit-scrollbar-thumb { background: #00a090; border-radius: 4px; }
hr { border: none !important; border-top: 1px solid rgba(0,200,180,0.1) !important; margin: 14px 0 !important; }
.footer {
    text-align: center; padding: 24px 0 10px;
    font-size: 0.73rem; color: rgba(168,213,208,0.18); letter-spacing: 0.8px;
}
</style>
""", unsafe_allow_html=True)


# ── MODEL ──
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## Smart Waste\nDetection")

    for p in ["dataset-cover.jpg", "sidebar.jpg", "logo.jpg"]:
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            break

    st.markdown("""
    <div class='sb-quote'>
    "The Earth is not a garbage bin — every piece of trash we ignore today becomes tomorrow's legacy of neglect."
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    <div style='color:#7ecdc7; font-size:0.85rem; line-height:1.85;'>
    Trained with <b style='color:#ff7a4f'>YOLOv8</b> on real-world waste image datasets.<br>
    Detects &amp; classifies garbage to support smarter recycling and waste management.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


# ── DETECT ──
def detect_image(image):
    arr = np.array(image)
    results = model(arr, conf=0.25)
    return results[0].plot(), results[0]


# ── HERO ──
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🌍 Garbage <span>Detection</span></div>
    <p class='hero-sub'>Every piece of waste identified today is a step towards a cleaner, greener tomorrow.</p>
</div>
""", unsafe_allow_html=True)
# ── TWO COLUMNS ──
col1, col2 = st.columns([1.05, 0.95], gap="large")

# ── COL 1: Upload ──
with col1:
    st.markdown('<div class="step-tag">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a JPG / PNG image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        detect_btn = st.button("🔍  DETECT GARBAGE")
    else:
        st.markdown("""
        <div class='empty-box'>
            <div class='empty-ring'>🖼️</div>
            <div class='empty-msg'>Drop or click to upload an image<br>JPG, JPEG or PNG supported</div>
        </div>""", unsafe_allow_html=True)
        detect_btn = False


# ── COL 2: Results ──
with col2:
    st.markdown('<div class="step-tag">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Detection Results</div>', unsafe_allow_html=True)

    if uploaded and detect_btn:
        with st.spinner("Analyzing image..."):
            annotated, results = detect_image(img)

        st.image(annotated, caption="Detection Output", use_container_width=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        boxes = results.boxes
        if boxes and len(boxes) > 0:
            total = len(boxes)
            cls_set, total_c = set(), 0.0
            for b in boxes:
                cls_set.add(model.names[int(b.cls.cpu())])
                total_c += float(b.conf.cpu())
            avg_c = total_c / total

            st.markdown(f"""
            <div class='stats-row'>
                <div class='stat-b'><div class='stat-n'>{total}</div><div class='stat-l'>Objects</div></div>
                <div class='stat-b'><div class='stat-n'>{len(cls_set)}</div><div class='stat-l'>Categories</div></div>
                <div class='stat-b'><div class='stat-n'>{avg_c:.0%}</div><div class='stat-l'>Avg Conf</div></div>
            </div>""", unsafe_allow_html=True)

            for b in boxes:
                name = model.names[int(b.cls.cpu())]
                cv   = float(b.conf.cpu())
                if cv > 0.70:
                    clr = "#00c8b4"; bg = "rgba(0,200,180,0.08)"
                elif cv > 0.45:
                    clr = "#ffb347"; bg = "rgba(255,179,71,0.08)"
                else:
                    clr = "#ff6b6b"; bg = "rgba(255,107,107,0.08)"

                st.markdown(f"""
                <div class='det-row' style='background:{bg}; border-left-color:{clr};'>
                    <span class='det-name'>
                        <span class='det-dot' style='background:{clr};'></span>
                        {name}
                    </span>
                    <span class='det-badge' style='color:{clr}; border-color:{clr}40; background:{clr}15;'>{cv:.0%}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='empty-box' style='min-height:200px;'>
                <div class='empty-ring'>✅</div>
                <div class='empty-msg'>No garbage detected in this image!</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='empty-box'>
            <div class='empty-ring' style='font-size:2.2rem; border-color:rgba(255,90,31,0.2);'>🎯</div>
            <div class='empty-msg'>Upload an image and click Detect<br>to see results here</div>
        </div>""", unsafe_allow_html=True)


