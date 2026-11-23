import streamlit as st
from PIL import Image

st.set_page_config(page_title="E-NaviWaste", layout="wide")

DEVICE_TYPES = [
    "Phone", "Laptop", "Tablet", "Television", "Router", "Hard Drive", "Smartwatch",
]
 
 
#replace with real calls
def detect_device(image: Image.Image):
    #fake
    return "phone", 1
 
 
def run_pipeline(device: str, zip_code: str) -> dict:
    #add disposal methods/wipe steps etc
    return {
        "device": device,
        "warnings": [
            "WARNINGS",
        ],
        "wipe_steps": [
            "GET WIPE STEPS",
        ],
        "source_url": "https://example.com/maker-guide",
        "last_checked": "2026-11-01",
        "dropoffs": [
            {"name": "NAME", "address": "ADDRESS", "distance_mi": "DISTANCE",
            "url": "URL"},
        ],
    }

st.title("E-NaviWaste")
st.caption("Upload an image of your old device to get safe wipe steps and nearby drop-off sites")

with st.sidebar:
    st.title("ADDITIONAL INFO")
    st.divider()
    st.caption("General guidance only. Confirm steps with the manufacturer.")

left, right = st.columns([1, 1], gap="large")
 
with left:
    st.subheader("1. Add a photo")
    source = st.radio("Input", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
    if source == "Upload":
        file = st.file_uploader("Device photo", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("Take a photo")
 
    image = Image.open(file) if file else None
    if image:
        st.image(image, use_container_width=True)
 
with right:
    st.subheader("2. Confirm the device")
    if image is None:
        st.info("Add a photo to get started.")
    else:
        key = f"{file.name}-{file.size}"
        if st.session_state.get("detect_key") != key:
            with st.spinner("Identifying device..."):
                st.session_state.detected, st.session_state.conf = detect_device(image)
                st.session_state.detect_key = key
 
        detected = st.session_state.detected
        st.write(f"Detected: **{detected}** ({st.session_state.conf:.0%} confidence)")
        device = st.selectbox(
            "Not right? Pick the correct device:",
            DEVICE_TYPES,
            index=DEVICE_TYPES.index(detected) if detected in DEVICE_TYPES else 0,
        )

        st.header("Your location")
        zip_code = st.text_input("ZIP code", max_chars=5, placeholder="12345")
 
        if st.button("Get disposal plan", type="primary", disabled=not zip_code):
            st.session_state.result = run_pipeline(device, zip_code)
        if not zip_code:
            st.caption("Enter a ZIP code to continue.")
 
result = st.session_state.get("result")
if result and image is not None:
    st.divider()
    st.header(f"Next Steps")
 
    for w in result["warnings"]:
        st.warning(w)
 
    tab_wipe, tab_drop = st.tabs(["Wipe your data", "Where to drop it off"])
 
    with tab_wipe:
        for i, step in enumerate(result["wipe_steps"], 1):
            st.markdown(f"**{i}.** {step}")
    # can add caption with source
 
    with tab_drop:
        for d in result["dropoffs"]:
            with st.container(border=True):
                st.markdown(f"**{d['name']}**")
                st.write(f"{d['address']} · {d['distance_mi']} mi")
                st.markdown(f"[Website]({d['url']})")
        st.caption("Verify hours before visiting.")

