import hashlib
import io
import os
import sys
import threading
from pathlib import Path

import pillow_heif
import streamlit as st

# iCloud can hide the .pth file of an editable install, and Python skips hidden .pth files.
# Adding the source folder here makes the app work either way.
CLASSIFIER_SRC = str(Path(__file__).resolve().parent / "electronics_classifier" / "src")
if CLASSIFIER_SRC not in sys.path:
    sys.path.insert(0, CLASSIFIER_SRC)

from electronics_classifier import ElectronicsClassificationPipeline
from electronics_classifier.cli import FAKE_SETS, InputError, load_image
from electronics_classifier.config import ConfigError
from ewaste_locator.find_recyclers import (
    RADIUS_MILES,
    LocatorError,
    find_recyclers,
    normalize_zip,
)

pillow_heif.register_heif_opener()  # lets Pillow (and so load_image) read iPhone HEIC photos

st.set_page_config(page_title="NavE-Waste", layout="wide")

DEVICE_TYPES = [
    "Phone", "Laptop", "Tablet", "Television", "Router", "Hard Drive", "Smartwatch", "Modem", "Smart Speaker",
    "Remote Control", "Game Console", "Monitor", "Webcam", "Camera"
]

# Classifier category -> DEVICE_TYPES entry. Categories that are not listed here (charger, cable,
# "electronic device", ...) have no match, so the user picks the device.
CATEGORY_TO_DEVICE = {
    "smartphone": "Phone",
    "laptop": "Laptop",
    "tablet": "Tablet",
    "television": "Television",
    "router": "Router",
    "modem": "Modem",
    "smart speaker": "Smart Speaker",
    "remote control": "Remote Control",
    "game console": "Game Console",
    "monitor": "Monitor",
    "webcam": "Webcam",
}

WARNING_TEXT = {
    "CATEGORY_MISMATCH": "The two models disagree about this device. Please check the type.",
    "LOW_DETECTOR_CONFIDENCE": "Low confidence. Please check the type.",
    "VLM_PARSE_FAILED": "The device details could not be read.",
    "MODEL_UNVERIFIED": "A model name was suggested but could not be confirmed in the photo.",
    "SMALL_CROP": "The device is small in the photo. A closer photo may work better.",
}

USE_FAKES = os.environ.get("EC_USE_FAKES") == "1"
RAM_CHECKPOINT = Path(CLASSIFIER_SRC).parent / "checkpoints" / "ram_plus_swin_large_14m.pth"


@st.cache_resource(show_spinner=False)
def get_classifier():
    """Load the models once and share them. The lock keeps it to one photo at a time."""
    # No preset: the built-in config is the Grounded-SAM + RAM++ pipeline. The old
    # vocabulary-prompted Grounding DINO pipeline is still available as presets=["baseline"].
    if USE_FAKES:
        sets = list(FAKE_SETS)
    else:
        # The config's checkpoint path is relative to the working directory, and Streamlit is
        # started from the repo root, so point at the file absolutely.
        sets = [f"pipeline.detector.params.ram_checkpoint={RAM_CHECKPOINT}"]
    return ElectronicsClassificationPipeline(sets=sets), threading.Lock()


# Keyed on the photo, so a rerun during a slow classification waits for it instead of repeating it.
@st.cache_data(show_spinner=False, max_entries=8)
def classify(photo: bytes):
    """Return the annotated image and the detected items, most confident first."""
    image = load_image(io.BytesIO(photo))
    pipeline, lock = get_classifier()
    with lock:
        annotated, result = pipeline.run(image)
    items = [item.model_dump(by_alias=True) for item in result.items]
    items.sort(key=lambda item: item["detector_confidence"], reverse=True)
    return annotated, items


def describe(item: dict) -> str:
    attrs = item["attributes"]
    known = [attrs[k] for k in ("color", "brand", "model") if attrs[k] != "unknown"]
    return " ".join([*known, attrs["category"]])


def run_pipeline(device: str, zip_code: str) -> dict:
    #add disposal methods/wipe steps etc
    # A failed search only affects the drop-off tab; the rest of the plan is still shown.
    dropoffs, dropoff_error = [], None
    try:
        dropoffs = find_recyclers(zip_code)
    except LocatorError as exc:
        dropoff_error = str(exc)
    return {
        "device": device,
        "zip_code": zip_code,
        "warnings": [
            "WARNINGS",
        ],
        "wipe_steps": [
            "GET WIPE STEPS",
        ],
        "source_url": "https://example.com/maker-guide",
        "last_checked": "2026-11-01",
        "dropoffs": dropoffs,
        "dropoff_error": dropoff_error,
    }

st.title("NavE-Waste")
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
        file = st.file_uploader("Device photo", type=["jpg", "jpeg", "png", "heic"])
    else:
        file = st.camera_input("Take a photo")
 
    photo = file.getvalue() if file else None
    image = None
    if photo:
        try:
            image = load_image(io.BytesIO(photo))
            st.image(image, width="stretch")
        except InputError:
            st.error("This photo could not be read. Please use a JPG, PNG or HEIC.")
 
with right:
    st.subheader("2. Confirm the device")
    if image is None:
        st.info("Add a photo to get started.")
    else:
        if st.session_state.get("photo") != photo:
            st.session_state.photo = photo
            st.session_state.pop("result", None)  # the old plan was for another photo

        # Widget keys include the photo, so a choice never carries over to the next photo.
        photo_id = hashlib.sha256(photo).hexdigest()[:12]
        try:
            with st.spinner("Identifying device... this can take several minutes."):
                annotated, items = classify(photo)
        except (ConfigError, ImportError, OSError) as exc:
            st.error(f"The classifier could not start: {exc}")
            annotated, items = None, []
        else:
            if not items:
                st.warning("No device was found. Try a closer photo, or pick the device below.")

        if USE_FAKES:
            st.warning("EC_USE_FAKES is on: these results are fake.")

        item = None
        if items:
            item = items[0]
            if len(items) > 1:
                number = st.selectbox(
                    f"{len(items)} possible devices found. Which one is it?",
                    range(len(items)),
                    format_func=lambda i: f"{items[i]['id']}: {describe(items[i])}",
                    key=f"item-{photo_id}",
                )
                item = items[number]
            st.image(annotated, width="stretch")
            st.write(f"Detected: **{describe(item)}** "
                     f"({item['detector_confidence']:.0%} detection confidence)")
            for code in item["warnings"]:
                if code in WARNING_TEXT:
                    st.caption(WARNING_TEXT[code])

        detected = CATEGORY_TO_DEVICE.get(item["attributes"]["category"]) if item else None
        device = st.selectbox(
            "Not right? Pick the correct device:" if detected else "Pick your device:",
            DEVICE_TYPES,
            index=DEVICE_TYPES.index(detected) if detected else None,
            placeholder="Choose a device",
            key=f"device-{photo_id}-{item['id'] if item else 'none'}",
        )

        st.header("Your location")
        zip_code = st.text_input("ZIP code", max_chars=5, placeholder="12345")
 
        if st.button("Get disposal plan", type="primary", disabled=not (zip_code and device)):
            st.session_state.pop("result", None)  # do not keep showing the plan for another ZIP
            try:
                zip_code = normalize_zip(zip_code)
            except LocatorError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Finding drop-off sites..."):
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
        dropoffs = result["dropoffs"]
        area = f"within {RADIUS_MILES} miles of {result['zip_code']}"
        if result["dropoff_error"]:
            st.error(result["dropoff_error"])
        elif not dropoffs:
            st.info(f"No EPA-listed e-waste recyclers were found {area}.")
        else:
            noun = "recycler" if len(dropoffs) == 1 else "recyclers"
            st.write(f"{len(dropoffs)} e-waste {noun} {area}, nearest first:")
            if not all(d["dropoff_verified"] for d in dropoffs):
                st.warning(
                    "Public drop-off is not confirmed for every site, and some may only serve "
                    "businesses. Contact a site before visiting."
                )
        for d in dropoffs:
            with st.container(border=True):
                st.markdown(f"**{d['name']}**")
                st.write(f"{d['address']} · {d['distance_mi']} mi")
                contact = [d["phone"], d["email"], f"[Website]({d['url']})" if d["url"] else ""]
                contact = [c for c in contact if c]
                if contact:
                    st.markdown(" · ".join(contact))
        if dropoffs:
            st.caption("Waste facility location data sourced through Esri ArcGIS REST Services. " \
                       "Original data publishers are credited where available. " \
                       "Distances are straight-line miles from the center of "
                       "the ZIP code. Verify hours before visiting.")
