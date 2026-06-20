"""Streamlit app for handwritten text recognition."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import SAVED_MODELS_DIR  # noqa: E402
from predict import predict_image  # noqa: E402


st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="📝",
    layout="wide",
)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Decode a Streamlit upload as an OpenCV BGR image."""

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def main() -> None:
    st.title("Handwritten Text Recognition Using Deep Learning")
    st.caption("CRNN + BiLSTM + CTC with uploaded-photo preprocessing")

    default_checkpoint = SAVED_MODELS_DIR / "best_crnn_ctc.pt"
    with st.sidebar:
        st.header("Inference")
        checkpoint_path = Path(st.text_input("Checkpoint path", value=str(default_checkpoint)))
        use_advanced = st.checkbox("Use photo cleanup preprocessing", value=True)

    uploaded_file = st.file_uploader(
        "Upload Handwritten Text Image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    if uploaded_file is None:
        st.info(
            """
    📌 For best results:

    • Use plain white paper (avoid ruled notebook pages)

    • Write clearly using dark ink

    • Keep handwriting reasonably large

    • Upload a well-lit image with minimal shadows

    • Best suited for handwritten words and short sentences
    """
        )
        return

    image = read_uploaded_image(uploaded_file)
    st.subheader("Original Upload")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    if not checkpoint_path.exists():
        st.error("No trained checkpoint found. Train the model first or provide a valid checkpoint path.")
        st.code("python src/train.py --dataset-dir dataset/huggingface_iam", language="bash")
        return

    with st.spinner("Preprocessing image and recognizing handwriting..."):
        result = predict_image(
            image,
            checkpoint_path=checkpoint_path,
            use_advanced_preprocessing=use_advanced,
        )

    st.subheader("Preprocessing Inspection")
    stages = result["preprocessing_stages"]
    columns = st.columns(4)
    for column, stage_name in zip(columns, ["grayscale", "contrast_enhanced", "thresholded", "final"]):
        with column:
            st.caption(stage_name.replace("_", " ").title())
            st.image(stages[stage_name], clamp=True, use_container_width=True)

    st.subheader("Recognized Text")
    recognized_text = result["text"]
    st.text_area("Copyable output", value=recognized_text, height=110)
    st.metric("Confidence score", f"{result['confidence'] * 100:.2f}%")
    st.download_button(
        label="Download recognized text",
        data=recognized_text,
        file_name="recognized_text.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
