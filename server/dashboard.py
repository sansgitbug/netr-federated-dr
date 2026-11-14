"""Dashboard for the federated DR demo. Run from the project root: streamlit run server/dashboard.py"""
import hashlib
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from PIL import Image

from netra import config
from netra.io import read_json
from netra.model import build_model, decode, preprocess

REPLAY = os.environ.get("NETRA_REPLAY") == "1" or (
    not config.LOGS_PATH.exists() and config.REPLAY_LOG_PATH.exists()
)
LOGS_PATH = config.REPLAY_LOG_PATH if REPLAY else config.LOGS_PATH
MODEL_PATH = (config.ROOT / "demo" / "global_model.weights.h5") if REPLAY else config.MODEL_PATH
EXAMPLES_DIR = config.ROOT / "examples"
REFRESH_MS = 2000

st.set_page_config(layout="wide", page_title="Federated DR Screening")
data = read_json(LOGS_PATH, {}) or {}

# Only poll while a run is live. Reruns after training would wipe an in-progress upload.
finished = data.get("status") in ("Training complete", "Stopped before all rounds finished")
if not REPLAY and not finished:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=REFRESH_MS, key="data_refresher")


def model_version() -> float:
    try:
        return MODEL_PATH.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading model...")
def load_model(version: float):
    if version == 0.0:
        return None
    try:
        model = build_model(pretrained=False)
        model.load_weights(str(MODEL_PATH))
        return model
    except Exception as e:
        print(f"Could not load model: {e}")
        return None


def run_prediction(model, image_bytes):
    batch = preprocess(Image.open(io.BytesIO(image_bytes)))
    return model.predict(batch, verbose=0)[0]


rounds = data.get("rounds", [])
last = rounds[-1] if rounds else None
required = data.get("clients_required", config.CLIENTS_REQUIRED)
connected = data.get("connected_clients", 0)

c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
with c1:
    st.title("Federated DR screening")
    status = data.get("status", "Waiting for the server to start")
    st.markdown(f"Status: `{status}`" + (" (replay of a recorded run)" if REPLAY else ""))
c2.metric("Accuracy", f"{last['global_accuracy'] * 100:.1f}%" if last else "n/a")
c3.metric("Quadratic kappa", f"{last['kappa']:.3f}" if last else "n/a")
c4.metric("Referable sensitivity", f"{last['referable_sensitivity'] * 100:.1f}%" if last else "n/a")
st.caption("Research demo. Not a diagnostic device and not clinically validated.")
st.divider()

col_info, col_chart = st.columns([1, 2])
with col_info:
    st.header("Session")
    with st.container(border=True):
        st.metric("Connected hospitals", f"{connected} / {required}")
        st.metric("Round", f"{data.get('current_round', 0)} / {data.get('total_rounds', 'n/a')}")
        st.markdown("Only model weights are shared with the server. Scans stay with each hospital.")
    if MODEL_PATH.exists():
        st.download_button("Download model weights", MODEL_PATH.read_bytes(),
                           "global_model.weights.h5", use_container_width=True)

with col_chart:
    st.header("Global model performance")
    if not rounds:
        st.info("Waiting for the first round to finish.")
    else:
        df = pd.DataFrame(rounds).set_index("round")
        tab_overall, tab_hospital = st.tabs(["Accuracy and kappa", "By hospital"])
        with tab_overall:
            st.line_chart(df[["global_accuracy", "kappa"]].rename(
                columns={"global_accuracy": "Accuracy", "kappa": "Quadratic kappa"}), height=300)
        with tab_hospital:
            rows = [{"Hospital": h, "Train scans": m.get("examples"),
                     "Local train accuracy": m.get("train_accuracy"),
                     "Global model accuracy on local test": m.get("accuracy"),
                     "Kappa": m.get("kappa")} for h, m in last["client_metrics"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("Raw log"):
    st.json(data)

st.divider()

st.header("Grade a retinal scan")
model = load_model(model_version())
if model is None:
    st.warning("There is no trained model yet. Start the server and wait for the first round.")
else:
    manifest = read_json(EXAMPLES_DIR / "manifest.json", []) or []
    left, right = st.columns(2)
    with left:
        st.caption("Please don't upload real patient images. Uploads are processed in memory "
                   "and are not saved.")
        up = st.file_uploader("Upload a fundus image (JPG or PNG)", type=["jpg", "jpeg", "png"])
        if manifest:
            st.markdown("Or load an example: five APTOS 2019 scans that were held out of training, "
                        "picked at random, one per grade.")
            for col, ex in zip(st.columns(len(manifest)), manifest):
                if col.button(f"Grade {ex['true_grade']}", key=f"ex{ex['true_grade']}",
                              use_container_width=True):
                    st.session_state["example"] = ex

    raw, truth, caption = None, None, ""
    if up is not None:
        raw, caption = up.getvalue(), "Uploaded scan"
    elif st.session_state.get("example"):
        ex = st.session_state["example"]
        raw, truth = (EXAMPLES_DIR / ex["file"]).read_bytes(), ex["true_grade"]
        caption = f"APTOS example, dataset label: grade {truth} ({config.CLASS_NAMES[truth]})"

    if raw is not None:
        with left:
            st.image(raw, caption=caption, use_column_width=True)

        # Cached by image hash and model version so a rerun doesn't repeat inference.
        key = (hashlib.sha1(raw).hexdigest(), model_version())
        if st.session_state.get("pred_key") != key:
            with st.spinner("Running the model..."):
                try:
                    st.session_state["pred"] = run_prediction(model, raw)
                except Exception as e:
                    st.session_state["pred"] = None
                    st.session_state["pred_err"] = str(e)
                st.session_state["pred_key"] = key
        probs = st.session_state.get("pred")

        with right:
            if probs is None:
                st.error(f"Could not read this image: {st.session_state.get('pred_err')}")
            else:
                grade, conf, referable = decode(probs)
                msg = f"Predicted grade {grade} ({config.CLASS_NAMES[grade]}), {conf * 100:.1f}% confidence."
                if referable:
                    st.error(msg + " Referable: this would go to a specialist.")
                elif grade == 1:
                    st.warning(msg)
                else:
                    st.success(msg)

                if truth is not None:
                    if grade == truth:
                        st.info("This matches the dataset label.")
                    else:
                        same_call = referable == (truth >= config.REFERABLE_FROM)
                        st.info(
                            f"Dataset label is grade {truth} ({config.CLASS_NAMES[truth]}). The grade is wrong, "
                            + ("but the refer or don't refer decision is the same." if same_call
                               else "and the refer or don't refer decision is the opposite.")
                        )

                st.bar_chart(pd.DataFrame({"Probability": probs}, index=config.CLASS_NAMES))
                st.caption("The confidence value is the model's own score, not the chance it is right, "
                           "and this model is often overconfident. It also doesn't check that an "
                           "image is a retinal scan.")
