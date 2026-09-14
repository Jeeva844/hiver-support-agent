"""Streamlit UI for the SpotifyCares support agent.

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import Pipeline  # noqa: E402

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_set.csv"


@st.cache_resource(show_spinner="Loading agent pipeline...")
def load_pipeline() -> Pipeline:
    return Pipeline()


@st.cache_data(show_spinner=False)
def load_golden() -> pd.DataFrame:
    return pd.read_csv(GOLDEN_PATH)


def main() -> None:
    st.set_page_config(page_title="SpotifyCares Support Agent", layout="wide")
    st.title("SpotifyCares AI Support Agent")
    st.caption(
        "Intent classification -> escalate-or-resolve -> grounded reply. "
        "Project: github.com/Jeeva844/hiver-support-agent"
    )

    pipe = load_pipeline()
    st.sidebar.title("Agent")
    st.sidebar.write(f"Classifier backend: **{pipe.classifier.backend}**")
    st.sidebar.write(f"Retrieval index docs: **{pipe.retriever.config['n_documents']}**")
    st.sidebar.markdown("Docs: `REPORT.md`, `results/misleading_metrics.md`.")

    tab_run, tab_golden = st.tabs(["Run the agent", "Golden-set explorer"])

    with tab_run:
        samples = {
            "My account got hacked and someone changed my email!": "security scenario",
            "I was charged twice for premium, I want a refund": "billing scenario",
            "Why does shuffle keep playing songs I don't like?": "playback scenario",
            "How do I get premium for 3 months free?": "subscription scenario",
            "Thanks, that worked!": "appreciation scenario",
            "Custom message below...": "custom",
        }
        picked = st.selectbox("Try an example", list(samples))
        text = st.text_area(
            "Customer message",
            value="" if picked.startswith("Custom") else picked,
            height=90,
        )
        run = st.button("Run agent", type="primary")

        if run and text.strip():
            result = pipe.run(text)
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", result["intent"])
            col2.metric("Confidence", f"{result['intent_confidence']:.2f}")
            col3.metric(
                "Escalate to human",
                "Yes" if result["escalate"] else "No",
            )
            st.info(f"Classification: {result['intent_reason']}")
            if result["escalate"]:
                st.warning(f"Escalated to a human agent. Reason: {result['escalation_reason']}")
            else:
                st.subheader("Draft reply")
                st.success(result["reply"] or "(no reply generated)")
                if result["retrieval"]:
                    r = result["retrieval"]
                    with st.expander("Retrieval provenance"):
                        st.write(f"Similarity: {r['similarity']:.3f}")
                        st.write(f"Source conversation: {r['source_conversation']}")
                        st.write(f"Source intent: {r['source_intent']}")
                        st.write(f"Source message: _{r['source_customer_text']}_")
            with st.expander("Full pipeline output (JSON)"):
                st.json(result)

    with tab_golden:
        golden = load_golden()
        st.subheader("200 hand-reviewed test messages")
        st.write(
            "You can test the agent against human-reviewed cases. "
            "A check means the agent matched the human label."
        )
        intent_filter = st.selectbox(
            "Filter by human intent", ["all"] + sorted(golden["intent"].unique())
        )
        view = golden if intent_filter == "all" else golden[golden["intent"] == intent_filter]
        ids = view["id"].tolist()
        pick = st.selectbox("Pick an example", ids)
        row = golden[golden["id"] == pick].iloc[0]

        st.write(f"**Message:** {row['text']}")
        st.write(
            f"Human label: **{row['intent']}** | "
            f"Human escalation: **{'Yes' if row['should_escalate'] else 'No'}**"
        )
        if row["human_notes"]:
            st.caption(f"Reviewer note: {row['human_notes']}")

        if st.button("Run agent on this example", key="golden_run"):
            result = pipe.run(row["text"])
            ok_intent = "✓" if result["intent"] == row["intent"] else "✗"
            ok_esc = "✓" if result["escalate"] == bool(row["should_escalate"]) else "✗"
            c1, c2 = st.columns(2)
            c1.metric(f"Intent {ok_intent}", f"{result['intent']} (human: {row['intent']})")
            c2.metric(
                f"Escalation {ok_esc}",
                "Yes" if result["escalate"] else "No",
            )
            if not result["escalate"] and result["reply"]:
                st.write("**Draft reply:**")
                st.success(result["reply"])


if __name__ == "__main__":
    main()