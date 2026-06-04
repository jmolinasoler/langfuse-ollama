"""
Featurebase Feedback Integration
Uses Featurebase SDK for anonymous feedback submission.
"""

import streamlit as st

FEATUREBASE_APP_ID = "6a1dad4c4a4eaeea48c79706"
SDK_VERSION = "1.0.3"
SDK_URL = f"https://esm.sh/featurebase-js@{SDK_VERSION}"


def _get_boot_script() -> str:
    """Generate Featurebase SDK boot script."""
    return f"""
<script type="module">
  if (window.__fbBooted) {{ return; }}
  window.__fbBooted = true;
  import Featurebase from '{SDK_URL}';
  Featurebase({{ appId: '{FEATUREBASE_APP_ID}' }});
</script>
"""


def render_feedback(disabled: bool = False) -> None:
    """
    Render feedback interface with textarea and Featurebase SDK integration.
    
    Args:
        disabled: If True, disable feedback functionality
    """
    if disabled:
        return

    st.markdown("## 💬 Send Feedback")
    st.markdown("Share your thoughts anonymously. Your feedback helps improve this tool.")
    st.markdown("---")

    if st.button("← Back to App", key="fb_back"):
        st.session_state.show_feedback = False
        st.rerun()

    st.markdown("---")
    
    feedback_text = st.text_area(
        "Your feedback",
        placeholder="What would you like to share? Bug reports, suggestions, or general comments...",
        height=200,
        key="feedback_text_input"
    )
    
    st.markdown("---")
    
    if st.button("📤 Send Feedback", type="primary", disabled=not bool(feedback_text.strip())):
        if not feedback_text.strip():
            st.error("Please enter your feedback before submitting.")
        else:
            st.session_state.fb_loaded = True
            st.rerun()
    
    if st.session_state.get("fb_loaded"):
        st.markdown("✅ Loading feedback panel...")
        st.markdown(_get_boot_script(), unsafe_allow_html=True)


def main():
    """Legacy main function for backward compatibility."""
    render_feedback()


if __name__ == "__main__":
    main()
