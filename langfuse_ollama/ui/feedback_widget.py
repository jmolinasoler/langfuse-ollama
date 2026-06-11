"""
Featurebase Feedback Integration — boot anónimo del SDK (Fase 1).

El SDK se carga como módulo ES desde esm.sh y tiene que montarse en el
documento PADRE (components.html renderiza dentro de un iframe height=0,
donde el widget no sería visible). `st.markdown` no sirve: el HTML que
inserta no ejecuta <script>.
"""

import streamlit as st
import streamlit.components.v1 as components

FEATUREBASE_APP_ID = "6a1dad4c4a4eaeea48c79706"
SDK_VERSION = "1.0.3"
SDK_URL = f"https://esm.sh/featurebase-js@{SDK_VERSION}"


def _boot_snippet() -> str:
    """Script que inyecta el boot del SDK Featurebase en el documento padre."""
    return f"""
<script>
(function () {{
  var parentWin = window.parent;
  var parentDoc = parentWin.document;
  if (parentWin.__featurebaseBooted) return;
  parentWin.__featurebaseBooted = true;
  var s = parentDoc.createElement('script');
  s.type = 'module';
  s.textContent = "import Featurebase from '{SDK_URL}'; " +
                  "Featurebase({{ appId: '{FEATUREBASE_APP_ID}' }});";
  parentDoc.head.appendChild(s);
}})();
</script>
"""


def render_feedback(disabled: bool = False) -> None:
    """
    Vista de feedback: arranca el widget de Featurebase (que gestiona el
    formulario y el envío) y ofrece volver a la app.
    """
    if disabled:
        st.info("Feedback is disabled (FEEDBACK_DISABLED).")
        if st.button("← Back to App", key="fb_back"):
            st.session_state.show_feedback = False
            st.rerun()
        return

    st.markdown("## 💬 Send Feedback")
    st.markdown(
        "The Featurebase widget loads in this page — use it to share bug "
        "reports, suggestions or general comments anonymously."
    )

    components.html(_boot_snippet(), height=0)

    st.markdown("---")
    if st.button("← Back to App", key="fb_back"):
        st.session_state.show_feedback = False
        st.rerun()
