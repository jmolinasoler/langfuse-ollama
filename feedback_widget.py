"""
Featurebase feedback widget — Camino B (CDN ESM, anónimo, lazy).

Carga el SDK moderno featurebase-js@1.0.3 desde esm.sh y lo inicializa en
modo anónimo (sin JWT, sin identify). El SDK lee los toggles de
"General → Manage modules" del workspace de Featurebase y decide qué
superficies arrancar (feedback / messenger / changelog / surveys).

Diseño:
  - Streamlit no tiene bundler, así que NO podemos usar
    `featurebase-js/react`. Cargamos el default export por ESM CDN y lo
    llamamos como Featurebase({ appId }).
  - El SDK monta el botón flotante en `document.body`. Si lo arrancamos
    desde dentro de un iframe de components.html, queda atrapado allí.
    Por eso inyectamos un <script type="module"> en
    `window.parent.document.head` — same-origin con la página padre,
    el botón flota sobre la app Streamlit real.
  - Idempotencia: Streamlit re-ejecuta el script en cada interacción.
    Usamos `window.parent.__featurebaseBooted` como flag para no
    bootear dos veces.

Modelo de seguridad (acordado en la fase de diseño):
  - `FEATUREBASE_APP_ID` es público (como un Stripe publishable key) →
    hardcodeado en este módulo, safe para distribuir.
  - Sin JWT secret en cliente → submissions anónimas. Featurebase
    aplica su propia moderación / antispam server-side.
  - Carga lazy: el SDK no se descarga hasta que el usuario pulsa el
    botón. Si nunca lo pulsa, ningún script de terceros toca la página.
  - Versión del SDK pineada a `1.0.3`. SRI sobre el grafo ESM de esm.sh
    no es viable (los submódulos cargan dinámicamente y el browser solo
    valida `integrity=` en el <script> raíz). Mitigación: pin de versión
    + opción de opt-out vía FEEDBACK_DISABLED.
  - Metadata: NO se envía nada de la sesión Streamlit. Featurebase solo
    ve lo que el usuario escriba en su formulario. Sin prompts, sin
    respuestas, sin claves Langfuse, sin user_id del sidebar.
"""

import streamlit as st
import streamlit.components.v1 as components

# Identificador público del workspace Featurebase (NO es un secreto).
FEATUREBASE_APP_ID = "6a1dad4c4a4eaeea48c79706"

# Versión pineada — actualizar a mano tras verificar release notes.
SDK_VERSION = "1.0.3"
SDK_URL = f"https://esm.sh/featurebase-js@{SDK_VERSION}"


def _boot_snippet() -> str:
    """
    HTML que se renderiza dentro de un iframe de components.html.
    Inyecta un <script type="module"> en el documento padre para que el
    SDK arranque allí (no dentro del iframe height=0).
    """
    return f"""
<script>
  (function () {{
    var parentWin = window.parent;
    var parentDoc = parentWin.document;

    // Idempotencia entre reruns de Streamlit.
    if (parentWin.__featurebaseBooted) return;
    parentWin.__featurebaseBooted = true;

    var s = parentDoc.createElement('script');
    s.type = 'module';
    s.textContent = [
      "import Featurebase from '{SDK_URL}';",
      "window.Featurebase = Featurebase;",
      "Featurebase({{ appId: '{FEATUREBASE_APP_ID}' }});"
    ].join('\\n');
    parentDoc.head.appendChild(s);
  }})();
</script>
"""


def render(disabled: bool = False) -> None:
    """
    Monta el toggle de feedback en el contenedor Streamlit actual.

    Args:
        disabled: si True, no hace nada (opt-out vía FEEDBACK_DISABLED).

    Comportamiento:
        - Muestra un botón "💬 Send feedback".
        - Al pulsarlo por primera vez, descarga e inicializa
          featurebase-js. A partir de ese momento aparece el launcher
          flotante de Featurebase (esquina inferior derecha por defecto,
          configurable en el dashboard).
        - Reruns posteriores de Streamlit NO re-bootean el SDK
          (idempotencia controlada por window.parent.__featurebaseBooted).
    """
    if disabled:
        return

    if "fb_widget_loaded" not in st.session_state:
        st.session_state.fb_widget_loaded = False

    label = "💬 Send feedback" if not st.session_state.fb_widget_loaded else "💬 Feedback loaded"
    help_text = (
        "Loads featurebase-js (anonymous). Nothing leaves the page until "
        "you actually submit the form."
    )

    if st.button(label, help=help_text, disabled=st.session_state.fb_widget_loaded):
        st.session_state.fb_widget_loaded = True
        st.rerun()

    if st.session_state.fb_widget_loaded:
        components.html(_boot_snippet(), height=0)
        st.caption(
            "Feedback panel served by featurebase.app. Only what you type "
            "in the form is sent — no prompts, responses or API keys."
        )
