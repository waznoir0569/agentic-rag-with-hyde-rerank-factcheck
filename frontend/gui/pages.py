import api_utils
import streamlit as st
from chat_components import authenticated_user_chat_interface_component, unauthenticated_user_chat_interface_component
from state_management import Page, authenticate_user


LIKERT_OPTIONS = {
    1: "1 - Strongly disagree",
    2: "2 - Disagree",
    3: "3 - Neither agree nor disagree",
    4: "4 - Agree",
    5: "5 - Strongly agree",
}

QUESTIONNAIRE = [
    ("debtrules3", "Taking on debt makes me feel uncomfortable."),
    ("debtnorms8", "People close to me generally believe that borrowing money is risky."),
    ("debtrules16", "I carefully evaluate the risks before making a financial decision."),
    ("debtpers5", "I prefer a safer financial option, even when it may offer less potential benefit."),
    ("debtnorms9", "Advice from people I trust strongly influences my financial decisions."),
    ("debtpers9", "A possible financial loss worries me more than an equal-sized gain excites me."),
    ("debtpers11", "I am comfortable waiting for a better long-term financial outcome."),
    ("debtrules9", "I prefer to avoid borrowing unless repayment is clearly manageable."),
]


@st.dialog("A few questions before we begin")
def questionnaire_dialog():
    st.write("There are no right or wrong answers. Choose the response that feels most like you.")
    st.caption("Your answers help tailor explanations and are not used to approve or deny a loan.")

    with st.form("persona_questionnaire"):
        answers = {}
        for field_name, question in QUESTIONNAIRE:
            selected = st.radio(
                question,
                options=list(LIKERT_OPTIONS),
                format_func=lambda value: LIKERT_OPTIONS[value],
                key=f"questionnaire_{field_name}",
                horizontal=True,
            )
            answers[field_name] = selected

        submitted = st.form_submit_button("Continue to chat", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Preparing your personalized assistant..."):
            profile = api_utils.compute_persona(answers)

        if not profile:
            st.error("We could not prepare your preferences. Please try again.")
            return

        st.session_state["persona_profile"] = profile
        st.session_state["persona_prompt"] = profile.get("persona_prompt", "")
        st.session_state["questionnaire_completed"] = True
        st.rerun()


def home_page():
    if not st.session_state["questionnaire_completed"]:
        questionnaire_dialog()
        st.stop()

    st.title("💬 AI-Powered Financial Assistant")
    if not st.session_state["user"].is_authenticated:
        st.markdown(
            """
            Welcome! This intelligent assistant uses advanced Retrieval-Augmented Generation (RAG) to help you find answers and understand your documents.

            ### ✨ Key Features for Logged-in Users:
            - Ask questions and receive smart, context-aware answers.
            - Upload your own documents for tailored assistance.
            - Use tavily search for general knowledge questions.
            - Enjoy **chat memory** to remember and recall previous conversations.
            """
        )
        st.subheader("💡 What would you like to ask?", divider="rainbow")
        unauthenticated_user_chat_interface_component()
    else:
        st.subheader("💡 What would you like to ask?", divider="rainbow")
        authenticated_user_chat_interface_component()


def login_page():
    st.subheader("🔐 Login", divider="rainbow")
    with st.form("login_form"):
        email = st.text_input("Email *")
        password = st.text_input("Password *", type="password")
        submitted = st.form_submit_button("Login")

    back_to_home_component()

    if submitted:
        with st.spinner("Logging in..."):
            login_response = api_utils.login_user(email, password)
            if message := login_response.get("message"):
                st.success(message)
                st.session_state["page"] = Page.HOME
                authenticate_user(login_response)
                st.rerun()
            else:
                st.error(login_response.get("detail", "Login failed. Please try again."))


def register_page():
    st.subheader("✍ Register", divider="rainbow")
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        email = col1.text_input("Email *")
        username = col1.text_input("Username *", max_chars=16)
        password = col1.text_input("Password *", type="password", max_chars=32)
        first_name = col2.text_input("First Name", max_chars=50)
        last_name = col2.text_input("Last Name", max_chars=50)

        submitted = st.form_submit_button("Register")

    back_to_home_component()

    if submitted:
        register_data = {
            "email": email,
            "username": username,
            "password": password,
            "first_name": first_name,
            "last_name": last_name,
        }
        with st.spinner("Registering..."):
            register_response = api_utils.register_user(register_data)
            if message := register_response.get("message"):
                st.success(message)
                st.session_state["page"] = Page.LOGIN
                st.rerun()
            else:
                st.error(register_response.get("detail", "Registration failed. Please try again."))


def back_to_home_component():
    if st.button("⬅️ Back to Home", type="tertiary"):
        st.session_state["page"] = Page.HOME
        st.rerun()
