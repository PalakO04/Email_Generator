import streamlit as st
from langchain_community.llms import Ollama
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from deep_translator import GoogleTranslator

st.title("Professional Email Draft Generator")

USE_OLLAMA = True  # ✅ Use local model for speed and privacy

@st.cache_resource
def load_llm():
    if USE_OLLAMA:
        return Ollama(model="mistral")
    else:
        gen_pipeline = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        return HuggingFacePipeline(pipeline=gen_pipeline)

llm = load_llm()

# 📥 Category-based templates
category = st.selectbox("Choose Email Category", [
    "Custom",
    "Leave Request",
    "Meeting Invitation",
    "Follow-Up",
    "Project Update",
    "Client Outreach",
    "General Inquiry"
])

templates = {
    "Leave Request": {
        "subject": "Leave Request for [Dates]",
        "context": "I would like to request leave from [start date] to [end date] due to [reason]."
    },
    "Meeting Invitation": {
        "subject": "Meeting Invitation: [Topic]",
        "context": "I’d like to schedule a meeting to discuss [topic]. Please let me know your availability."
    },
    "Follow-Up": {
        "subject": "Follow-Up on [Previous Topic]",
        "context": "I’m following up on our previous conversation regarding [topic]. Let me know if you need anything further."
    },
    "Project Update": {
        "subject": "Project Update: [Project Name]",
        "context": "Here’s a brief update on the progress of [project name]. We’ve completed [milestone] and are now working on [next step]."
    },
    "Client Outreach": {
        "subject": "Introduction and Services Overview",
        "context": "I’m reaching out to introduce our services and explore how we can support your goals."
    },
    "General Inquiry": {
        "subject": "Inquiry Regarding [Topic]",
        "context": "I’d like to inquire about [topic]. Could you please provide more details or point me to the right contact?"
    }
}

# Autofill subject/context if category is selected
if category != "Custom":
    subject = st.text_input("Email Subject", value=templates[category]["subject"])
    context = st.text_area("Email Details or Context", value=templates[category]["context"])
else:
    subject = st.text_input("Email Subject")
    context = st.text_area("Email Details or Context")

tone = st.selectbox("Select Tone", ["Formal", "Friendly", "Persuasive"])
language = st.selectbox("Select Output Language", ["English", "Hindi", "Gujarati", "French", "Spanish"])
sender_name = st.text_input("Your Name (optional)")

# 🚀 Generate email
if st.button("Generate Email Draft"):
    if subject and context:
        clean_context = context.strip().replace("\n", " ")

        prompt = (
            f"Write a {tone.lower()} and professional email.\n"
            f"Subject: {subject}\n"
            f"Details: {clean_context}\n"
            "The email should be polite, clear, and well-structured."
        )

        with st.spinner("Generating email..."):
            try:
                email_draft = llm.invoke(prompt)
                if isinstance(email_draft, str) and len(email_draft.strip()) > 20:
                    if sender_name:
                        email_draft += f"\n\nBest regards,\n{sender_name}"

                    # 🌍 Translate if needed
                    if language != "English":
                        try:
                            email_draft = GoogleTranslator(source='auto', target=language.lower()).translate(email_draft)
                        except Exception:
                            st.warning("Translation failed. Showing original English version.")

                    # 🖼️ Styled preview
                    st.markdown("#### ✉️ Email Preview")
                    st.markdown(f"<div style='background-color:#000000;padding:15px;border-radius:5px;white-space:pre-wrap'>{email_draft}</div>", unsafe_allow_html=True)

                    # 📋 Copy and download
                    st.code(email_draft, language="text")
                    st.download_button("Download Email as .txt", email_draft, file_name="email_draft.txt")
                else:
                    raise ValueError("Output too short or incoherent.")
            except Exception:
                fallback = f"""
Subject: {subject}

Dear [Recipient's Name],

I hope you're doing well. I am writing to discuss the following matter: {clean_context}. Please let me know if you need any additional information.

Thank you for your time and consideration.

Best regards,
{sender_name if sender_name else "[Your Name]"}
"""
                st.subheader("Fallback Email Draft:")
                st.markdown(f"<div style='background-color:#f9f9f9;padding:15px;border-radius:5px;white-space:pre-wrap'>{fallback}</div>", unsafe_allow_html=True)
                st.code(fallback, language="text")
                st.download_button("Download Fallback Email", fallback, file_name="email_draft.txt")
    else:
        st.warning("Please provide both subject and details.")
