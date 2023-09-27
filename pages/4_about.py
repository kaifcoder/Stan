# pages/about.py
import streamlit as st


st.title("About Stan")

st.write("Stan is a powerful text analysis tool designed to summarize content and answer questions using large language models.")

st.header("Key Features")
st.markdown("- Summarize long texts into concise paragraphs.")
st.markdown(
    "- Answer questions about your text, such as 'What,' 'When,' and 'Who' questions.")
st.markdown("- Works offline, ensuring data privacy and security.")
st.markdown("- User-friendly interface for easy text analysis.")

st.header("How Stan Works")
st.write("Stan is built upon the latest advancements in natural language processing. It utilizes a locally deployed language model to perform tasks such as summarization and question answering.")

st.header("Contact Information")
st.write("For questions and support, please contact us at contact@stan-tool.com.")

st.header("Team")
st.write("Stan is developed and maintained by a team of dedicated software engineers and AI enthusiasts.")

st.header("Privacy")
st.write("We value your privacy. Stan does not require an internet connection and does not store your data.")

st.header("Terms of Use")
st.write("By using Stan, you agree to our terms of use. Please review them on our website.")

st.header("Acknowledgments")
st.write("We would like to acknowledge the open-source community and the developers of the underlying libraries and tools that made Stan possible.")
