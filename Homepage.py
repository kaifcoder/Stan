# main.py
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map='auto', torch_dtype=torch.float32)

# create text2text pipeline


def llm_pipeline(text):
    pipe_sum = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=400,
        min_length=50)

    result = pipe_sum(text)
    result = result[0]['generated_text']
    return result


# Sidebar navigation
st.set_page_config(
    page_title="Stan: Summarization and Text Analysis", page_icon="✒️")
st.title("Stan: Summarization and Text Analysis")
st.write("Welcome to Stan, your tool for summarizing and analyzing text.")

# User Input Section
st.header("Input Text")
user_input = st.text_area("Enter the text you want to analyze:", height=200)

# Summarization and Analysis Options
st.header("Summarization and Text Analysis Options")

# Checkbox for summarization
should_summarize = st.checkbox("Summarize Text")

# Checkbox for question answering
fix_grammer = st.checkbox(
    "Fix Grammatical Errors ")

# Button to initiate analysis
if st.button("Analyze"):
    if user_input:
        # Perform text analysis based on user's selection
        if should_summarize:
            # summary = stan.summarize(user_input)  # Replace with your summarization function
            summary = llm_pipeline(
                f"Summarize : {user_input} and the summary is :")
            st.subheader("Summary:")
            st.write(summary)

        if fix_grammer:
            # answers = stan.answer_questions(user_input, questions)  # Replace with your QA function
            answers = llm_pipeline(
                f"Fix the grammatical mistake in the given text : {user_input} and the correct sentence is :")
            st.subheader("Answers:")
            st.write(answers)
    else:
        st.warning("Please enter some text for analysis.")

# About Section
st.header("About Stan")
st.write("Stan is a tool that utilizes large language models for text summarization and answering questions.")
st.write("It works offline, making it a useful resource for text analysis and summarization tasks.")

# Contact Information
st.header("Contact Information")
st.write("For questions and support, contact us at contact@stan-tool.com")

# Footer
st.markdown("© 2023 Stan Tool. All rights reserved.")
