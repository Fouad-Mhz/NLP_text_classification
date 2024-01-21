import streamlit as st
from inference_files.tag_suggestion import *

def show():
    st.title("Inference Page")

    # User input
    user_input = st.text_area("Enter a sentence for tag suggestion:", "")

    multi = []
    meta = []

    # Make inference when the user clicks the button
    if st.button("Get Tag Suggestions"):
        st.subheader("Tag Suggestions:")

        # Make the inference
        multi, meta = inference(user_input, embed=embed, mlb=mlb, base_models=loaded_models, meta_model=meta_model)

        # Display Multi Model results
        st.subheader("Multi Model:")
        multi_line = " | ".join(f" {tag.upper()}" for tag in multi)
        st.markdown(multi_line)

        # Display Meta Model results
        st.subheader("Meta Model:")
        meta_line = " | ".join(f" {tag.upper()}" for tag in meta)
        st.markdown(meta_line)

if __name__ == "__main__":
    show()
