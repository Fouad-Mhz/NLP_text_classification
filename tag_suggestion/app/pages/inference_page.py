import streamlit as st
from inference_files.tag_suggestion import *

def show():
    st.title("Inference Page")

    # Load resources
    embed, mlb, loaded_models, meta_model = import_resources()

    # User input
    user_input = st.text_area("Enter a sentence for tag suggestion:", "")

    # Make inference when the user clicks the button
    if st.button("Get Tag Suggestions"):
        st.subheader("Tag Suggestions:")
        inference(user_input, embed=embed, mlb=mlb, base_models=loaded_models, meta_model=meta_model)

if __name__ == "__main__":
    show()