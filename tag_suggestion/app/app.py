import streamlit as st
from pages import monitor_page, meta_page, inference_page

def main():
    st.sidebar.title("Navigation")

    # Create a sidebar with page options
    page_options = ["Monitor Page", "Meta Page", "Inference Page"]
    selected_page = st.sidebar.radio("Select Page", page_options)

    # Render the selected page
    if selected_page == "Monitor Page":
        monitor_page.show()
    elif selected_page == "Meta Page":
        meta_page.show()
    elif selected_page == "Inference Page":
        inference_page.show()

if __name__ == "__main__":
    main()