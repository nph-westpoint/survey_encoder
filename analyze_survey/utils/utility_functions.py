import streamlit as st



def session():
    for key in st.session_state.keys():
        st.write(key, st.session_state[key])
    return None

def change_attribute(key1,key2):
    """
    changes a session_state attribute
    key1 - session_state attribute
    key2 - key for the widget
    """
    val = st.session_state[key2]
    st.session_state[key1]=val
    return None

def display_page_links(pages):
    for key in pages.keys():
        st.sidebar.page_link(pages[key],label=key)
    return None


        








