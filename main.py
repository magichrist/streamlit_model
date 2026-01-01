from dotenv import load_dotenv
import streamlit_authenticator as stauth
import streamlit as st
import os
from streamlit_extras.let_it_rain import rain
from streamlit_extras.switch_page_button import switch_page

keys = st.session_state.keys()
if all(i not in keys for i in ["n_samples", "data_preprocessing", "n_features", "centers_std", "seed", "lock_seed"]):
    st.session_state.n_samples = 200
    st.session_state.n_features = 5
    st.session_state.centers = 2
    st.session_state.centers_std = 1
    st.session_state.seed = 42
    st.session_state.lock_seed = True
    st.session_state.data_preprocessing = None
col1, col2 = st.columns(2)
col2.image("./krist.jpeg", width=200)
if col2.button("Behave",
               on_click=lambda: rain(emoji="🥷🏻☦️🪄", font_size=80, animation_length=1, falling_speed=6), type="primary"):
    col2.text("How dare you godless infidel!")

st.sidebar.button("clear cache", on_click=lambda: st.cache_data.clear())
