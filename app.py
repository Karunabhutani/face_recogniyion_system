import streamlit as st
import pandas as pd
import time 
from datetime import datetime

# Load background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('k.png') center;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add title or header
st.title('Attendance Dashboard')

# Read attendance data
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
df = pd.read_csv("Attendance/Attendance_" + date + ".csv")

# Display data
st.dataframe(df.style.highlight_max(axis=0))

# Add additional components or styling as needed
