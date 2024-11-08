import streamlit as st

local_gpt = st.Page("01_local_GPT_agent.py", title="Local GPT", icon="💬")
document_ai = st.Page("02_Document_AI.py", title="Document AI", icon="📄")
# esrs_ai = st.Page("03_ESRS_AI.py", title="ESRS AI", icon="🔍")
# rba_ai = st.Page("04_RBA_AI.py", title="RBA AI", icon="🔍")

pg = st.navigation([local_gpt, document_ai])

pg.run()
