import streamlit as st
import pandas as pd
from st_files_connection import FilesConnection

def run():
    print('Testing....')
    

    # Create connection object and retrieve file contents.
    # Specify input format is a csv and to cache the result for 600 seconds.
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read("annotbucket/EntitiesLegend.csv", input_format="csv", ttl=600)
    print(df.head(10))
    # Print results.
    #for row in df.itertuples():
    #    st.write(row.EntityName)