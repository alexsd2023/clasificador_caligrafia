import streamlit as st
import spacy
from spacy.lang.es.examples import sentences 
from spacy.lang.es.stop_words import STOP_WORDS
from spacy import displacy
from spacy.tokens import Span
import streamlit.components.v1 as components
import tempfile
from pathlib import Path

def run():
        
        tab1, tab2= st.tabs(['Entities', 'Parts of speech'])
        nlp = spacy.load('es_core_news_sm')
        fp= None
        flag= False
        texto= ""

        if 'annot_file' in st.session_state:
                uploaded_file= st.session_state['annot_file']
                texto= ""
                if uploaded_file is not None:
                    print(uploaded_file) 
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        #st.markdown("# Original text file")
                        fp = Path(tmp_file.name)
                        fp.write_bytes(uploaded_file.getvalue())
                        #print(fp)
                    flag= True  
                    with open(fp,'r') as file:
                        texto = " ".join(line for line in file)

        with tab1:
            if flag:        
                tab1.text_area("Annotated text", value= texto,  key= "text", height=520)              
                doc= nlp(texto)
                ent_html= displacy.render(doc, style="ent", jupyter= False)
                tab1.markdown(ent_html, unsafe_allow_html= True)
                #displacy.serve(doc, style="dep")
        
        with tab2:
            if flag:
                ent_html= displacy.render(doc, style="dep", jupyter= True)
                tab2.markdown(ent_html, unsafe_allow_html= True)
                 
                