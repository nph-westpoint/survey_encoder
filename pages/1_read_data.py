import streamlit as st
import analyze_survey as ase
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'..')
from analyze_survey import SurveyEncoder

def text_input_callback():
    missing_values = st.session_state['missing_text']
    se = st.session_state['survey_encoder']
    if se is not None:
        df = se.df.replace(missing_values,np.nan).copy()
        st.session_state['survey_encoder'] = SurveyEncoder(df,mapping=True)
    

se = st.session_state['survey_encoder']
pages_pict = st.session_state['pages_dict']
selected_response = st.session_state['selected_response']
ase.display_page_links(pages_pict)
missing_values = st.sidebar.text_input(label = "Text used for missing values:",
                                       key = 'missing_text',
                                       on_change = text_input_callback,
                                       )
tab1, tab2,tab3,tab4 = st.tabs(["Import Survey",
                            "Import Mapping",
                            "Map Responses",
                            "Save Mapping"])

with tab1:
    st.subheader("Import csv File")
    if st.session_state['survey_encoder'] is None:
        uploaded_file = st.file_uploader("Select .csv survey file.",type='csv')
        if uploaded_file is not None:
            
            df = pd.read_csv(uploaded_file,index_col=0,keep_default_na=False)
            df = df.replace(missing_values,np.nan)
            se = SurveyEncoder(df,mapping=True)
            st.session_state['survey_encoder'] = se
            st.write(se.df)
    else:
        st.write("Survey has been imported.")
        st.write(se.df)
with tab2:
    st.subheader("Import previously saved mapping.")
    se = st.session_state['survey_encoder']
    if se is not None:
        if not se.mapped_encoder:
            mapping = st.file_uploader("Select a mapping.",type="pkl")
            if mapping is not None:
                se.load_mapping(mapping)
        if se.mapped_encoder:
            st.write("Survey Encoder Mapping has been imported.")
        st.write(se.type)
            
with tab3:
    st.subheader("Map Survey Responses")
    se = st.session_state['survey_encoder']
    if se is not None:
        def radio_callback():
            se.reassign_type(selected_response,st.session_state['radio_rsl'])
        rsl = list(se.response_sets.keys())
        if selected_response is not None:
            idx = rsl.index(selected_response)
        else:
            idx = 0
        selected_response = st.selectbox(label="Select a response to map.",
                                         options=rsl,
                                         index = idx,
                                         key = "dd_rsl",
                                         on_change=ase.change_attribute,
                                         args = ('selected_response','dd_rsl'))
        
        if selected_response is not None:
            N = len(selected_response)
            assigned_values = list(range(N))
            current_map = se.response_map[selected_response]
            current_map = se.order_dict(current_map)
            all_columns = se.response_sets[selected_response]
            var_type = se.type[all_columns[0]]
            st.write("Questions that have this response:")
            st.write(str(all_columns))
            if selected_response in se.response_map.keys():
                keys = list(current_map.keys())
                assigned_values = [current_map[keys[i]] for i in range(len(current_map))]
            else:
                current_map={selected_response[i]:assigned_values[i] for i in range(len(selected_response))}
            idx = ["ordinal","categorical"].index(var_type)
            st.radio(label = "Type of variable that have these responses.",
                    options = ["ordinal","categorical"],
                    key = 'radio_rsl',
                    index = idx,
                    on_change = radio_callback,
                    )
            st.write(var_type)
            new_values = []
            for i,key in enumerate(current_map):
                new_values.append(st.slider(label=key,min_value=-1,max_value=N,
                                            value =assigned_values[i],key=key))
            record_change = st.button("Record")
            if record_change:
                se.reassign_values(selected_response,new_values)
                st.rerun()
        
with tab4:
    st.subheader("Save Mapping")
    se = st.session_state['survey_encoder']
    if se is not None:
        filename = st.text_input("Filename:",
                        placeholder='filename.pkl')
        save_mapping_btn = st.button('Save', key = 'save_mapping_key')
        if save_mapping_btn:
            ase.save_mapping(se,filename)
        
        