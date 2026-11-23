import pandas as pd
import numpy as np
import pickle

import streamlit as st
from datetime import datetime


def response_rate_question(se):
    """
    [nonresponse rate]
    response_rate_question - for a particular number of missing values
        chosen by the user, this function gives the number with at least
        that many missing values for each question. When 'OK' button is
        clicked, it gives the questions and number missing.
    """
    def display(tot):
        st.write(non_response[non_response['null']>=tot])
    df = se.df.copy()
    df=df.replace('nan',np.nan,regex=True)
    non_response = (df.isnull()).sum(axis=0)
    non_response.name = "null"
    non_response = pd.DataFrame(non_response)
    non_response["%"]=non_response['null']/len(df)
    tot = st.number_input(
                    label="Number of missing values or more to display.",
                    value=10,
                    key = "Question")
    disp=st.button('OK',key="ButtonQuestion")
    st.write(len(non_response[non_response['null']>=tot]),'out of',
                len(df.columns), 'missing at least',str(tot),'.')
    if disp:
        display(tot)
        disp = not(disp)

def response_rate_individual(se):
    """
    [nonresponse rate]
    response_rate_individual - for a particular number of missing values
        chosen by the user, this function gives the number with at least
        that many missing values for each individual. When 'OK' button is
        clicked, it gives the index and number of missed questions.
    """
    def display(tot):
        st.write(non_response[non_response['null']>=tot])
    df = se.df.copy()
    df = df.replace('nan',np.nan,regex=True)
    non_response = (df.isnull()).sum(axis=1)
    non_response.name = "null"
    non_response = pd.DataFrame(non_response)
    non_response["%"]=non_response['null']/len(df)
    tot = st.number_input(
                    label="Number of missing values or more to display.",
                    value=10,
                    key="Individual")
    disp=st.button('OK',key="ButtonIndividual")
    st.write(len(non_response[non_response['null']>=tot]),'out of',
                len(df), 'missing at least',str(tot),'.')
    if disp:
        display(tot)
        disp = not(disp)


def display_data(se,grp,top):
    data = pd.DataFrame(se.cluster_stats[grp]).T
    data['abs'] = np.abs(data['diff'])
    
    data = data.sort_values('abs',ascending=False)
    data = data.iloc[:top,:]
    if 'response' in data.columns:
        data = data[['diff','response']]
        data.columns = ['SD from Mean','response']
        return data.style.background_gradient(subset=['SD from Mean'],
                                              cmap='coolwarm',vmin=-1,vmax=1)
    else:
        cols = ['diff','mean','overall_mean']
        return data[cols].style.background_gradient(subset=['diff'],
                                              cmap='coolwarm',vmin=-1,vmax=1)


def col_assign_values(se):
        """
        col_assign_values - allows the user to assign values to all of the
            response_sets, response_map, and dict values using a drop down
            menu to select the responses and slider bars to adjust the 
            response values. It also allows the user to change the variable
            type.
        """
        if len(se.response_sets)==0:
            return "No need to map any responses"
        def assign_values(response,values):
            """
            assign_values - assigns the values (from the slider bars)  
                            to to the response map and then further assigns 
                            the dict for each column/question the response
                            map which was changed.
            Input: response - the responses for the columns in order with the
                              values for each of those responses.
                   values - the value for each response in the same order as 
                            the response.
                   map_ - is a global variable which is equal to the 
            """
            if q_type == "ordinal":
                for i,key in enumerate(map_.keys()):
                    se.response_map[response][key]=values[i]
                for col in se.response_sets[response]:
                    se.dict[col] = se.response_map[response].copy()
                    se.type[col] = 'ordinal'
            if q_type == "categorical":
                for i,key in enumerate(map_.keys()):
                    se.response_map[response][key]=values[i]
                for col in se.response_sets[response]:
                    se.dict[col] = se.response_map[response].copy()
                    se.type[col] = 'categorical'
            if q_type == "numerical":
                
                for col in se.response_sets[response]:
                    try:
                        se.newdf[col]=se.df[col].fillna(se.df[col].mean())
                        se.type[col]='numerical'
                    except:
                        st.write("Variable cannot be set to numerical")              
        def type_change():
            """
            works with the radio buttons to interact with the user.
            """
            new_type = st.session_state.radio_rsl
            se.variable_type = new_type
        def response_change():
            """
            works with radio buttons to interact with the user.
            """
            response = st.session_state.dd_rsl
            col = se.response_sets[response][0]
            se.current_response = response
            se.variable_type = se.type[col]
        
        ## rsl is the entire list of possible unique responses in the data
        rsl = list(se.response_sets.keys())
        st.selectbox(label="Select a response to map.",
                                         options=rsl,
                                         index=0,
                                         key = "dd_rsl",
                                         on_change=response_change)
        ## response_selected is tied to the select box using the callback
        response_selected = se.current_response
        N = len(response_selected)
        assigned_values = list(range(N))
        map_ = se.response_map[response_selected]
        map_ = se.order_dict(map_)
        ## all_columns are all of the columns with the select box response
        all_columns = se.response_sets[response_selected]
        st.write(str(all_columns))
        
        ## test to see if the response is in the response_map
        ## if it is, return the current assigned values for the responses
        if response_selected in (se.response_map.keys()):
            ## if currently mapped, assign the current values
            if len(map_.keys())>0:
                keys = list(map_.keys())
                assigned_values = [map_[keys[i]] for i in range(len(map_))]
            ## if not currently mapped, return the 1 to N numbers
            else:
                map_={response_selected[i]:assigned_values[i] 
                      for i in range(len(response_selected))}
        ## if not in the response map, let the user know by writing to the screen
        else:
            st.write(se.response_sets[response_selected])
            
        ## idx is the index of the current displayed variable type
        ## if it is changed vs if the dropdown changes
        response_change()
        idx = ["ordinal","categorical","numeric"].index(se.variable_type)
        st.radio(label="Variable for responses to these question.",
                          options=["ordinal","categorical","numeric"],
                          on_change = type_change,
                          key='radio_rsl',
                          index = idx)
        q_type = se.variable_type
        new_values=[]
        ## if the question type is ord or cat, then give slider bars for the 
        ## user to change if needed.
        if q_type == 'ordinal' or 'categorical':
            for i, key in enumerate(map_):
                new_values.append(st.slider(label=key,min_value=-1,max_value=N,
                          value=assigned_values[i],key=key))
        ## when the record button is clicked, then record the new assigned values
        ## and variable type.
        record_type = st.button("Record")
        if record_type:
            q_type = st.session_state.radio_rsl
            assign_values(response_selected,new_values)
            response_change()
            type_change()
            record_type = not(record_type)
            return se
        
def change_mapping(se,col,values):
    curr_map = se.order_dict(se.dict[col])
    for i,key in enumerate(curr_map.keys()):
        curr_map[key] = values[i]
        se.dict[col] = curr_map.copy()
        

def pivot_analysis(se):
    def row_change():
        pass
    def col_change():
        pass
        
        
    df = se.newdf.copy()
    df['cluster'] = se.cluster_assignments.values
    rows = list(se.columns)
    cols = se.cluster_assignments.unique()
    val = list(se.columns)
    col1,col2,col3 = st.columns(3)
    
    with col1:
        st.selectbox("Select questions to aggregate",
                        options=val,
                        index=len(val)-1,
                        key = "pvt_val")
        value = st.session_state.pvt_val
    
    with col2:
        st.multiselect("Select row questions",
                        options=rows,
                        default = rows[1],
                        on_change = row_change,
                        key = "pvt_row")
        selected_rows = st.session_state.pvt_row
    with col3:
        cols = sorted(cols)
        st.multiselect("Select cluster for Pivot Table",
                        options = sorted(cols),
                        default = cols,
                        on_change = col_change,
                        key="pvt_col")
        selected_cols = st.session_state.pvt_col
    st.write(f"Pivot Table - Inside: {st.session_state.pvt_val}")
    st.write(f"Columns: Clusters {sorted(selected_cols)}")
    st.write(f"Rows: Question(s): {sorted(selected_rows)}")
    if len(st.session_state.pvt_row)>0 and len(st.session_state.pvt_col)>0:
        df=df[df['cluster'].isin(selected_cols)]
        tbl = df.pivot_table(values=value,
                                index=selected_rows,
                                columns = 'cluster',
                                aggfunc=np.mean)
        st.write(tbl)

def save_pickle(se, filename='survey_mapping.pkl'):
    """
    save_mapping - saves a copy of the response_map for use in 
                    future survey questions.
    get all of the unique responses from response_sets and
    assign response_map to the first val - assumes they all
    have the same response values
    """
    save_map = se.save_mapping()
    st.download_button("Download Mapping",
                        data = pickle.dumps(save_map),
                        file_name = filename)