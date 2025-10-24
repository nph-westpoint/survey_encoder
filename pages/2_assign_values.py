import streamlit as st
import analyze_survey as ase

se = st.session_state['survey_encoder']
pages_pict = st.session_state['pages_dict']
ase.display_page_links(pages_pict)

if se is not None:
    st.header('Assign Values')

    tab1, tab2, tab3, tab4 = st.tabs(["Counts","Current Mapping",
                                "Question Non-Response Rate",
                                "Individual Non-Response Rate"])
    with tab1:
        st.subheader('Counts')
        if se is not None:
            cols = se.non_numeric_cols()
            if len(cols)>0:
                col = st.selectbox(label="Select Question to Count.",
                                    options = cols, index=0)
                st.write(se.counts(col))
    with tab2:
        st.subheader('Current Mapping')
        if se is not None:
            cols = se.non_numeric_cols()
            if len(cols)>0:
                col = st.selectbox(label = "Select a Question to Map.",
                        options = cols, index=0)
                curr_map = se.order_dict(se.dict[col])
                N = len(curr_map)
                st.write(se.current_mapping(col))
                new_values=[]
                for key,val in curr_map.items():
                    new_values.append(st.slider(label=key,min_value=-1,max_value=N,
                                value=val,key=key))
                ## Change button
                change_type = st.button("Change")
                if change_type:
                    ase.change_mapping(se,col,new_values)
                    change_type = not(change_type)

    with tab3:
        st.subheader("Question Non-Response Rate")
        st.write("Below are the number of individuals who did not answer the displayed question.")
        if se is not None:
            tot = st.number_input(
                    label="Number of missing values or more to display.",
                    value=10,
                    key = "Question")
            disp=st.button('Display Questions',key="ButtonQuestion")
            non_response = se.response_rate_question(tot)
            res = f"{len(non_response)} questions out of {len(se.columns)} were missing \
                            at least {tot} answers."
            st.markdown(res)
            if disp:
                st.write(non_response)
                disp = not(disp)
    with tab4:
        st.subheader("Individual Non-Response Rate")
        st.write("Below are the number of questions that an individual did not attempt.")
        if se is not None:
            tot = st.number_input(
                    label="Number of missing values or more to display.",
                    value=10,
                    key = "individual")
            disp=st.button('Display Individuals',key="ButtonIndividual")
            non_response = se.response_rate_individual(tot)
            res = f"{len(non_response)} out of {len(se)} total individuals failed \
                    to answer at least {tot} questions."
            st.markdown(res)
        if disp:
                st.write(non_response)
                disp = not(disp)