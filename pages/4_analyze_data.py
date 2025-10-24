import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import analyze_survey as ase

se = st.session_state['survey_encoder']
pages_pict = st.session_state['pages_dict']
ase.display_page_links(pages_pict)

if se is not None:
    st.header('Analyze Clusters')
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Cluster Differences",
                        "One to One Analysis",
                        "Bar Chart",
                        "KDE Chart",
                        "Pivot Analysis"])
    opts = list(se.columns)
    idx = opts.index(se.compare_col)
    col_ = st.sidebar.selectbox(label="Select a comparison column.",options = se.columns,index=idx)
    with tab1:
        st.subheader("Cluster Differences")
        label = "The columns listed separate the cluster from the rest "
        label += "of the data."
        st.write(label)
        
        if len(se.kdata)> 0:
            cols = st.columns(3)
            with cols[0]:
                lbl = "Select a cluster->0"+"-"+str(se.clusters-1)+"."
                grp = st.selectbox(lbl,
                                index = 0,
                                options = np.arange(se.clusters),
                                )
            with cols[1]:
                tot = st.slider("Number of differences to display.",
                                value = 5,
                                min_value = 5,
                                max_value = len(se.df.columns),
                                step = 5)
            with cols[2]:
                include_negative = st.checkbox("Include missing values.",
                                               value = False,
                                               )
            
            st.write(se.differentiate_cluster(grp,tot,include_negative))

    with tab2:
        st.subheader("One to One Analysis")
        if (hasattr(se,'cluster_assignments')):
            clusters = np.arange(se.clusters)
            grps= st.multiselect("Select two clusters to compare.",
                           options = clusters,
                           max_selections = 2)
            if len(grps)==2:
                col1,col2 = st.columns([1,3])
                with col1:
                    tot = st.slider("Number of differences to display.",
                                    value = 5,
                                    min_value=5,
                                    max_value = len(se.k_columns),
                                    key = 'one_one_tot',
                                    step=5)
                with col2:
                    st.write(se.cluster_summary(se.compare_col,grps))
                include_missing = st.checkbox("Include missing data",value = False)
                st.write(se.one_to_one_analysis(grps[0],grps[1],top=tot,include_missing=include_missing))
                
                
                
    with tab3:
        st.subheader("Bar Chart Comparison")
        fig,ax = plt.subplots(figsize=(10,6))
        def calc_again():
            st.session_state.calc=True
        st.button("Show bar chart",key = "show_bar")
        if st.session_state.show_bar:
            calc_again()
        if len(se.kdata)>0 and (hasattr(se,'cluster_assignments')):
            col1,col2 = st.columns(2)
            all_clusters = sorted(list(se.cluster_assignments.unique()))
            all_questions = list(se.columns)
            with col1:
                clusters = st.multiselect("Select the clusters to compare.",
                                          options = all_clusters,
                                          key = 'selected_clusters')
            with col2:
                question = st.selectbox("Select the question to compare.",
                                        options = all_questions,
                                        key = 'selected_question')
            if (len(clusters)>0) and (question is not None):
                distn = se.col_distribution_by_cluster(col=question,grps=clusters)
                cols = list(distn.columns)
                cols[-1] = 'distn: '+str([int(c) for c in clusters])
                distn.columns = cols
                distn = distn.T
                if se.type[question] == 'numerical':
                    distn.columns = distn.columns.map(lambda x:f"{x*100:.2f}%")
                    cols = distn.columns
                    distn = distn.style.format({col: '{:.2f}' for col in cols})
                    distn.set_properties(**{'width':'50px'})
                else:
                    cols = distn.columns
                    distn = distn*100
                    distn = distn.style.format({col: '{:.2f}%' for col in cols})
                    distn.set_properties(**{'width':'100px'})
                st.pyplot(se.display_col_distn_by_cluster(col=question,grps=clusters,alpha=0.4))
                st.write(distn)

                
                
    with tab4:
        st.subheader("Density Chart Comparison")
        def calc_again():
            st.session_state.calc=True
        st.button("Show density chart",key = "show_bar_density")
        if st.session_state.show_bar:
            calc_again()
        if len(se.kdata)>0 and (hasattr(se,'cluster_assignments')):
            col1,col2 = st.columns(2)
            all_clusters = sorted(list(se.cluster_assignments.unique()))
            all_questions = list(se.columns)
            with col1:
                clusters = st.multiselect("Select the clusters to compare.",
                                          options = all_clusters,
                                          key = 'density_clusters',
                                          default = all_clusters)
            with col2:
                question = st.selectbox("Select the question to compare.",
                                        options = all_questions,
                                        key = 'density_question')
            if len(clusters)>0 and (question is not None):
                st.pyplot(se.density_chart(question,clusters))
                st.write("Overall Distribution and Mapping for : "+str(question))
                distn = se.col_distribution_by_cluster(col=question,grps=clusters)
                cols = list(distn.columns)
                cols[-1] = 'distn: '+str([int(c) for c in clusters])
                distn.columns = cols
                distn = distn.T
                if se.type[question] == 'numerical':
                    distn.columns = distn.columns.map(lambda x:f"{x*100:.2f}%")
                    cols = distn.columns
                    distn = distn.style.format({col: '{:.2f}' for col in cols})
                    distn.set_properties(**{'width':'50px'})
                else:
                    cols = distn.columns
                    distn = distn*100
                    distn = distn.style.format({col: '{:.2f}%' for col in cols})
                    distn.set_properties(**{'width':'100px'})


                st.write(distn)

