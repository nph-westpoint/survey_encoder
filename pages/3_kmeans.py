import streamlit as st


import analyze_survey as ase

se = st.session_state['survey_encoder']
pages_pict = st.session_state['pages_dict']
k = st.session_state['k']
n_dim = st.session_state['n_dim']
ase.display_page_links(pages_pict)

if se is not None:
    st.header('Kmeans')

    tab1,tab2,tab3,tab4 = st.tabs(["Set Seed",
                                                "Scree Plot",
                                                "K-means",
                                                "Summary"])
    with tab1:
        st.subheader("Set Seed")
        seed_value = st.number_input("Set seed value.",
                                          value=se.seed_value)
        left,right = st.columns(2)
        set_seed = left.button("True",key = "Seed_True")
        unset_seed = right.button("False",key = "Seed_False")
        if set_seed:
            se.set_seed(seed_value,True)
        if unset_seed:
            se.set_seed(seed_value,False)
        st.write("Currently the seed is set:", se.seed_set)
        if se.seed_set:
            st.write("The seed value is:", se.seed_value)
    with tab2:
        st.subheader("Scree Plot")
        cols = st.columns(2)
        min_,max_ = se.min_max
        with cols[0]:
            min_ = st.number_input("Lower Bound on Scree Plot.",value=2)
        with cols[1]:
            max_ = st.number_input("Upper Bound on Scree Plot.",value=10)
        show_scree = st.button(label="Show Scree")
        if show_scree:
            fig = se.scree_plot(min_,max_)
            st.pyplot(fig)
    with tab3:
        cols = st.columns(2)
        with cols[0]:
            st.subheader("PCA plot of K-means clusters")
        with cols[1]:
            pca_plot = st.button("See Plot",key = 'see_pca_plot')
        cols = st.columns(3)
        with cols[0]:
            k = st.number_input("Number of clusters to group data.",
                                min_value = 2,
                                max_value = 10,
                                value=k,
                                on_change = ase.change_attribute,
                                key='k_input',
                                args = ('k','k_input'))
        with cols[1]:
            n_dim = st.radio('Number of dimensions to plot.',
                            options = [2,3],
                            index = 0,
                            key = 'n_dim_input',
                            on_change = ase.change_attribute,
                            args = ('n_dim','n_dim_input'))
            
        with cols[2]:
            alpha = st.slider(r"Select $\alpha$ for the scatter plot.",
                              min_value = 10,
                              max_value = 100,
                              step = 10,
                              value = 40
                              )

        if (se.pca_plot_configs!=(k,n_dim,alpha/100)) or (pca_plot):
            fig = se.PCA_plot(k=k,n_dim=n_dim,alpha=alpha/100)
            st.plotly_chart(fig, theme = None, use_container_width=True)
            # if n_dim == 2:
            #     st.pyplot(se.PCA_plot(k=k,n_dim=n_dim,alpha=alpha/100))
            # else:
            #     fig = se.PCA_plot(k=k,n_dim=n_dim,alpha=alpha/100)
            #     st.plotly_chart(fig, theme = None, use_container_width=True)

        
    with tab4:
        st.subheader("Summary of Clusters")
        if hasattr(se, 'diff'):
            idx = list(se.k_columns).index(se.compare_col)
            col = st.session_state['selected_col']
            if col is not None:
                idx = list(se.k_columns).index(col)
            else:
                idx = None
            st.selectbox("Choose a column to compare.",
                               index=idx,
                               options=se.k_columns,
                               key = 'col_select_box',
                               on_change=ase.change_attribute,
                               args=('selected_col','col_select_box')
                               )
            if col is not None:
                st.write(se.cluster_summary(col))
                st.write("Mapped data used for the K-means calculation. Download using the attached button.")
                st.write(se.kdata)
                se.compare_col = col
            