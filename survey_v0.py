# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 07:49:18 2024

@author: grover.laporte
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import tempfile
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class SurveyEncoder(object):
    def __init__(self,df,mapping=False):
        df = df.replace('','nan',regex=True)
        self.df = df.copy()
        self.columns = df.columns
        self.dict = {col:dict() for col in df.columns}
        self.response_sets = {}
        self.response_map = {}
        self.type = {col:"categorical" for col in df.columns}
        self.newdf = pd.DataFrame([],columns = df.columns)
        self.clusters = 4
        self.seed_value = 42
        self.seed_set = False
        self.kdata = []
        ### Run at the start for all objects ####
        self.col_assign_types()
        self.definitions = {}
        self.mapped_encoder = False
        if mapping:
            #set to False if you are going to bring in your own mapping
            self.col_assign_values()
    
    ###########################################################################
    ########### Data Munging ##################################################
    ###########################################################################
    def map_values(self,col):
        """
        map_values - given a particular column / question, maps the values
         in self.dict to the values in the newdf dataframe.

        Parameters
        ----------
        col : string
            the column of the question that needs to be mapped.

        Returns
        -------
        self.newdf that has the column mapped to the values in self.dict.

        """
        dict1 = self.dict[col]
        self.newdf[col]=self.df[col].map(dict1)
    

        
    def auto_assign_response(self,response):
        """
        auto_assign_response - automatically assigns nominal numeric values to all
            columns in a response set and creates a dict of col values that can be updated later.
        """
        self.response_map[response]={}
        #st.write(self.response_map[response].keys())
        dict1 = {v:k for k,v in enumerate(response)}
        if 'nan' in dict1.keys():
            dict1['nan']=-1
        self.response_map[response]=dict1
        cols = self.response_sets[response]
        for col in cols:
            self.dict[col]=dict1
        
        
    def auto_assign_values(self,col):
        """
        auto_assign_values - automatically assigns nominal numeric values to columns
                            and creates a dict of col values that can be updated later.
        """

        dict1 = {v:k for k,v in enumerate(self.dict[col].keys())}
        if 'nan' in dict1.keys():
            dict1['nan']=-1
        self.dict[col]=dict1
        self.map_values(col)
        

        
    
    def col_assign_types(self):
        """
        For each column, do three things: 
            1) determine if it is numerical or 
                possibly categorical or ordinal. The constructor method assumes 
                all columns are categorical. This will be changed to numerical 
                here or to ordinal later if the user changes the mapping.
            2) creates the response_sets dictionary that will be used later to 
                map the responses to numerical values.
            3) creates a response_map dictionary that will also be used later.
        """
        df = self.df.copy()
        for col in self.columns:
            try:
                ## if the column are numbers except the 'nan' values, then 
                ## give that column an np.float type which will eventually
                ## be translated below as a numerical column
                df[col]=df[col].replace('nan',0).astype(np.float64)
            except:
                pass
            if df[col].dtype == 'object':
                ## if the column is anything but numerical, then the user needs
                ## to classify it as categorical or ordinal based on the response
                ## so this code sets up all of the possible responses and 
                ## finds the columns that have exactly those responses.
                curr_set = set(self.df[col].astype("str").fillna("nan").unique())
                ## we add nan to every response possibility so that it one
                ## column missing a value with the same responses does not 
                ## have a different response set value than one not missing one
                curr_set.add("nan")
                ## sort and make a tuple so that we can make these keys to a 
                ## dictionary - having them as keys lets you check to see if 
                ## that resonse has already been seen in the columns.
                curr_set= tuple(sorted(curr_set))
                ## if the current set of responses is not in the response set
                ## keys, then add it and a standard response map of 1 to N
                if curr_set not in list(self.response_sets.keys()):
                    self.response_sets[curr_set]=[col]
                    self.response_map[curr_set] = {v:k for k,v in enumerate(curr_set)}
                    self.response_map[curr_set]['nan']=-1
                ## if it is in the response set already, append the column to
                ## the list of columns that have that set of responses.
                else:
                    self.response_sets[curr_set].append(col)
            ## if the column type is not an object, then assume it must be 
            ## numerical and assign the values in newdf to those values
            else:
                self.type[col]='numerical'
                ## this will keep 'nan' values in the numerical columns of 
                ## newdf which is problematic later.
                self.newdf[col]=self.df.loc[:,col]
                self.newdf[col]=self.df.loc[:,col].replace('nan',np.nan)
                #self.newdf[col]=self.newdf[col].fillna(self.newdf[col].mean())
        self.current_response = list(self.response_sets.keys())[0]
        self.variable_type = self.type[self.response_sets[self.current_response][0]]
                
                
    def order_dict(self,d):
        """
        order_dict - orders a dictionary by the item values instead of the
                    random order that a dictionary usually orders them.
        Input: d - dictionary with keys and items
        Output dictionary ordered by item order, either alphabetically or
                    numerically.
        """
        return {k:v for k,v in sorted(d.items(),key=lambda item: item[1])}
                
    def col_assign_values(self):
        """
        col_assign_values - allows the user to assign values to all of the
            response_sets, response_map, and dict values using a drop down
            menu to select the responses and slider bars to adjust the 
            response values. It also allows the user to change the variable
            type.
        """
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
                    self.response_map[response][key]=values[i]
                for col in self.response_sets[response]:
                    self.dict[col] = self.response_map[response].copy()
                    self.type[col] = 'ordinal'
            if q_type == "categorical":
                for i,key in enumerate(map_.keys()):
                    self.response_map[response][key]=values[i]
                for col in self.response_sets[response]:
                    self.dict[col] = self.response_map[response].copy()
                    self.type[col] = 'categorical'
            if q_type == "numerical":
                
                for col in self.response_sets[response]:
                    try:
                        self.newdf[col]=self.df[col].fillna(self.df[col].mean())
                        self.type[col]='numerical'
                    except:
                        st.write("Variable cannot be set to numerical")              
        def type_change():
            """
            works with the radio buttons to interact with the user.
            """
            new_type = st.session_state.radio_rsl
            self.variable_type = new_type
        def response_change():
            """
            works with radio buttons to interact with the user.
            """
            response = st.session_state.dd_rsl
            col = self.response_sets[response][0]
            self.current_response = response
            self.variable_type = self.type[col]
        
        ## rsl is the entire list of possible unique responses in the data
        rsl = list(self.response_sets.keys())
        st.selectbox(label="Select a response to map.",
                                         options=rsl,
                                         index=0,
                                         key = "dd_rsl",
                                         on_change=response_change)
        ## response_selected is tied to the select box using the callback
        response_selected = self.current_response
        N = len(response_selected)
        assigned_values = list(range(N))
        map_ = self.response_map[response_selected]
        map_ = self.order_dict(map_)
        ## all_columns are all of the columns with the select box response
        all_columns = self.response_sets[response_selected]
        st.write(str(all_columns))
        
        ## test to see if the response is in the response_map
        ## if it is, return the current assigned values for the responses
        if response_selected in (self.response_map.keys()):
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
            st.write(self.response_sets[response_selected])
            
        ## idx is the index of the current displayed variable type
        ## if it is changed vs if the dropdown changes
        response_change()
        idx = ["ordinal","categorical","numeric"].index(self.variable_type)
        st.radio(label="Variable for responses to these question.",
                          options=["ordinal","categorical","numeric"],
                          on_change = type_change,
                          key='radio_rsl',
                          index = idx)
        q_type = self.variable_type
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
            
    
    ##########################################################################        
    ##############  Radio button 2 ###########################################
    ##########################################################################        
    def counts(self):
        """
        counts - displays the count for each response of the selected question.
        """
        cols = self.columns
        col = st.selectbox(label="Select Question to Count.",
                           options = cols, index=0)
        st.write(self.df[col].value_counts())
        
        
    def current_mapping(self):
        """
        current_mapping - shows the current mapping from smallest to largest
                    value and gives the ability for the user to change the
                    mapping for each question/column.
        """
        def change_mapping(col,values):
            for i,key in enumerate(curr_map.keys()):
                curr_map[key] = values[i]
                self.dict[col] = curr_map.copy()
            
        
        cols = self.columns
        col = st.selectbox(label = "Select a Question to Map.",
                           options = cols, index=0)
        #order the response of the dictionary
        curr_map = self.order_dict(self.dict[col])
        N = len(curr_map)
        df = pd.DataFrame(curr_map.values(),index=curr_map.keys(),columns=['map'])
        counts = self.df[col].value_counts()
        df['count']=counts
        st.write(df)

        
        # Create the data structure and I/O for the new values to be stored.
        new_values=[]
        for key,val in curr_map.items():
            new_values.append(st.slider(label=key,min_value=-1,max_value=N,
                      value=val,key=key))
        ## Change button
        change_type = st.button("Change")
        if change_type:
            change_mapping(col,new_values)
            change_type = not(change_type)
                  
            
    def response_rate_question(self):
        """
        [nonresponse rate]
        response_rate_question - for a particular number of missing values
            chosen by the user, this function gives the number with at least
            that many missing values for each question. When 'OK' button is
            clicked, it gives the questions and number missing.
        """
        def display(tot):
            st.write(non_response[non_response['null']>=tot])
        df = self.df.copy()
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
            
    def response_rate_individual(self):
        """
        [nonresponse rate]
        response_rate_individual - for a particular number of missing values
            chosen by the user, this function gives the number with at least
            that many missing values for each individual. When 'OK' button is
            clicked, it gives the index and number of missed questions.
        """
        def display(tot):
            st.write(non_response[non_response['null']>=tot])
        df = self.df.copy()
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
    
    ##########################################################################
    #############  K-Means / GMM #############################################
    ##########################################################################
    
    def set_seed(self):
        self.seed_value = st.number_input("Set seed value.",
                                          value=self.seed_value)
        left,right = st.columns(2)
        set_seed = left.button("True",key = "Seed_True")
        unset_seed = right.button("False",key = "Seed_False")
        if set_seed:
            self.seed_set = True
        if unset_seed:
            self.seed_set = False
        st.write("Currently the seed is set:", self.seed_set)
        if self.seed_set:
            st.write("The seed value is:", self.seed_value)
            
    def set_kdata(self):
        cols = [col for col in self.columns 
                if self.type[col]=='ordinal' or 
                self.type[col]=='numerical']
        data = self.newdf.loc[:,cols].replace('nan',-1).copy()
        for col in data.columns:
            data[col] = pd.to_numeric(data[col],errors='coerce').fillna(-1,downcast='infer')
        self.kdata = data.copy()

        
    def scree_plot(self):
        self.set_kdata()
        # cols = [col for col in self.columns 
        #         if self.type[col]=='ordinal' or 
        #         self.type[col]=='numerical']
        # ## How to deal with missing data here...
        # self.kdata = self.newdf.loc[:,cols]
        X = self.kdata.copy()
        distortion = []
        min_ = st.number_input("Lower Bound on Scree Plot.",value=2)
        max_ = st.number_input("Upper Bound on Scree Plot.",value=10)
        max_+=1 #to get the next number for all of the max_ uses
        for i in range(min_,max_):
            km = KMeans(n_clusters = i, init = 'random',n_init=10,
                        max_iter = 300, tol=1e-4)
            km.fit(X)
            distortion.append(km.inertia_)
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        ax.plot(range(min_,max_),distortion,marker='o')
        ax.set_xlabel('Number of clusters (K)')
        ax.set_ylabel('Distortion')
        ax.set_title("Determine the Number of Clusters (K)")
        y_max = ax.get_ylim()[1]
        for i in range(min_,max_):
            ax.annotate(str(i),xy=(i,distortion[i-min_]+0.01*y_max))
        st.pyplot(fig)
        clusters = st.number_input("Cluster Number for K-means.",
                                   value=self.clusters) 
        self.clusters = clusters
        st.write("The number of clusters is set to ",self.clusters)
        
        return None
    
    def gmm_scree(self):
        self.set_kdata()
        X = self.kdata.copy()
        bic = []
        min_ = st.number_input("Lower Bound on Scree Plot.",value=2,key='min_gmm')
        max_ = st.number_input("Upper Bound on Scree Plot.",value=10,key = 'max_gmm')
        max_+=1
        for i in range(min_,max_):
            gm = GMM(n_components=i,n_init=1, max_iter = 300, tol=1e-4)
            gm.fit(X)
            bic.append(gm.bic(X))
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        ax.plot(range(min_,max_),bic,marker='o')
        ax.set_xlabel('Number of clusters (K)')
        ax.set_ylabel('Bayesian Information Criterion')
        ax.set_title("Determine the Number of Clusters (K)")
        y_max = ax.get_ylim()[1]
        for i in range(min_,max_):
            ax.annotate(str(i),xy=(i,bic[i-min_]+0.01*y_max))
        st.pyplot(fig)
        return None
        
    def k_means(self):
        def calc_again():
            st.session_state['calc']=True
        st.button("K-means",key='kmeans')
        if st.session_state.kmeans:
            calc_again()

        k = self.clusters

        if len(self.kdata)>0 and st.session_state.calc == True:
            k = st.number_input("Number of clusters to group data.",
                                value=self.clusters,
                                on_change = calc_again,
                                key='k')
            
            self.clusters=st.session_state.k
            X = self.kdata.copy()

            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            m,n = X.shape
            if self.seed_set:
                model = KMeans(n_clusters=k,init='random',n_init=10,
                               max_iter = 300, tol=1e-4, random_state=self.seed_value)
            else:
                model = KMeans(n_clusters=k,init='random',n_init=10,
                               max_iter = 300, tol=1e-4)
            model.fit(X_std)
            y_hat = model.predict(X_std)
            self.cluster_assignments = pd.Series(y_hat,name='cluster')
            
            pca = PCA(n_components=2)
            pca.fit(X_std)
            X_pca = pca.transform(X_std)
            
            groups = np.zeros((k,n))
            
            for i, cl in enumerate(sorted(self.cluster_assignments.unique())):
                groups[i]=X[y_hat==cl].mean(axis=0)
            groups_std = scaler.transform(groups)
            groups_pca = pca.transform(groups_std)
            st.session_state['calc']=True
            self.calculate_cluster_stats()
            
            fig,ax = plt.subplots()
            sns.scatterplot(x=X_pca[:,0],y=X_pca[:,1],hue=y_hat,palette="icefire",alpha=0.2,ax=ax)
            sns.scatterplot(x=groups_pca[:,0],y=groups_pca[:,1],hue=range(len(groups_pca))
                            ,palette="icefire",marker="P",s=50,ax=ax)
            h,l=ax.get_legend_handles_labels()
            plt.legend(h[0:k],l[0:k],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xlabel("PCA1")
            plt.ylabel("PCA2")
            plt.title("Clusters for first two Principal Components after k-Means("+str(k)+")")
            st.pyplot(fig)
            st.write("Cluster current value = ",self.clusters)
            st.session_state['calc'] = False
            

    def top_bottom(self,col):
        d = list(self.dict[col].keys())
        bottom = d[0]
        idx=-1
        top = d[idx]
        while top == 'nan' or top == 'N/A' or top =="Does not apply":
            idx-=1
            top = d[idx]
        return str(bottom)+"<->"+str(top)
    

    def calculate_cluster_stats(self):
        if len(self.kdata) >0 and st.session_state['calc']:
            data = self.kdata.copy()
            self.stats = {}
            for col in self.kdata.columns:
                self.stats[col]={}
                self.stats[col]['mean']=data[col].mean()
                self.stats[col]['std']=data[col].std()
            self.cluster_stats = {}
            self.cluster_size = {}
            groups = self.cluster_assignments
            for g in groups.unique():
                self.cluster_stats[g]={}
                temp = pd.DataFrame(data.values[groups==g],columns=self.kdata.columns)
                self.cluster_size[g]=len(temp)
                for col in self.kdata.columns:
                    try:
                        self.cluster_stats[g][col]={}
                        self.cluster_stats[g][col]['diff'] = (abs(temp[col].mean()-self.stats[col]['mean'])/
                                                                      self.stats[col]['std'])
                        self.cluster_stats[g][col]['mean']=temp[col].mean()
                        self.cluster_stats[g][col]['overall_mean']=self.stats[col]['mean']
                        self.cluster_stats[g][col]['overall_std']=self.stats[col]['std']
                        self.cluster_stats[g][col]['group_std']=temp[col].std()
                        if self.type[col] == 'ordinal':
                            self.cluster_stats[g][col]['response']=self.top_bottom(col)
                        
                    except:
                        print("Error in calculate cluster stats")
                        
    def analyze_clusters(self):
        def display_data(grp,tot):
            data = pd.DataFrame(self.cluster_stats[grp]).T.sort_values("diff",ascending=False)
            st.write(data.iloc[:tot,:])
        if len(self.kdata)> 0:
            lbl = "Select a cluster->0"+"-"+str(self.clusters-1)+"."
            grp = st.slider(lbl,
                            value = 0,
                            min_value=0,
                            max_value=self.clusters-1)
            tot = st.slider("Number of differences to display.",
                            value = 5,
                            min_value = 1,
                            max_value = len(df.columns))
            
            if hasattr(self, 'cluster_size') and hasattr(self, 'cluster_stats'):
                st.markdown(f"***:blue[Size: {self.cluster_size[grp]} || "+
                            f"BMI (mean): {self.cluster_stats[grp]['bmi']['mean']:0.3f} || "+
                            f"BMI (std): {self.cluster_stats[grp]['bmi']['group_std']:0.3f}]***")
            try:
                display_data(grp,tot)
            except:
                pass
            
    def one_one_compare(self):
        if len(self.kdata)>0:
            clusters = np.arange(self.clusters)
            grps= st.multiselect("Select two clusters to compare.",
                           options = clusters,
                           max_selections = 2)
            #msgs=[]
            if len(grps)==2:
                for grp in grps:
                    st.markdown(f"***:blue[Cluster: {grp} || "+
                        f"Size: {self.cluster_size[grp]} || " + 
                        f"BMI (mean): {self.cluster_stats[grp]['bmi']['mean']:0.3f}]***")

                tot = st.slider("Number of differences to display.",
                                value = 5,
                                min_value=1,
                                max_value = len(df.columns),
                                key = 'one_one_tot')
                
                
                
                
                include_negative = st.checkbox("Include -1",value = True)
                d1 = pd.DataFrame(self.cluster_stats[grps[0]]).T[['mean']]
                d2 = pd.DataFrame(self.cluster_stats[grps[1]]).T[['mean']]
                overall = pd.DataFrame(self.cluster_stats[grps[0]]).T[['overall_std','response']]
                diff = (np.abs(d1-d2)/overall[['overall_std']].values)
                diff.columns = ['difference']
                diff['mean1']=d1
                diff['mean2']=d2
                diff['response']=overall['response']
                diff['std']=overall['overall_std'].values
                
                if include_negative:
                    diff=diff.sort_values('difference',ascending=False)
                else:
                    diff=diff[(diff['mean1']>0)&(diff['mean2']>0)].sort_values('difference',ascending=False)
                diff.columns = ["difference","cluster"+str(grps[0]),"cluster"+str(grps[1]),'response',"std"]
                
                st.write(diff.iloc[:tot,:])

    def data_pull(self,grps,quest):
        if isinstance(quest,list) and isinstance(grps,list):
            pass
        else:
            quest = list([quest])
            grps = list(grps)
        data = self.df[quest].copy()
        data["cluster"] = self.cluster_assignments.values
        data = data[data['cluster'].isin(grps)]
        tot = data['cluster'].value_counts()
        data['len']=data['cluster'].map(tot)
        data['value']=1/data['len']
        return data
            
                
    def bar_chart(self):
        def calc_again():
            st.session_state.calc=True
        st.button("Show bar chart",key = "show_bar")
        if st.session_state.show_bar:
            calc_again()
        if len(self.kdata)>0 and (hasattr(self,'cluster_assignments')):
            col1,col2 = st.columns(2)
            all_clusters = sorted(list(self.cluster_assignments.unique()))
            all_questions = list(self.columns)
            with col1:
                clusters = st.multiselect("Select the clusters to compare.",
                                          options = all_clusters,
                                          key = 'selected_clusters')
            with col2:
                question = st.selectbox("Select the question to compare.",
                                        options = all_questions,
                                        key = 'selected_question')
            if (clusters is not None) and (question is not None):
                col3, col4 = st.columns(2)
                data = self.data_pull(clusters,question)
                x=self.dict[question]
                order = [k for k,v in sorted(x.items(),key=lambda item: item[1])]
                tbl=data.fillna("-1 None").pivot_table(values='value', 
                                                       index=question,columns='cluster',
                                                       aggfunc=sum)
                tbl=tbl.reindex(index=order)
                with col3:
                    st.write("Cluster Stats")
                    st.write(tbl)
                with col4:
                    q = st.session_state.selected_question
                    tbl2 = data.fillna("-1 None").pivot_table(values='value', 
                                                           index=q,
                                                           aggfunc=sum)
                    tbl2 = tbl2.reindex(index=order)
                    tbl2 = tbl2/len(all_clusters)
                    st.write("Overall Stats")
                    st.write(tbl2)
                tbl.columns = [int(col) for col in tbl.columns]
                if self.type[question] != "numerical":
                    try:
                        fig,ax = plt.subplots(figsize=(10,6))
                        tbl.plot.bar(ax=ax)
                        st.pyplot(fig)
                    except:
                        st.write("No data to plot for groups selected.")
                else:
                    colors = ["salmon","cadetblue","goldenrod","hotpink"]
                    fig,ax = plt.subplots()
                    for i,g in enumerate(clusters):
                        temp = data.pivot(values=question,
                                          columns='cluster')[g].dropna().reset_index(drop=True)
                        try:
                            ax.hist(temp,color=colors[i%len(colors)],
                                    alpha=1/len(clusters),label = str(g)) 
                            plt.legend()
                            st.pyplot(fig)
                        except:
                            print('No data to plot for this group.')
                            
    def density_chart(self,col,clusters):
        """
        kde_column - kde plot for comparing clusters
        
        Input:
            col - the column that you want to compare
            clusters - the clusters that you want on the chart in list format.
        """        
        df = self.newdf.copy()
        df['cluster'] = self.cluster_assignments.values
        fig,ax = plt.subplots(len(clusters),2,sharex=True)
        fig.suptitle(col)
        cmap = matplotlib.cm.get_cmap('tab20b')
        colors = [cmap(c) for c in np.linspace(0,1,len(clusters))]
        #create a boolean mask for each cluster selected
        cluster_data = [df['cluster'].isin([cluster]) for cluster in clusters]
        if len(clusters)==1:
            try:
                data = df[cluster_data[0]][col]
                data.plot(kind="density",ax=ax[0])
                ax[0].get_yaxis().set_visible(False)
                ax[1].axis('off')
                xlim=ax[1].get_xlim()
                ax[1].annotate("Cluster "+str(clusters), xy=(xlim[0],0.5))
            except:
                ax[0].axis=('off')
                xlim = ax[0].get_xlim()
                ylim = ax[0].get_ylim()
                y = (ylim[0]+ylim[1])/2
                ax[1].annotate("No data to plot",xy=(xlim[0],y))
        elif len(clusters)>1:            
            for i,clust in enumerate(cluster_data):
                data = df[clust][col]
                try:
                    # plots the line
                    data.plot(kind="density",ax=ax[i,0],color=colors[i])
                    #gets the data to plot the fill between
                    xy = ax[i,0].get_lines()[0].get_xydata()
                    ax[i,0].fill_between(xy[:,0],[0]*len(xy),xy[:,1],color=colors[i],alpha=0.3)
                except:
                    ax[i,0].axis=('off')
                    xlim = ax[i,0].get_xlim()
                    ylim = ax[i,0].get_ylim()
                    y = (ylim[0]+ylim[1])/2
                    ax[i,0].annotate("No data to plot",xy=(xlim[0],y))
                ax[i,0].get_yaxis().set_visible(False)
                ax[i,1].axis('off')
                xlim=ax[i,1].get_xlim()
                ax[i,1].set_ylim(0,1)
                ax[i,1].annotate("Cluster "+str(clusters[i]), xy=(xlim[0],0.5))
        else:
            pass
        if len(clusters)>0:    
            fig.patch.set_linewidth(5)
            fig.patch.set_edgecolor('black')
            st.pyplot(fig)
        return None
    
    def density(self):
        def calc_again():
            st.session_state.calc=True
        st.button("Show density chart",key = "show_bar_density")
        if st.session_state.show_bar:
            calc_again()
        if len(self.kdata)>0 and (hasattr(self,'cluster_assignments')):
            col1,col2 = st.columns(2)
            all_clusters = sorted(list(self.cluster_assignments.unique()))
            all_questions = list(self.columns)
            with col1:
                clusters = st.multiselect("Select the clusters to compare.",
                                          options = all_clusters,
                                          key = 'density_clusters',
                                          default = all_clusters)
            with col2:
                question = st.selectbox("Select the question to compare.",
                                        options = all_questions,
                                        key = 'density_question')
            if len(clusters)>0 and (clusters is not None) and (question is not None):
                self.density_chart(question,clusters)
                
    def gaussian_mix_model(self):
        def calc_again():
            st.session_state['calc']=True
        st.button("GMM",key='gmm_calc')
        if st.session_state.gmm_calc:
            calc_again()

        k = self.clusters
        if len(self.kdata)>0 and st.session_state.calc == True:
            k = st.number_input("Number of clusters to group data.",
                                value=self.clusters,
                                on_change = calc_again,
                                key='gmm_k')
            
            self.clusters=st.session_state.gmm_k
            X = self.kdata.fillna(0).values
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            m,n = X.shape
            if self.seed_set:
                model = GMM(n_components=k,n_init=1,
                               max_iter = 300, tol=1e-4, 
                               random_state=self.seed_value)
            else:
                model = GMM(n_components=k,n_init=1,
                               max_iter = 300, tol=1e-4)
            model.fit(X_std)
            y_hat = model.predict(X_std)
            self.cluster_assignments = pd.Series(y_hat,name='cluster')
            
            pca = PCA(n_components=2)
            pca.fit(X_std)
            X_pca = pca.transform(X_std)
            
            groups = np.zeros((k,n))
            for i, cl in enumerate(sorted(self.cluster_assignments.unique())):
                groups[i]=X[y_hat==cl].mean(axis=0)
            groups_std = scaler.transform(groups)
            groups_pca = pca.transform(groups_std)
            st.session_state['calc']=True
            self.calculate_cluster_stats()
            
            fig,ax = plt.subplots()
            sns.scatterplot(x=X_pca[:,0],y=X_pca[:,1],hue=y_hat,palette="icefire",alpha=0.2,ax=ax)
            sns.scatterplot(x=groups_pca[:,0],y=groups_pca[:,1],hue=range(len(groups_pca))
                            ,palette="icefire",marker="P",s=50,ax=ax)
            h,l=ax.get_legend_handles_labels()
            plt.legend(h[0:k],l[0:k],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xlabel("PCA1")
            plt.ylabel("PCA2")
            plt.title("Clusters for first two Principal Components after GMM("+str(k)+")")
            st.pyplot(fig)
            st.write("Cluster current value = ",self.clusters)
            st.session_state['calc'] = False
            
    def summary(self):
        if hasattr(self, 'cluster_size') and hasattr(self, 'cluster_stats'):
            clusters = np.arange(self.clusters)
            df = self.df.copy()
            kdata = self.kdata.copy()
            df['cluster']=self.cluster_assignments.values
            df=df.replace('nan',np.nan)
            label = ""
            for grp in clusters:
                na = df[df['cluster']==grp].isnull().sum().sum()
                c_size = self.cluster_size[grp]*len(df.columns)
                label+=(f"***Cluster {grp}: :blue[Size: {self.cluster_size[grp]} || "+
                            f"BMI (mean): {self.cluster_stats[grp]['bmi']['mean']:0.3f} || "+
                            f"BMI (std): {self.cluster_stats[grp]['bmi']['group_std']:0.3f} || "+
                            f"Total null values: {na} - {na/c_size*100:0.3f}%]***"+"\n\n")

            st.markdown(label)
            st.write(df)
            st.write(kdata)
            
    def pivot_analysis(self):
        def row_change():
            pass
        def col_change():
            pass
            
            
        df = self.newdf.copy()
        df['cluster'] = self.cluster_assignments.values
        rows = list(self.columns)
        cols = self.cluster_assignments.unique()
        val = list(self.columns)
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
        try:
            if len(st.session_state.pvt_row)>0 and len(st.session_state.pvt_col)>0:
                df=df[df['cluster'].isin(selected_cols)]
                tbl = df.pivot_table(values=value,
                                     index=selected_rows,
                                     columns = 'cluster',
                                     aggfunc='mean')
                st.write(tbl)
        except:
            pass
            
    
        
        
        
    def save_mapping(self, filename='survey_mapping.pkl'):
        """
        save_mapping - saves a copy of the response_map for use in 
                       future survey questions.
        get all of the unique responses from response_sets and
        assign response_map to the first val - assumes they all
        have the same response values
        """
        for key, val in self.response_sets.items():
            self.response_map[key]=self.dict[val[0]]
        self.response_map['type']=self.type
        st.download_button("Download Mapping",
                           data = pickle.dumps(self.response_map),
                           file_name = filename)
            
    def load_mapping(self,filename):
        ## load the dictionary that corresponds to response_map and type
        
        
        d = pickle.load(filename)
        
        ## look at all of the key:cols in response_sets and determine if 
        ## the response_set needs to be mapped
        for key,cols in self.response_sets.items():
            ## 
            if key in d.keys():
                for col in cols:
                    self.type[col]=d['type'][col]
                    self.dict[col]=d[key].copy()
                    self.map_values(col)
            else:
                for col in cols:
                    if len(self.dict[col]) == 0:
                        self.type[col] = 'categorical'
                        self.dict[col] = self.response_map[key]
                        self.auto_assign_values(col)
        
        for key in self.response_map.keys():
            if key in d.keys():
                self.response_map[key]=d[key]
            else:
                if key not in self.response_map.keys():
                    self.auto_assign_response(key)
                
        self.mapped_encoder = True

           
st.header("Survey Analysis Tool")

rerun = st.sidebar.button("Start over with new data.",
                          key = "rerun_")

if rerun:
    st.session_state['enc']=None
    st.session_state['calc']=True 


#we want data to be persistent, later we will want the 
if 'rerun' not in st.session_state:
    st.session_state['rerun']=False

if 'calc' not in st.session_state:
    st.session_state['calc']=True

if 'enc' not in st.session_state:
    st.session_state['enc']=None

options = ["Files",
           "Survey Data",
           "K Means",
           "Gaussian Mixed Model",
           "Analyze Clusters"]
select = st.sidebar.radio(label = "Select the tool:",
                      options = options,
                      key='sb_select')
body = "#### Video: [1_Load Data](https://youtu.be/DO_ISzSvMIE)"
st.sidebar.markdown(body)
body = "#### Video: [2_Your Data](https://youtu.be/ovkxzvYeGOg)"
st.sidebar.markdown(body)
body = "#### Video: [3_Cluster](https://youtu.be/C3NqjHwb65Y)"
st.sidebar.markdown(body)
body = "#### Video: [4_Analyze Cluster]( https://youtu.be/twTz861WH88)"
st.sidebar.markdown(body)


if select == options[0]:
    se = st.session_state['enc']
    tab1, tab2,tab3,tab4 = st.tabs(["Import Survey",
                               "Import Mapping",
                               "Map Responses",
                               "Save Mapping"])
    with tab1:
        st.subheader("Import csv File")
        if st.session_state['enc'] is None:
            uploaded_file = st.file_uploader("Select .csv survey file.",type='csv')
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file,index_col=0,keep_default_na=False)
                se = SurveyEncoder(df)
                st.session_state['enc'] = se
                st.write(se.df)
        else:
            st.write("Survey has been imported.")
    with tab2:
        st.subheader("Import previously saved mapping.")
        se = st.session_state['enc']
        if se is not None:
            if not se.mapped_encoder:
                mapping = st.file_uploader("Select a mapping.",type="pkl")
                if mapping is not None:
                    se.load_mapping(mapping)
            if se.mapped_encoder:
                st.write("Survey Encoder has been imported.")
                
    with tab3:
        st.subheader("Map Survey Responses")
        se = st.session_state['enc']
        if se is not None:
            se.col_assign_values()
            
    with tab4:
        st.subheader("Save Mapping")
        se = st.session_state['enc']
        if se is not None:
            
            filename = st.text_input("Filename:",
                          placeholder='filename.pkl')
            save_mapping_btn = st.button('Save',
                                        key = 'save_mapping_key')
            if save_mapping_btn:
                se.save_mapping(filename)

if select == options[1]:
    se = st.session_state['enc']
    tab1, tab2, tab3, tab4 = st.tabs(["Counts","Current Mapping",
                                "Question Non-Response Rate",
                                "Individual Non-Response Rate"])
    with tab1:
        st.subheader('Counts')
        if se is not None:
            se.counts()
    with tab2:
        st.subheader('Current Mapping')
        if se is not None:
            se.current_mapping()
    with tab3:
        st.subheader("Question Non-Response Rate")
        if se is not None:
            se.response_rate_question()
    with tab4:
        st.subheader("Individual Non-Response Rate")
        if se is not None:
            se.response_rate_individual()
            
        
if select == options[2]:
    se = st.session_state['enc']
    tab1,tab2,tab3,tab8 = st.tabs(["Set Seed",
                                            "Scree Plot",
                                            "K-means",
                                            "Summary"])
    with tab1:
        st.subheader("Set Seed")
        if se is not None:
            se.set_seed()
    with tab2:
        st.subheader("Scree Plot")
        if se is not None:
            se.scree_plot()
    with tab3:
        st.subheader("PCA-2 plot of K-means clusters")
        if se is not None:
            se.k_means()
    # with tab4:
    #     st.subheader("Analyze Clusters")
    #     if se is not None:
    #         se.analyze_clusters()
    # with tab5:
    #     st.subheader("One to One Analysis")
    #     if se is not None:
    #         se.one_one_compare()
    # with tab6:
    #     st.subheader("Bar Chart Comparison")
    #     if se is not None:
    #         se.bar_chart()
    # with tab7:
    #     st.subheader("Density Chart Comparison")
    #     if se is not None:
    #         se.density()
    with tab8:
        st.subheader("Summary of Clusters")
        if se is not None:
            se.summary()

if select == options[3]: #"Gaussian Mixed Model"
    se = st.session_state['enc']
    tab1,tab2,tab3,tab8 = st.tabs(["Set Seed",
                                            "Scree Plot",
                                            "GMM",
                                            "Summary"])
    with tab1:
        st.subheader("Set Seed")
        if se is not None:
            se.set_seed()
    with tab2:
        st.subheader("Scree Plot")
        if se is not None:
            se.gmm_scree()
            
    with tab3:
        st.subheader("PCA-2 plot of GMM clusters")
        if se is not None:
            se.gaussian_mix_model()
    # with tab4:
    #     st.subheader("Analyze Clusters")
    #     if se is not None:
    #         se.analyze_clusters()
    # with tab5:
    #     st.subheader("One to One Analysis")
    #     if se is not None:
    #         se.one_one_compare()
    # with tab6:
    #     st.subheader("Bar Chart Comparison")
    #     if se is not None:
    #         se.bar_chart()
    # with tab7:
    #     st.subheader("Density Chart Comparison")
    #     if se is not None:
    #         se.density()
    with tab8:
        st.subheader("Summary of Clusters")
        if se is not None and (hasattr(se,'kdata')):
            se.summary()
            
if select == options[4]:
    se = st.session_state['enc']
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Cluster Differences",
                               "One to One Analysis",
                               "Bar Chart",
                               "KDE Chart",
                               "Pivot Analysis"])
    with tab1:
        st.subheader("Cluster Differences")
        label = "The columns listed separate the cluster with the rest"
        label += "of the data."
        st.write(label)
        if se is not None and hasattr(se,'kdata'):
            se.analyze_clusters()
    with tab2:
        st.subheader("One to One Analysis")
        if se is not None:
            se.one_one_compare()
    with tab3:
        st.subheader("Bar Chart Comparison")
        if se is not None:
            se.bar_chart()
    with tab4:
        st.subheader("Density Chart Comparison")
        if se is not None:
            se.density()
    with tab5:
        st.subheader("Pivot Analysis of Clusters")
        if se is not None:
            se.pivot_analysis()
