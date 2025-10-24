import streamlit as st
import analyze_survey as ase
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pickle
#from IPython.display import display, Image, HTML
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
os.environ["OMP_NUM_THREADS"] = "4"

class SurveyEncoder(object):
    def __init__(self,df,mapping=False,col_diff = None, missing_values = 11111):
        plt.style.use('ggplot')
        df = df.replace('',np.nan,regex=True)
        if missing_values != 11111:
            df = df.replace(missing_values,np.nan)
        self.df = df.copy()
        self.shape = self.df.shape
        self.columns = list(df.columns)
        self.col_diff = col_diff
        self.dict = {col:dict() for col in df.columns}
        self.responses = {}
        self.response_sets = {}
        self.response_map = {}
        self.type = {col:"categorical" for col in df.columns}
        self.newdf = pd.DataFrame([],columns = df.columns)
        self.clusters = 4
        self.min_max = (0,0)
        self.pca_plot_configs = (0,0,0)
        self.seed_value = 42
        self.seed_set = False
        self.scree_plot_=None
        self.compare_col = None
        ### Run at the start for all objects ####
        self.col_assign_types()
        self.definitions = {}
        self.mapped_encoder = False
        if mapping:
            #set to False if you are going to bring in your own mapping
            if len(self.response_sets)>0:
                self.col_assign_values()
    
    def col_assign_values(self):
        """
        assigns values that are in the self.response_map to the columns
            in the self.response_sets by changing the self.dict to be the
            assignment in self.response_map. Also, maps these values to 
            self.newdf. This function does this for all responses in 
            self.response.sets. 
        """
        for key,cols in self.response_sets.items():
            for col in cols:
                self.dict[col] = self.response_map[key]
                self.responses[col]=self.top_bottom(col)
                self.type[col] = 'categorical'
                self.map_values(col)


    def map_values(self,col):
        """
        map_values - for a col in the object, this method maps
            the values that are in self.dict to the dataframe 
            self.newdf.

        col : string
            the column of the question that needs to be mapped.

        RESULT: changes self.newdf to a mapping using self.dict.
        """
        dict1 = self.dict[col]
        self.newdf[col]=self.df[col].map(dict1)
    

    def auto_assign_response(self,response):
        """
        auto_assign_response - automatically assigns nominal numeric values 
            to all columns in a response set and creates a dict of col values
            that can be updated.

        ONLY USED IN LOAD MAPPING WHEN THERE IS NOT A RESPONSE IN THE MAPPING 
            BEING LOADED - why? bringing in a response map that may not have all
            of the responses in the current 
        """
        self.response_map[response]={}
        dict1 = {v:k for k,v in enumerate(response)}
        if 'nan' in dict1.keys():
            dict1['nan']=-1
        self.response_map[response]=dict1
        cols = self.response_sets[response]
        for col in cols:
            self.dict[col]=dict1
            self.responses[col]=self.top_bottom(col)
            self.type[col]='categorical'
            self.map_values(col)
        
    def auto_assign_values(self,col):
        """
        auto_assign_values - automatically assigns nominal numeric values to columns
                            and creates a dict of col values that can be updated later
                            with `assign_values`.

        THIS METHOD CHANGES A COLUMN ONLY TO NOMINAL VALUES ONLY USED IN MAPPING BEING
            LOADED. 
        """

        dict1 = {v:k for k,v in enumerate(self.dict[col].keys())}
        if 'nan' in dict1.keys():
            dict1['nan']=-1
        self.dict[col]=dict1
        self.responses[col]=self.top_bottom(col)
        self.map_values(col)

    def reassign_type(self,response,q_type):
        """
        reassigns the type passed in to all questions that are in the
            response_sets
        """
        cols = self.response_sets[response]
        for col in cols:
            self.type[col] = q_type



    def reassign_values(self,response,values):
        """
        assign_values - changes the `response` to the values passed for all questions
        Input: response - the responses for the columns in the order that matches
                        the self.response_map.
                values - the value for each response in the same order as 
                        the self.response_map
                q_type - the new type for the selected columns.

        THIS METHOD CHANGES ALL OF THE RESPONSES TO THE VALUES PASSED.
        """
        if response in self.response_map.keys():
            map_ = self.response_map[response]
            map_ = self.order_dict(map_)
        else:
            return None #write code to create a response_map and response_set
        
        for i,key in enumerate(map_.keys()):
            self.response_map[response][key]=values[i]
        for col in self.response_sets[response]:
            self.dict[col]=self.response_map[response].copy()
            self.responses[col]=self.top_bottom(col)
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
            3) creates a response_map dictionary.
        """
        df = self.df.copy()
        for col in self.columns:
            try:
                ## if the column are numbers except the 'nan' values, then 
                ## give that column an np.float type which will eventually
                ## be translated below as a numerical column
                df[col]=df[col].astype(np.float64)
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
                #self.newdf[col]=self.df.loc[:,col].replace('nan',np.nan)
                self.newdf[col]=pd.to_numeric(self.df.loc[:,col],errors='coerce')
        self.df = df.copy()
                
        if len(self.response_sets)>0:
            self.current_response = list(self.response_sets.keys())[0]
            self.variable_type = self.type[self.response_sets[self.current_response][0]]
        else:
            self.current_response = None
            self.variable_type = 'numerical'

    def __repr__(self):
        res = pd.DataFrame(self.type.values(),index=self.columns,columns=['type'])
        res['missing_values']=self.df.isna().sum()
        res['mean'] = self.newdf.mean()
        res['std'] = self.newdf.std()
        res['max'] = self.newdf.max()
        res['min'] = self.newdf.min()
        try:
            pd.set_option('display.float_format', '{:,.2f}'.format)
            res = str(res)
            pd.reset_option('display.float_format')
            res+=str(self.summarize_type())
            return ""
        except:
            return repr(res)
        
    def __call__(self,num_rows=0):
        if num_rows == 0:
            num_rows = len(self.columns)
        res = pd.DataFrame(self.type.values(),index=self.columns,columns=['type'])
        res['missing_values']=self.df.isna().sum()
        res['mean'] = self.newdf.mean()
        res['std'] = self.newdf.std()
        res['max'] = self.newdf.max()
        res['min'] = self.newdf.min()
        try:
            pd.set_option('display.float_format', '{:,.2f}'.format)
            pd.set_option('display.max_rows', None)
            res = str(res.head(num_rows))
            pd.reset_option('display.float_format')
            pd.reset_option('display.max_rows')
            return res
        except:
            return repr(res)
        
    def __len__(self):
        return len(self.df)
    
    def non_numeric_cols(self):
        cols = []
        for col in self.columns:
            if self.type[col] != 'numerical':
                cols.append(col)
        return cols
    
        

    ###################################################################################
    ###  General Information ##########################################################
    ###################################################################################
    def top_bottom(self,col):
        d = sorted(self.dict[col].items(),key = lambda kv:kv[1])
        d_num = [e[0]+'('+str(e[1])+')' for e in d]
        top = d_num[-1]
        idx=0
        bottom = d[idx][0]
        while bottom in ['nan','N/A',
                         'Does not apply','.']:
            idx+=1
            bottom = d[idx][0]
        bottom = d_num[idx]
        return str(bottom)+"<->"+str(top)
             
                
    def order_dict(self,d):
        """
        order_dict - orders a dictionary by the item values instead of the
                    random order that a dictionary usually orders them.
        Input: d - dictionary with keys and items
        Output dictionary ordered by value order, either alphabetically or
                    numerically.
        """
        return {k:v for k,v in sorted(d.items(),key=lambda item: item[1])}
    
    def order_col_responses_by_mapping(self,col):
        """
        returns the responses from self.dict in the order lowest to highest
            according to the mapping for a column.
        """
        order = self.order_dict(self.dict[col])
        response_order = order.keys()
        return list(response_order)
    
    def current_mapping(self,col):
        """
        current_mapping - shows the current mapping from smallest to largest
                    value and gives the ability for the user to change the
                    mapping for each question/column.
        """ 
        curr_map = self.order_dict(self.dict[col])
        df = pd.DataFrame(curr_map.values(),index=curr_map.keys(),columns=['map'])
        counts = self.df[col].replace(np.nan,'nan').value_counts()
        df['count']=counts
        df.loc['TOTAL',:]=[counts.sum(),counts.sum()]
        try:
            df['map']=df['map'].astype(int)
            df['count']=df['count'].astype(int)
        except:
            pass
        return df
    
    def counts(self,col):
        """
        counts - displays the count for each response of the selected question.
        """
        return self.df[col].value_counts()
    
    def summarize_type(self):
        return pd.DataFrame(self.type.values(),index=self.type.keys(),
                     columns = ['type'])['type'].value_counts()
    
    def response_rate_question(self,tot='not_passed'):
        if tot == 'not_passed':
            tot = 1
        df = self.df.copy()
        df=df.replace('nan',np.nan,regex=True)
        non_response = (df.isnull()).sum(axis=0)
        non_response.name = "null"
        non_response = pd.DataFrame(non_response)
        non_response["%"]=non_response['null']/len(df)
        return non_response[non_response['null']>=tot]
    
    def response_rate_individual(self,tot='not_passed'):
        if tot == 'not_passed':
            tot = 1
        df = self.df.copy()
        df = df.replace('nan',np.nan,regex=True)
        non_response = (df.isnull()).sum(axis=1)
        non_response.name = "null"
        non_response = pd.DataFrame(non_response)
        non_response["%"]=non_response['null']/len(df)
        return non_response[non_response['null']>=tot]

    
    ##########################################################################
    #############  K-Means / GMM #############################################
    ##########################################################################

    def set_seed(self, seed=42, seed_set = True):
        self.seed_set = seed_set
        self.seed_value = seed
    
    def set_kdata(self):
        cols = [col for col in self.columns 
                if self.type[col]=='ordinal' or 
                self.type[col]=='numerical']
        data = self.newdf.loc[:,cols].copy()
        for col in data.columns:
            data[col] = (pd.to_numeric(data[col],errors='coerce')
                         .fillna(-1).infer_objects(copy=False))
        self.kdata = data.copy()
        self.k_columns = self.kdata.columns
        if self.compare_col == None:
            self.compare_col = self.k_columns[0]

        return self.kdata.copy()
    
    def scree_plot(self,min,max):
        if (min,max) == (self.min_max[0],self.min_max[1]):
            return self.scree_plot_
        else:
            self.min_max = (min,max)
        X = self.set_kdata()
        distortion = []
        max += 1
        for i in range(min,max):
            km = KMeans(n_clusters = i, init='random',
                        n_init=10,max_iter=300,tol=1e-4)
            km.fit(X)
            distortion.append(km.inertia_)
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        ax.plot(range(min,max),distortion,marker='o')
        ax.set_xlabel('Number of clusters (K)')
        ax.set_ylabel('Distortion')
        ax.set_title("Determine the Number of Clusters (K)")
        y_max = ax.get_ylim()[1]
        for i in range(min,max):
            ax.annotate(str(i),xy=(i,distortion[i-min]+0.01*y_max))
        self.scree_plot_ = fig
        return fig
    
    def kmeans_cluster(self,k=0):
        if k<=1:
            k = self.clusters
        else:
            self.clusters = k
        X = self.set_kdata()
        self.scaler_model = StandardScaler()
        X_std = self.scaler_model.fit_transform(X)
        
        if self.seed_set:
            model = KMeans(n_clusters=k,init='random',n_init=10,
                                max_iter = 300, tol=1e-4, 
                                random_state=self.seed_value)
        else:
            model = KMeans(n_clusters=k,init='random',n_init=10,
                            max_iter = 300, tol=1e-4)
        model.fit(X_std)
        y_hat = model.predict(X_std)
        self.cluster_assignments = pd.Series(y_hat,
                                             index = self.kdata.index,
                                             name='cluster')
        self.calc_cluster_stats()
        return self.cluster_assignments.values
    
    def attach_cluster(self, data):
        """
        attaches the cluster to the passed data so that it can be used for other
            analysis
        """
        data = data.copy()
        if np.all(data.index == self.cluster_assignments.index):
            data['cluster']=self.cluster_assignments.copy()
            return data
        else:
            X = self.cluster_assignments.copy().reset_index(drop=True)
            data = data.reset_index(drop=True)
            data['cluster']=X
            return data
    
    
    def calc_cluster_stats(self,include_missing=True):
        """
        calculates the mean and standard deviation of all k_columns and gives the
            difference between each cluster and the rest of the data.

        Output: diff - a dataframe with the columns used in the clustering
            and the rows as the number of clusters.
        """
        def mean(x):
            return x.mean()
        def std(x):
            return x.std()
        if include_missing:
            data = self.kdata.copy().reset_index(drop=True)
            data = self.attach_cluster(data)
        else:
            data = self.kdata.copy().replace(-1,np.nan).reset_index(drop=True)
            data = self.attach_cluster(data)
        self.cluster_mean = data.groupby(['cluster']).apply(mean,include_groups=False)
        self.cluster_std = data.groupby(['cluster']).apply(std,include_groups=False)
        self.diff = ((self.cluster_mean-data.mean())/data.std())
        self.diff = self.diff.drop('cluster',axis=1)
        self.diff = self.diff[self.k_columns]
        return self.diff

    def PCA(self,n_comp=2):
        """
        computes PCA components for n_components used in kdata.

        Input: n_comp - number of principal components to calculate.
        Output: numpy array for the n_dim principal components for kdata
        """
        X = self.set_kdata()
        X_std = self.scaler_model.transform(X)
        self.pca_model = PCA(n_components=n_comp)
        self.pca_model.fit(X_std)
        X_pca = self.pca_model.transform(X_std)
        self.X_pca = X_pca
        
        return X_pca
    
    def PCA_plot(self,k=0,n_dim=2,alpha=0.4):
        if self.pca_plot_configs == (k,n_dim,alpha):
            return self.pca_plot_fig
        if k <= 1:
            k = self.clusters
        else:
            self.clusters = k
        y_hat = self.kmeans_cluster(k)
        X_pca = self.PCA(n_dim)
        scaler = self.scaler_model
        pca = self.pca_model

        groups = self.cluster_mean.copy()

        groups_std = scaler.transform(groups)
        groups_pca = pca.transform(groups_std)
        
        if n_dim == 2:

            df = pd.DataFrame(X_pca,columns = ['pca1','pca2'])
            df = self.attach_cluster(df)
            fig = px.scatter(df,x='pca1',y='pca2',color = 'cluster',opacity = alpha)
            fig.add_trace(go.Scatter(x=groups_pca[:,0],
                                       y=groups_pca[:,1],
                                       showlegend=False,
                                       mode='markers',
                                       marker=dict(
                                           size=12,
                                           symbol = 'cross',
                                           color=np.arange(k),
                                       )

            ))
        if n_dim > 2:
            pio.renderers.default = 'iframe'
            pio.templates.default = "plotly"
            df = pd.DataFrame(X_pca,columns = ['pca1','pca2','pca3'])
            df = self.attach_cluster(df)
            fig =  px.scatter_3d(df,x='pca1',y='pca2',z='pca3',color='cluster',
                                 hover_data=[df.index],
                                 opacity=alpha)

            fig.add_trace(go.Scatter3d(x=groups_pca[:,0],
                                       y=groups_pca[:,1],
                                       z=groups_pca[:,2],
                                       showlegend=False,
                                       mode='markers',
                                       marker=dict(
                                           size=12,
                                           symbol = 'cross',
                                           color=np.arange(k),
                                       )))
        self.pca_plot_configs = (k,n_dim,alpha)
        self.pca_plot_fig = fig
            
        return fig
    
    def cluster_summary(self,col,grps=None):
        def count(x):
            return len(x)
        def null_(x):
            return x.isna().sum()
        def make_pretty(styler):
            cols = ['size','std_'+col,'%_null']
            styler.set_caption("Cluster Summary")
            styler.set_table_styles([
                {'selector':'th.column_heading',
                 'props':[('font-weight','bold'),('font-size','20pt')]}
            ])
            styler.set_properties(**{'background-color': 'lightblue'})
            styler.format({'mean_'+col:'{:.3f}','std_'+col:'{:.3f}','%_null':'{:.2f}%'})

            return styler
        
        if grps == None:
            grps = np.arange(self.clusters)
        elif isinstance(grps,int) or isinstance(grps,str):
            grps=[grps]
        m,n = self.kdata.shape
        X = self.kdata.copy().reset_index(drop=True).replace(-1,np.nan)
        X = self.attach_cluster(X)
        summary = X.groupby('cluster').apply(count).to_frame()
        summary.columns = ['size']
        summary['mean_'+col] = self.cluster_mean.loc[:,col]
        summary['std_'+col] = self.cluster_std.loc[:,col]
        summary['total_null']=X.groupby('cluster').apply(null_).sum(axis=1)
        summary['%_null']=summary['total_null']/(summary['size']*n)*100
        summary = summary.loc[grps]
        return make_pretty(summary.style)   


    ##########################################################################
    #############  Analyze / Clusters  #######################################
    ##########################################################################

    
    def differentiate_cluster(self,cluster,top=10,include_missing=True):
        self.calc_cluster_stats(include_missing=True)
        data = self.diff.loc[cluster,:].to_frame()
        data.columns = ['SD_from_mean']
        data['mean'] = self.cluster_mean.loc[cluster]
        data['abs'] = np.abs(data['SD_from_mean'])
        data['response']=data.index.map(self.responses)
        if not(include_missing):
            # this is better than just recalculating with include_missing=False
            data = data[data['mean']>0]
        data = data.sort_values('abs',ascending=False).iloc[:top]
        data = data.drop('abs',axis=1)
        return data.style.background_gradient(subset=['SD_from_mean'],
                                              cmap='coolwarm',vmin=-1,vmax=1)
    
    def data_pull(self,col,grps=None,pull=None, include_missing = True):
        """
        pulls all of the data for a particular column and groups. If pull is not 
            specified, then the data returned is prior to mapping, if pull is
            anything but 0, then the newdf (or mapped data) is returned
        """
        if grps is None:
            grps = np.arange(self.clusters)
        if pull is None:
            X = self.attach_cluster(self.df.replace(np.nan,'nan')).copy()
            if not(include_missing):
                X = X[X[col]!='nan']
        else:
            X = self.attach_cluster(self.newdf.fillna(-1)).copy()
            if not(include_missing):
                X = X[X[col]>=0]
        X = X[X['cluster'].isin(grps)]
        return X[[col,'cluster']]
    
    def col_distribution_by_cluster(self,col,grps = None):
        if grps == None:
            grps = np.arange(self.clusters)
        
        if self.type[col] != 'numerical':
            def distn(x,**kwargs):
                denominator = len(x)
                res = x.value_counts().to_frame()
                res['dist'] = res/denominator
                return res
            ## Distribution by cluster
            X = self.data_pull(col,grps) #pull all clusters
            X['ones'] = 1
            temp = X[[col,'cluster']].groupby('cluster').apply(distn)
            temp=temp[['dist']].pivot_table(index=col,columns = 'cluster',
                                    aggfunc='mean').fillna(0).loc[:,'dist']
            
            ## Overall distribution
            temp2 = X[[col,'ones']].groupby('ones').apply(distn)['dist']
            index = temp2.index
            idx = [ind[1] for ind in index]
            temp2 = pd.DataFrame(temp2.values,index=idx, columns=['overall'])

            ## Merge the two dataframes
            temp = pd.merge(temp,temp2,left_index=True,right_index=True)
            
            ## Get the order correct
            cols_ordered = self.order_col_responses_by_mapping(col)
            for col in cols_ordered:
                if col not in temp.index:
                    cols_ordered.remove(col)

            return temp.loc[cols_ordered]
        else:
            ## Distribution of 10,25,50,75 and 90 percentile
            def distn(x,**kwargs):
                col = kwargs['col']
                return x[col].quantile(q=[0.1,0.25,0.5,0.75,0.9]).T
            X = self.data_pull(col,grps,pull=0) #pull all clusters
            X['ones'] = 1
            temp = X[[col,'cluster']].groupby('cluster').apply(distn,**{'col':col}).T
            temp2 = X[[col,'ones']].groupby('ones').apply(distn,**{'col':col}).T
            temp2.columns = ['overall']
            temp = pd.merge(temp,temp2,left_index=True,right_index=True)
            return temp

    def display_col_distn_by_cluster(self,col,grps=None,alpha=0.5):
        if isinstance(grps,int):
            grps=[grps]
        if grps is None:
            grps = list(np.arange(self.clusters))
        
        if self.type[col] != 'numerical':
            fig,ax = plt.subplots(figsize=(10,6))
            grps.append('overall')
            X = self.col_distribution_by_cluster(col,grps)
            X[grps].plot.bar(ax=ax)
            return fig
        else:
            fig,ax = plt.subplots(figsize=(10,6))
            X = self.data_pull(col,grps,pull=0)
            sns.boxenplot(data=X,x=col,hue='cluster',alpha=alpha)
            return fig
           
    def density_chart(self,col,grps):
        """
        kde_column - kde plot for comparing clusters
        
        Input:
            col - the column that you want to compare
            grps - the clusters that you want on the chart in list format.
        """
        if isinstance(grps,int):
            grps=[grps]
         
        df = self.data_pull(col,grps,pull=1)
        fig,ax = plt.subplots(len(grps),figsize=(12,2*len(grps)),sharex=True)
        fig.suptitle("Kernel Density Estimator")
        cmap = matplotlib.cm.get_cmap('tab20b')
        colors = [cmap(c) for c in np.linspace(0,1,len(grps))]
        if len(grps)==1:
            try:
                cl = grps[0]
                sns.kdeplot(df,x=col,ax=ax,fill=True,alpha=0.4,
                                label="cluster"+str(cl))
                ax.get_yaxis().set_visible(False)
             
            except:
                ax.axis=('off')
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                y = (ylim[0]+ylim[1])/2
                ax.annotate("No data to plot",xy=(xlim[0],y))
        elif len(grps)>1:            
            for i,cl in enumerate(grps):
                data = df[df['cluster']==cl]
                if len(data)>0:
                    # seaborn to plot kdeplot
                    sns.kdeplot(data,x=col,ax=ax[i],fill=True,alpha=0.4,
                                label="cluster"+str(cl),
                                color=colors[i])

                else:
                    ax[i].axis=('off')
                    xlim = ax[i].get_xlim()
                    ylim = ax[i].get_ylim()
                    y = (ylim[0]+ylim[1])/2
                    ax[i].annotate("No data to plot",xy=(xlim[0],y))
                ax[i].get_yaxis().set_visible(False)
                ax[i].tick_params(axis='x', labelbottom=False)
                ax[i].legend(loc='best')
                if cl == grps[-1]:
                    ax[i].spines['bottom'].set_visible(True)
                    ax[i].xaxis.set_ticks_position('bottom')
        else:
            pass
        if len(grps)>0:    
            fig.patch.set_linewidth(5)
            fig.patch.set_edgecolor('black')
            
        return fig
    
    def one_to_one_analysis(self,grp1,grp2,top=10,include_missing=False):
        def highlight_higher_of_two(row):
            clr = 'background-color: lightgreen'
            styles = [''] * len(row)  # Initialize with empty styles for all cells
            col_a_val = row[grp1]
            col_b_val = row[grp2]

            if col_a_val > col_b_val:
                styles[df.columns.get_loc(grp1)] = clr
            elif col_b_val > col_a_val:
                styles[df.columns.get_loc(grp2)] = clr
            return styles
        self.calc_cluster_stats(include_missing=True)
        Z = self.cluster_mean.loc[[grp1,grp2],:].T
        Z['diff'] = Z.loc[:,grp1]-Z.loc[:,grp2]
        Z['std']=(self.cluster_std.loc[grp1,:]+self.cluster_std.loc[grp2,:])/2
        Z['SD_diff']=Z['diff']/Z['std']
        Z['abs']=Z['SD_diff'].abs()
        
        if not(include_missing):
            Z = Z[(Z[grp1]>0)&(Z[grp2]>0)]
        Z = Z.sort_values('abs',ascending=False)
        Z = Z.drop(['diff','std','abs'],axis=1)
        Z['response'] = Z.index.map(self.responses)
        df = Z.iloc[:top,:]
        return df.style.apply(highlight_higher_of_two,axis=1)


    
            
    def load_mapping(self,filename):
        ## load the dictionary that corresponds to response_map and type
        
        if isinstance(filename, str):
            with open(filename,'rb') as pickle_file:
                d = pickle.load(pickle_file)
        else:
            d = pickle.load(filename)
        
        ## look at all of the key:cols in response_sets and determine if 
        ## the response_set needs to be mapped
        for key,cols in self.response_sets.items():
            ## 
            if key in d.keys():
                for col in cols:
                    self.type[col]=d['type'][col]
                    self.dict[col]=d[key].copy()
                    self.responses[col]=self.top_bottom(col)
                    self.map_values(col)
            else:
                for col in cols:
                    if len(self.dict[col]) == 0:
                        self.type[col] = 'categorical'
                        self.dict[col] = self.response_map[key]
                        self.responses[col]=self.top_bottom(col)
                        self.auto_assign_values(col)
        
        for key in self.response_map.keys():
            if key in d.keys():
                self.response_map[key]=d[key]
            else:
                if key not in self.response_map.keys():
                    self.auto_assign_response(key)
                
        self.mapped_encoder = True
        return d
    
    def save_mapping(self):
        """
        save_mapping(): creates a dictionary to pickle so it can be
            unpickled later and restore the object back into its configuration
        """
        save_map = self.response_sets.copy()
        for key,val in save_map.items():
            if key not in save_map.keys():
                save_map[key] = self.dict[val[0]]
        save_map['type']=self.type
        return save_map
        