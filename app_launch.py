import streamlit as st
import analyze_survey as ase

pages_master = {0:[":house: Home","app_launch.py"],
         1:[":information_source: Files","pages/1_read_data.py"],
         2:[":clipboard: Survey Data","pages/2_assign_values.py"],
         3:[":books: K Means","pages/3_kmeans.py"],
         4:[":chart_with_upwards_trend: Analyze Clusters","pages/4_analyze_data.py"],
}

if 'pages_master' not in st.session_state:
    st.session_state['pages_master'] = pages_master
if 'rerun' not in st.session_state:
    st.session_state['rerun']=False
if 'calc' not in st.session_state:
    st.session_state['calc']=True
if 'survey_encoder' not in st.session_state:
    st.session_state['survey_encoder']=None
if 'pages_dict' not in st.session_state:
    st.session_state['pages_dict']={pages_master[0][0]:pages_master[0][1],
                                    pages_master[1][0]:pages_master[1][1],
                                    pages_master[2][0]:pages_master[2][1],
                                    pages_master[3][0]:pages_master[3][1],
                                    pages_master[4][0]:pages_master[4][1],
    }
if 'k' not in st.session_state:
    st.session_state['k'] = 4
if 'n_dim' not in st.session_state:
    st.session_state['n_dim']=2
if 'selected_response' not in st.session_state:
    st.session_state['selected_response']=None
if 'selected_col' not in st.session_state:
    st.session_state['selected_col'] = None


st.header("Survey Encoder Analysis Tool")
st.subheader("About")
body = """The Survey Encoder and Analysis tool was developed to assist researchers with survey 
    data by West Point's AI Data Engineering and Machine Learning (AIDE-ML) Center. 
    The focus of the tool is to allow researchers to understand the data in their 
    survey, assist them in grouping like individuals together and visualizing the 
    differences between those groups. Our focus for this tool is a csv file organized as 
    questions in the columns with question name as the first row and individual responses 
    as rows coming after the question name. 
"""
st.markdown(body)
st.subheader("File Format")
body = """The file used for this app should be one that contains the questions and 
    answers to those questions organized in rows and columns as described in the 
    figure. The essential things to understand are that the format is in the form 
    of questions in columns and responses in rows. Each nonresponse needs to have 
    the same string as its value (nan, NA, N/A are examples). The first row 
    is the row of questions which are used to identify the questions in the analysis. 
    The first column is the index used to identify the individual response to 
    the questions.
"""
st.markdown(body)
img_link ="https://images.squarespace-cdn.com/content/v1/5be5c21e75f9ee21b5817cc2/"
img_link +="262a0995-9040-4efb-9ee6-6fbd3dd4006a/csv_file_explained.png?format=1500w"
st.image(img_link,
             width=700,
             caption = "Figure 1: Example of csv survey file.")
st.subheader("`Files` page")
body = """Use the `Files` page to upload the file. Once it has been uploaded and 
    the missing value is accepted, you will 
    note that the missing values are changed to a grey `None` instead of the value 
    previously used. If you have a previously saved mapping file (.pkl) load it and note 
    the changes in the question `type`. If you have not ever mapped the responses to a 
    numerical value, you can do that in the `Map Responses` section and then 
    use the `Save Mapping` to save the mapping so you do not have to do that 
    process again.

"""
st.markdown(body)
st.subheader("`Survey Data` page")
body = """This page is used to get an idea of the response in your survey. 
    There are four tabs used to do that. The first simply gives a count of 
    the responses for each question not including the missing values. It arranges
    the responses from most to least. The `Current Mapping` tab shows the user 
    the current mapping
    of the responses, the total number of each response, and allows the researcher
    to change the mapping of a response if desired. The `Question Non-Response Rate`
    and `Individual Non-Response Rate` allow the user to investigate the effect of
    non-responses on the clustering of the data.
"""
st.markdown(body)
st.subheader("`K Means` page")
body = """This page is used to cluster the responses together in groups with 
    like responses. The first tab `Set Seed` allows the user to choose a number and
    set the seed to that number so that the groupings do not change if the data 
    stays the same and this app is used again. The `Scree Plot` tab is designed to 
    help the user determine the number of clusters to use. The `K-means` tab is a 
    plot of the PCA plot of the K-means cluster for the number of clusters selected.
    The user has the ability to choose 2 or 3 principal components to inspect. The 
    final tab `Summary` gives a summary of each cluster based on the
    column that is selected and the mapped data used in the calculation 
    of the clusters. 
"""
st.markdown(body)
st.subheader("`Analyze Clusters` page")
body = """This page is used to analyze the clusters that were created in the
    previous page. There are four tabs that allow for comparison allowing the 
    researcher to figure out what makes this cluster different or how two 
    clusters are different from each other. 
"""
st.markdown(body)
pages_dict = st.session_state['pages_dict']
ase.display_page_links(pages_dict)

rerun = st.sidebar.button("Start over with new data.",
                          key = "rerun_")

if rerun:
    st.session_state['survey_encoder']=None
    st.session_state['calc']=True
    st.session_state['k'] = 4
    st.session_state['n_dim']=2
    st.session_state['selected_response']=None
    st.session_state['selected_col'] = None