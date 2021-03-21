import streamlit as st
import pandas as pd 
import pickle
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler

# defining relevant functions
def cutoff_creator(fire_size, cutoff):
    if fire_size >= cutoff:
        return 1
    else:
        return 0

def county_visualizer(df, metric, colorscale = 'thermal', title = ''):
    '''
    Given a dataframe, a metric to visualize, this function returns a map of the counties of the US based on
    whatever metric you passed. 
    
    df: pandas dataframe to get the data from.
    metric (str): the name of the column in df that you want to visualize by county. 
    colorscale (str): the name of the built-in color scale you want to use. Defaults to the thermal scale.
    
    '''
    
    # importing shapefiles for US counties
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/AmyB47/household_income_regression/main/json/geojson-counties-fips.json') as response:
        counties = json.load(response)

    # plotting figure
    import plotly.express as px

    fig = px.choropleth(df, geojson=counties, locations='fips', color=metric,
                               color_continuous_scale=colorscale,
                               scope="usa", title = title) 
                             

    fig.update_traces(hovertemplate=None, hoverinfo='skip')

    fig.update_layout(geo=dict(bgcolor = 'rgba(0,0,0,0)', lakecolor = '#0E1117'),
    				  coloraxis_showscale=False,
    				  width=1000, height=1000, dragmode = False,
    				  margin={"r":0,"t":0,"l":0,"b":0})

    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.update_geos(fitbounds="locations")

    st.plotly_chart(fig)

def fire_predictor(cutoff, year, colorscale = 'magma'):
    stage_13['large_fire'] = stage_13['FIRE_SIZE'].apply(lambda x: cutoff_creator(x,cutoff)) 
    
    # split data for modeling
    x = stage_13.drop(columns=['FIRE_SIZE','large_fire','fips','FIPS_x','prev_id','name','state','fips_yr'])
    y = stage_13['large_fire']

    cat_x = stage_13.loc[:, ['state']]
    cat_y = stage_13.loc[:, 'large_fire']

    ohe = OneHotEncoder(drop='first',sparse=False)
    ohe.fit(cat_x)
    ohe_x = ohe.transform(cat_x)
    columns = ohe.get_feature_names(['state'])
    ohe_x_df = pd.DataFrame(ohe_x, columns = columns, index = cat_x.index)

    x = pd.concat([x,ohe_x_df], axis=1)
    
    # counter any class imbalance present
    ros = RandomOverSampler(random_state=0)
    x, y = ros.fit_resample(x,y)

    x_train_val, x_test, y_train_val, y_test = train_test_split(x,y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)
    
    # model 
    et = ExtraTreesClassifier()
    et.fit(x_train_val, y_train_val)
    
    # create predictions for given year
    fireyear = x[x['year_x'] == str(year)]
    
    et_predictions = pd.Series(et.predict(fireyear)).reset_index().drop(columns='index')
    fireyear_fipslist = stage_13[stage_13['year_x'] == str(year)]['fips'].reset_index().drop(columns='index')
    
    pred_locations = pd.concat([fireyear_fipslist,et_predictions], axis=1).rename(columns={0:'large_fire'})
    
    # calculate performance metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, et.predict(x_test), average='binary')
        
    # output
    print('Accuracy score:',round(et.score(x_test, y_test),2))
    print('Precision:',round(precision,2))
    print('Recall:', round(recall,2))
    print('F-Score:',round(fscore,2))
    print('Total percentage of large fire obervations:', round(100*(stage_13[stage_13['large_fire']==1].shape[0]/ stage_13.shape[0]),1),'%' )
    
    print('\nPredicted distribution of burn areas exceeding',cutoff,'acres for the year,',year)
    
    county_visualizer(pred_locations, 
                      'large_fire', 
                      colorscale = colorscale) 
                      



# importing data
stage_13 = pd.read_pickle('/Users/patricknorman/Documents/stage_13.pkl')



# streamlit stuff
st.title('Wildfire Prediction Engine')

'''
This tool allows you to see where fires are predicted to be larger
than a given threshold, in total acres burned per year. This tool
uses weather, topography, and fuel data to make predictions.
'''

#st.plotly_chart(fire_predictor(1500, 2012, colorscale='peach'))

cut = st.sidebar.number_input(
	'What threshold for fire extent (acres per county)?',
	min_value = 1, max_value = 500_000, value = 150)

fire_predictor(cut, 2014, colorscale = 'peach')

st.title('How it Works')

'''
This tool has access to a database of previous wildfire locations, as well as 
climate, topography, and fuel data. Using an Extra Trees classifier, we can 
make predictions about which counties are likely to have a total burnt
area more than a given cutoff per year. 
'''


