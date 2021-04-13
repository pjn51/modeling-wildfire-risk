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
def cutoff_creator(fire_pct, cutoff):
    if fire_pct >= cutoff:
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
                               scope="usa", title = ' ') 
                             

    fig.update_traces(hovertemplate=None, hoverinfo='skip')

    fig.update_layout(geo=dict(bgcolor = 'rgba(0,0,0,0)', lakecolor = '#0E1117'),
    				  coloraxis_showscale=False,
    				  width=1000, height=400, dragmode = False,
    				  margin={"r":250,"t":0,"l":0,"b":0})

    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.update_geos(fitbounds="locations")

    st.plotly_chart(fig)

def fire_predictor(cutoff, year, colorscale = 'magma'):
    stage_15['large_fire'] = stage_15['fire_pct'].apply(lambda x: cutoff_creator(x,cutoff)) 
    
    # split data for modeling
    x = stage_15.drop(columns=['large_fire','state','fire_pct','fips_yr'])
    y = stage_15['large_fire']

    cat_x = stage_15.loc[:, ['state']]
    cat_y = stage_15.loc[:, 'large_fire']

    ohe = OneHotEncoder(drop='first',sparse=False)
    ohe.fit(cat_x)
    ohe_x = ohe.transform(cat_x)
    columns = ohe.get_feature_names(['state'])
    ohe_x_df = pd.DataFrame(ohe_x, columns = columns, index = cat_x.index)

    x = pd.concat([x,ohe_x_df], axis=1)

    x_train_val, x_test, y_train_val, y_test = train_test_split(x,y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)
    
    # counter any class imbalance present
    ros = RandomOverSampler(random_state=0)
    x_train_val, y_train_val = ros.fit_resample(x_train_val,y_train_val)
    
    # model 
    et = ExtraTreesClassifier()
    et.fit(x_train_val, y_train_val)
    
    # create predictions for given year
    fireyear = x[x['year_x'] == str(year)]
    
    et_predictions = pd.Series(et.predict(fireyear)).reset_index().drop(columns='index')
    fireyear_fipslist = stage_15[stage_15['year_x'] == str(year)]['fips'].reset_index().drop(columns='index')
    
    pred_locations = pd.concat([fireyear_fipslist,et_predictions], axis=1).rename(columns={0:'large_fire'})
    
    # calculate performance metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, et.predict(x_test), average='binary')
        
    # output
    print('Accuracy score:',round(et.score(x_test, y_test),2))
    print('Precision:',round(precision,2))
    print('Recall:', round(recall,2))
    print('F-Score:',round(fscore,2))
    print('Total percentage of large fire obervations:', round(100*(stage_15[stage_15['large_fire']==1].shape[0]/ stage_15.shape[0]),1),'%' )
    
    print('\nPredicted distribution of burn areas exceeding',cutoff,'acres for the year,',year)
    
    county_visualizer(pred_locations, 
                      'large_fire', 
                      colorscale = colorscale, 
                      title = 'Counties predicted to exceed '+str(cutoff)+' acres burned in '+str(year))
                      
    return precision, recall

# importing data
stage_15 = pd.read_pickle('/Users/patricknorman/Documents/Python/Data/stage_15.pkl')

# streamlit stuff
st.title('Wildfire Prediction Engine')

'''
This tool allows you to see where fires are predicted to be larger
than a given threshold, in percent of county burned per year. This tool
uses weather, topography, and fuel data to make predictions.
'''

#st.plotly_chart(fire_predictor(1500, 2012, colorscale='peach'))

cut = st.sidebar.select_slider(
	label='What threshold for fire extent (percent of county burned)?',
	options=[5e-08, 5e-07, 5e-06, 5e-05, 5e-04 ])

precision, recall = fire_predictor(cut, 2014, colorscale = 'peach')

st.title('Performance Metrics')

f'''
Precision: `{round(precision,2)}` \n
Recall: `{round(recall,2)}`

_Precision_ is the proportion of true positives out of false positives. 
This tells us how confident we can be that counties we predict to 
burn more than the threshold actually will. At this cutoff, 
**{round(precision,2)*100}%** of the predicted risky counties actually burned
more than the threshold.


_Recall_ is the proportion of true positives out of false negatives.
This tells us how good we were at identifying counties that would
go on to burn more than the threshold. At this threshold, we correctly
identified **{round(recall,2)*100}%** of the counties that were destined to 
burn more than the threshold.

These metrics are calculated across the whole dataset for the specified
cutoff, while the visualization is only for a single year.
'''

st.title('How it Works')

'''
This tool has access to a database of previous wildfire locations, as well as 
climate, topography, and fuel data. Using an Extra Trees classifier, we can 
make predictions about which counties are likely to have a total burnt
area more than a given cutoff per year. For an introduction as to how these
sorts of classifiers make their predictions, see this 
[blog post](https://towardsdatascience.com/an-intuitive-explanation-of-random-forest-and-extra-trees-classifiers-8507ac21d54b).

If you'd like to see the way this model was created, 
or investigate my other projects, see my 
[GitHub!](https://github.com/pjn51/metis-project-5)
'''


