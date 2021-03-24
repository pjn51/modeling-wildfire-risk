# Distributing Wildfire Prevention Resources Using Machine Learning

## Purpose
As my capstone project for the intensive data science bootcamp 
offered by Metis, I wanted to create a flexible tool that would be 
useful for wildfireprevention and resource deployment. Wildfires are 
a growing threat, especially on the West Coast of the United States. A tool that could
predict high-risk areas would be useful so that expensive resources
such as planes and equipment could be deployed efficiently. 

## Tools and Methods
I explored multiple possible routes to solving this problem, but 
eventually settled on using an Extra Trees classifier trained on
a dataset of [climate indicators](https://www.ncdc.noaa.gov/cag/county/mapping/42/pcp/199806/12/value), [topography](https://www.ers.usda.gov/data-products/natural-amenities-scale/), and 
[coarse woody debris](https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html) 
coverage per county. 

For a target feature, I leveraged a dataset of 1.88 million wildfires
from [Kaggle](https://www.kaggle.com/rtatman/188-million-us-wildfires). 
This also allowed me to feature-engineer two lagged features to leverage
past fire behavior as a predictive feature. 

## Results
Using an Extra Trees classifier and oversampling the minority class, I was 
able to achieve precision and recall scores above 0.7 for most thresholds.

## Deployment
As a proof-of-concept, I attached this model to an interactive, web hosted 
visualization of the predictied spatial distribution of fires. I made this using Plotly and Streamlit.

# Using this Repository
If you want to check out all the steps that I took in this project, check out
`fire_predition_nuts_and_bolts.ipynb`. 

If you're interested in how I created the interactive visualization and deployed
it to the web, check out `wildfire.py`. 

Make sure to check out my slide deck and presentation that I gave for this project! You can also find my work in the [Metis Graduate Directory.](https://www.thisismetis.com/graduates)









