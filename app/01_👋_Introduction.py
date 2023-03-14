import streamlit as st
from PIL import Image
import base64
from pathlib import Path
from streamlit_toggle import st_toggle_switch
import pandas as pd

# Set the configuration of the current page in the dashboard
# Set the title of the page
# Set the layout of the page to 'wide'
# Set the icon of the page to the contents of the 'logo.png' file
st.set_page_config(page_title="FlightForecaster",layout="wide",
    page_icon=Path('logo.png').read_bytes())


# Open and read the 'style.css' file
# Display the contents of the file as an HTML style element
with open('style.css') as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

@st.cache
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
    
@st.cache
def img_to_html(img_path):
    img_html = "<img width=150px, height=150px src='data:image/png;base64,{}' class='img-fluid'>".format(img_to_bytes(img_path))
    return img_html

# Add header
st.markdown("<div><h1> FlightForecaster </h1></div>""", unsafe_allow_html=True)

# Display logo
st.markdown( """
    <div style='text-align: center'>""" 
    + img_to_html('logo.png')+"""
    </div>
    """, unsafe_allow_html=True)

# Display introduction
st.markdown(
"""
<div style="text-align: justify; font-family:sans-serif">
 Welcome!

Thank you for using FlightForecaster for your data analytics needs. This project was created as part of my MBA studies in Business and Data Analytics.
This application is designed to help you predict the number of arrivals at the Athens International Airport, up to 60 days in advance.
The forecasts are used to view the performance of different predictive models.

Firstly, you can use this application to visualize and filter your data in various ways. As part of the main forecasting functionality, you can choose from a variety of exogenous variables and predictive models. Also you can enable differencing to test the accuracy of your predictions. 

I hope you find FlightForecaster to be a valuable tool in your data analysis journey. If you have any feedback or questions, please don't hesitate to reach out.

Sincerely,<br>
Emmanouil Vouvakis
</div>
""", unsafe_allow_html=True)

# Create expander for more informations
with st.expander("More Info"):

    st.header('Models')

    st.markdown(
    """
    <div style="text-align: justify; font-family:sans-serif">
    In total there are 7 different models. The selected models can be categorized as Statistical, Neural Networks and Ensemble. 
    </div>
    """, unsafe_allow_html=True)

    # Show table with the available models
    models={'Models':['AutoArima','MLP','BLSTM','TCN','HGBoost','AdaBoost','XGBoost'],
            'Category':['Statistical','Neural Network','Neural Network','Neural Network','Ensemble','Ensemble','Ensemble']}
    m=pd.DataFrame(models)
    st.dataframe(m)

    # Exogenous variables informations
    st.header('Exogenous Variables')
    st.markdown(
    """
    <div style="text-align: justify; font-family:sans-serif">
    Exogenous variables/features can be utilized in time series forecasting to increase prediction accuracy. Economic indicators, weather patterns, and occasions like holidays or sales are a few examples of exogenous variables. Exogenous variables can help a time series model better represent the underlying relationships and patterns that underlie the time series data, producing forecasts that are more precise. In this case the selected features are: <br>
    <ul>
     <li>Time ( Weekday(1-7), Quarter(1-4), Year )</li>
     <li>Quarantines ( Quarantine: 1 , No Quarantine: 0 )</li>
     <li> Covid data ( People_vaccinated )</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Sources informations
    st.header('Sources')
    st.markdown(
    """
    <div style="text-align: justify; font-family:sans-serif">
    <ul>
     <li> Covid : <a href=" https://ourworldindata.org/coronavirus/country/greece">ourworldindata.org</a></li>
     <li> Flights :  <a href="https://zenodo.org/record/7323875#.Y6QbBHZBzIV">OpenSkyNetwork</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
