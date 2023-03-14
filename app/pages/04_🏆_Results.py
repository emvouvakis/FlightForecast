import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import os

# Set the configuration of the current page in the dashboard
# Set the title of the page
# Set the layout of the page to 'wide'
# Set the icon of the page to the contents of the 'logo.png' file
st.set_page_config(page_title="FlightForecaster",layout="wide",
    page_icon=Path('logo.png').read_bytes())

with open('style.css') as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

# Read the results
data = os.path.abspath(os.path.join(os.getcwd(), os.pardir,'results','all.csv'))
df= pd.read_csv(data)

# Filter results based on models
col1, col2, col3 = st.columns(3)
with col1:
    models = st.multiselect('Models:',
        options=df.Model.unique().tolist()+['All'],
        default='All'
    )

if 'All' not in models:
    filtered = df.query('Model in @models').reset_index(drop=True)
else:
    filtered = df.copy()

# Filter results based on features
with col2:
    features = st.multiselect('Features:',
        options=df.Features.unique(),
        default='None'
    )

filtered = filtered.query('Features in @features').reset_index(drop=True)

# Filter results based on differencing
with col3:
    diff = st.multiselect('Differencing:',
        options=df.Differencing.unique(),
        default=False
    )

filtered = filtered.query('Differencing in @diff').reset_index(drop=True)

# Function that makes interactive bargraphs for the filtered results
def plots(filtered, metric):
    r=1 if len(features)<=2 else 2
    c=1 if len(features)<2 else 2

    fig = make_subplots(rows=r, cols=c, subplot_titles=features, vertical_spacing=0.40, horizontal_spacing=0.1)
    for i,group in enumerate(features):

        if i<2:
            subplot_index = (1, i+1)
        else:
            subplot_index = (2, i-1)

        legend=True if i==0 else False

        filtered_group1 = filtered[(filtered.Features==group) & (filtered.Differencing ==diff[0])]
        name1='Stationary' if diff[0] else 'Not Stationary'
        fig.add_trace(go.Bar(x=filtered_group1['Model'], y=filtered_group1[metric], name=name1, showlegend=legend, marker_color = '#FB6102'),
        row=subplot_index[0],col=subplot_index[1])

        if len(diff)>1:
            name2='Stationary' if diff[1] else 'Not Stationary'
            filtered_group2 = filtered[(filtered.Features==group) & (filtered.Differencing ==diff[1])]
            fig.add_trace(go.Bar(x=filtered_group2['Model'], y=filtered_group2[metric], name=name2, showlegend=legend, marker_color = 'blue'),
        row=subplot_index[0],col=subplot_index[1])

    for i in range(1,r+1):
        fig.update_yaxes(showgrid=False, row=i, col=1)
        fig.update_yaxes(showgrid=False, row=i, col=2)
        fig.update_xaxes(tickangle=-90, row=i, col=1)
        fig.update_xaxes(tickangle=-90, row=i, col=2)


    fig.update_layout(updatemenus=[], dragmode='pan',
        font=dict(family="Arial, sans-serif"),
        margin=dict(t=20, b=20, l=0, r=0) , 
        legend=dict(xanchor="center", yanchor="top", y=-0.2, x=0.5))
    st.plotly_chart(fig ,use_container_width=True, full_screen=True)

# Visualize results if all criteria are met
if len(models)>0 and len(features)>0 and len(diff)>0:

    # Create tabs for each metric
    tab1, tab2, tab3 = st.tabs(["MAPE", "MAE", "RMSE"])
    with tab1:
        plots(filtered, 'MAPE')
    with tab2:
        plots(filtered, 'MAE')
    with tab3:  
        plots(filtered, 'RMSE')

    # Create expander for the results shown in tabe format and download button
    with st.expander("Show table"):
        st.dataframe(filtered.reset_index(drop=True).set_index('Model'), use_container_width=True)

        # Download button for the filtered results
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(filtered.reset_index(drop=True).set_index('Model'))
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='results.csv',
            mime='text/csv',
        )

# Show Conclusions
st.markdown(f"<div id='headers'> Conclusions </div>""", unsafe_allow_html=True)
st.markdown(
"""
<div style="text-align: justify">
<p>
The first conclusion is the ineffectiveness of the differencing method (time series conversion to stationary) in improving the results of the models. 
Based on all metrics, it is observed that it significantly improved only the results of the MLP model specifically in the case where the number of vaccinated people was used as an exogenous variable.
</p>

<p>
In addition, it is worth comparing the results between the different categories of the models. 
Taking into account all three metrics for evaluating predictions on non-stationary data, it is observed that the ensemble and neural models in some cases provided better predictions compared to the statistical model. 
In more detail, using the dataset with no exogenous variables but also with time as exogenous features, it is observed that these models were evaluated better than AutoArima.
</p>

<p>
Continuing, it is worth mentioning the great efficiency of the ensemble models in non-stationary data.
More specifically, based on all three evaluation metrics, these models consistently performed better in all datasets compared to the AutoArima model.
</p>

<p>
A final conclusion is the fact that the introduction of exogenous variables did not create any significant improvement in the considered models.
</p>
</div>
""", unsafe_allow_html=True)