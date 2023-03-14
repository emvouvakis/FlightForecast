import streamlit as st
import plotly.graph_objects as go
from utils import flights
from pathlib import Path

# Set the configuration of the current page in the dashboard
# Set the title of the page
# Set the layout of the page to 'wide'
# Set the icon of the page to the contents of the 'logo.png' file
st.set_page_config(page_title="FlightForecaster",layout="wide",
    page_icon=Path('logo.png').read_bytes())

# Open and read the 'style.css' file
# Display the contents of the file as an HTML style element
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize class
f = flights('LGAV')

# Get necessary dataframe for yearly filtering
dfs = f.get_dfs()
df=dfs[0]

# Create five columns in the dashboard
# Used only second, third, fourth to align the widgets nicer
col1, col2, col3, col4, col5 = st.columns(5)

# Use the second column to display a radio button widget that allows the user to enable or disable the rolling mean
with col2:
    rolling_mean=st.checkbox('Enable Rolling Mean?')
    if rolling_mean:
        roll_mean=st.selectbox(
        "Select Backstep:",
        options=[7,14,30,60]
    )

# Use the third column to display a radio button widget that allows the user to enable or disable the yearly filtering
with col3:
    enable_time_filter=st.checkbox('Enable yearly filtering?')
    if enable_time_filter:
        year_list = set(df.index.year)
        year = st.selectbox('Year', year_list, len(year_list)-1)
        df=df.loc[str(year)]

# Use the fourth column to display a radio button widget that allows the user to enable or disable the average
with col4:
    average=st.checkbox('Enable Average?')

# Create a plotly figure for arrivals
fig1 = go.Figure()

# Add a trace to the figure, consisting of a scatter plot of the df[f.airport] column of the df DataFrame
fig1.add_traces(go.Scatter(x=df.index, y=df[f.airport], name=f.airport, mode = 'lines', line=dict(color="#FB6102")))

# If the average option is selected, add a horizontal line to the figure representing the mean value of the df[f.airport] column
if average:
    fig1.add_hline(round(df[f.airport].mean(),2), name="Average",line=dict(color='red'))

# If the rolling_mean option is selected, add to the figure the rolling mean of the df[f.airport] column
if rolling_mean:
    fig1.add_traces(go.Scatter(x=df.index, y=df[f.airport].rolling(roll_mean).mean(), name=" Rolling Mean "+str(roll_mean), mode = 'lines', line=dict(color="#E6E6E6") ))

fig1.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    xaxis=dict(title=dict(text="Date", font=dict(size=18, family="Arial, sans-serif")),
     showgrid=False ),
    yaxis=dict(title=dict(text="Arrivals", font=dict(size=18, family="Arial, sans-serif")),
     showgrid=False )
    )

# Visualize plot
st.plotly_chart(fig1 ,use_container_width=True, full_screen=True)

# Create columns for plots based on season and weekday
col1, col2 = st.columns(2)

# Group data by quarter
df_seasons = df.groupby(df.index.quarter).mean()
# Set the x-axis tick labels to the names of the seasons
seasons = ['Winter', 'Spring', 'Summer', 'Fall']

# Create plot based on seasons
fig2 = go.Figure()
fig2.add_traces(go.Scatter(x=df_seasons.index, y=df_seasons[f.airport], name=f.airport, line=dict(color="#FB6102")))
fig2.update_layout(updatemenus=[], dragmode='pan',
font=dict(size=18, family="Arial, sans-serif"),
xaxis=dict(fixedrange=True, tickvals=df_seasons.index, ticktext=seasons, tickangle=-40),
yaxis=dict(showgrid=False, zeroline=False,fixedrange=True),
title={'text': "Arrivals throughout the seasons"}
)

# Visualize plot
with col2:
    st.plotly_chart(fig2 ,use_container_width=True, full_screen=True)

# Group data by weekday
df_weekday = df.groupby(df.index.weekday).mean()

# Set the x-axis tick labels to the names of the weekdays
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create plot based on weekdays
fig3 = go.Figure()
fig3.add_traces(go.Scatter(x=df_weekday.index, y=df_weekday[f.airport], name=f.airport, line=dict(color="#FB6102")))
fig3.update_layout(updatemenus=[], dragmode='pan',
font=dict(size=18, family="Arial, sans-serif"),
xaxis=dict(fixedrange=True, tickvals=df_weekday.index, ticktext=weekday_names, tickangle=-40),
yaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
title={'text': "Arrivals throughout the week"}
)

# Visualize plot
with col1:
    st.plotly_chart(fig3 ,use_container_width=True, full_screen=True)