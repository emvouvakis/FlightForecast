import streamlit as st
from pathlib import Path
from utils import flights
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Set the configuration of the current page in the dashboard
# Set the title of the page
# Set the layout of the page to 'wide'
# Set the icon of the page to the contents of the 'logo.png' file
st.set_page_config(page_title="FlightForecaster",layout="wide",
    page_icon=Path('logo.png').read_bytes())

# # Open and read the 'style.css' file
# # Display the contents of the file as an HTML style element
with open('style.css') as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

# Initialize class
f = flights('LGAV')

#Clear results from revious executions
for key in f.res:
    f.res[key].clear()
# f.results=pd.DataFrame()

# Create a list of options for the selection box
# These are the variations of the exogenous features 
features = ['No Exogenous','Time','Quarantines','Covid']

# Split the streamlit dashboard into two columns
col1, col2 = st.columns(2)

# Create a selection box in the first column
# The options are the elements of the 'features' list
# The default selection is 'No Exogenous'
with col1:
    df__ = st.selectbox('Exogenous:',
        options=features,
        index=0
    )

# Get selected dataframe
df_ = f.get_dfs()[features.index(df__)]
times = list()

# Create a multi-select box in the second column
# The options are the elements of the 'options' list
# The default selection is 'MLPRegressor'
with col2:
    models = st.multiselect('Models:',
        options=['AutoArima','XGBoost','AdaBoost','HGBoost','MLP','BDLSTM','TCN'],
        default='MLP'
    )

# Create a radio button group in the second column
# The options are 'True' and 'False'
with col2:
    diff_ = st.radio('Enable differencing:',
    options=[True,False])

if diff_:
    diff = [True, False]
else:
    diff = [False]

# Increment the global progress bar by a value calculated based on the number of elements in the 'models' and 'diff' lists
def incrementBar():
    global bar, models, diff, value
    value += ( 1 / (len(models)*len(diff)) )
    bar.progress(value)

# Initialize progress bar
value=0
bar = st.progress(value)

for differencing in diff:

    title = df_.name
    df=df_.copy()
    
    if differencing:
        if title != 'Covid data as exog':
            df[f.airport] = df[f.airport].diff()
        else:
            df = df.diff()

    # Create training and testing datasets using the 'add_lags' function
    x_train, y_train, x_test, y_test = f.add_lags(df, past=7, test_size=60)
    
    # If the corresponding model is selected
    # Train the model and get the predictions and elapsed time
    # Save the elapsed time
    # If differencing was applied, invert the predictions
    # Update the progress bar
    if 'HGBoost' in models:
        hgboostResults, elapsed_time = f.hgboost(x_train, y_train, x_test)
        times.append(elapsed_time)
        if differencing:
            hgboostResults = f.invert_predictions(hgboostResults)
        f.save_results(f.truth, hgboostResults, title, differencing)
        incrementBar()

    if 'AdaBoost' in models:
        adaboostResults, elapsed_time = f.adaboost(x_train, y_train, x_test)
        times.append(elapsed_time)
        if differencing:
            adaboostResults = f.invert_predictions(adaboostResults)
        f.save_results(f.truth, adaboostResults, title, differencing)
        incrementBar()
    
    if 'MLP' in models:
        mlpResults, elapsed_time = f.mlp(x_train, y_train, x_test, (14,7))
        times.append(elapsed_time)
        if differencing:
            mlpResults = f.invert_predictions(mlpResults)
        f.save_results(f.truth , mlpResults, title, differencing)
        incrementBar()

    if 'TCN' in models:
        tcnResults, elapsed_time = f.tcn(x_train, y_train, x_test, (30,7))
        times.append(elapsed_time)
        if differencing:
            tcnResults = f.invert_predictions(tcnResults)
        f.save_results(f.truth , tcnResults, title, differencing)
        incrementBar()
    
    if 'BDLSTM' in models:
        bdlstmResults, elapsed_time = f.bdlstm(x_train, y_train, x_test, (14,7))
        times.append(elapsed_time)
        if differencing:
            bdlstmResults = f.invert_predictions(bdlstmResults)
        f.save_results(f.truth , bdlstmResults, title, differencing)
        incrementBar()

    if 'XGBoost' in models:
        xgboostResults, elapsed_time = f.xgboost(x_train, y_train, x_test)
        times.append(elapsed_time)
        if differencing:
            xgboostResults = f.invert_predictions(xgboostResults)
        f.save_results(f.truth, xgboostResults, title, differencing)
        incrementBar()

    if 'Arima' in models:
        all_summaries=dict()
        train, test = train_test_split(df_, test_size=60, shuffle=False)
        arimaResults, elapsed_time = f.arima(train, test, differencing)
        times.append(elapsed_time)
        all_summaries[title+", Differencing :" + str(differencing)] = arimaResults[1]
        f.save_results(test[f.airport].values , arimaResults[0].values.round(), title, differencing)
        incrementBar()

    if len(models)>0:
        best_ = f.results.reset_index().set_index(['Model','Differencing']).MAPE.idxmin()
        if (best_[0]=='TCN' and differencing==best_[1]):
            best = tcnResults
        elif (best_[0]=='MLP' and differencing==best_[1]):
            best = mlpResults
        elif (best_[0]=='HGBoost' and differencing==best_[1]):
            best = hgboostResults
        elif (best_[0]=='BDLSTM' and differencing==best_[1]):
            best = bdlstmResults
        elif (best_[0]=='XGBoost' and differencing==best_[1]):
            best = xgboostResults
        elif (best_[0]=='AdaBoost' and differencing==best_[1]):
            best = adaboostResults
        elif (best_[0]=='Arima' and differencing==best_[1]):
            best = arimaResults

# Hide progress bar when completed
bar.empty()

# Create a dropdown menu in the first column of the streamlit dashboard
with col1:
    view = st.selectbox(
        "View:",
        options=['60 Days','90 Days','120 Days','365 Days','All Data']
    )

# Try to convert the value of the 'view' variable to an integer
# Split the string on the space character and take the first element
# If the conversion fails, set the 'view' variable to the length of the 'df' dataframe
try:
    view=int(view.split(' ')[0])
except:
    view=len(df)


# Plot predictions and residuals of the best model based on MAPE
if len(models)>0:
    # Add header for the above table
    st.markdown("<div id='headers'> Evaluation Metrics </div>""", unsafe_allow_html=True)
    
    # Show evaluation metrics table
    f.results['Time']=times
    st.dataframe(f.results.reset_index(), use_container_width=True)

    # Set the title of the plot to the name of the model that produced the best results
    st.markdown(f"<div id='headers'> Best Mape (Model: {best_[0]} , Differencing: {best_[1]}) </div>""", unsafe_allow_html=True)
    
    # Create a Plotly figure
    # Add a line plot for the 'Truth' data
    # Use the date as the x-values
    # Use the number of arrivals and the predicted number of arrivals as the y-values
    # Use 'lines' mode to connect the data points with a line
    fig1 = go.Figure()

    fig1.add_traces(go.Scatter(x=df[-view:].index, y=df[f.airport].iloc[-view:].values, name='Truth' , marker_color = '#FB6102', mode = 'lines'))

    fig1.add_traces(go.Scatter(x=df[-60:].index, y=best, name='Prediction',  mode = 'lines', marker_color = '#E6E6E6'))

    # Update the layout of the figure
    # Label the x-axis with the title 'Date' and hide the grid lines
    # Label the y-axis with the title 'Arrivals' and hide the grid lines
    fig1.update_layout(
        font=dict(size=18, family="Arial, sans-serif"),
        xaxis=dict(title=dict(text="Date", font=dict(size=18, family="Arial, sans-serif")),
        showgrid=False ),
        yaxis=dict(title=dict(text="Arrivals", font=dict(size=18, family="Arial, sans-serif")),
        showgrid=False ),
    )

    # Display the plot
    st.plotly_chart(fig1, use_container_width=True)

    # Display resid analysis
    fig2 = f.resid_analysis(best)
    st.pyplot(fig2)