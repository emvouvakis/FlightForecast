import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import gzip
import os
import shutil
from urllib.parse import urljoin
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import requests
import seaborn as sns
import xgboost as xgb
from bs4 import BeautifulSoup
from keras import losses
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, Dense,Lambda
from keras.models import Sequential
from keras.optimizers import RMSprop
from tensorflow import expand_dims
from scipy.stats import kstest
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, AdaBoostRegressor
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError, MeanSquaredError, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import datetime
import inspect
from tcn import TCN
from keras import Input
from keras import Model
from functools import cache

# @cache
class flights:

    """
    Class built to predict the number of flights (arrivals) in an airport.

    Models:
    -------
    1. Auto Arima
    2. Bidirectional LSTM
    3. MLP Regressor
    4. TCN
    4. XGBoost Regressor
    5. HistGradientBoosting Regressor
    6. AdaBoost Regressor
    """

    def __init__(self, airport:str):

        """
        Initialize and create the desired dataframes.

        Dataframes infos:
        -----------------
        - self.df1 : No exogenous features
        - self.df2 : Time as exogenous features ( Weekday(1-7), Quarter(1-4), Year )
        - self.df3 : Quarantines as exogenous feature ( Quarantine: 1 , No Quarantine: 0 )
        - self.df4 : Covid data as exogenous feature ( Selected: People_vaccinated, Source: ourworldindata.org )

        Args:
        -----
            airport (str): The selected airport - ICAO airport code

        Raises:
        -------
            Exception: Raise error if airport not in dataframe
        """
        self.origin = os.getcwd()
        dataLocation = os.path.abspath(os.path.join(os.getcwd(), os.pardir,'data'))
        os.chdir(dataLocation)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        df = pd.read_csv('data.csv')

        if airport in df.columns:
            self.airport = airport
        else:
            raise Exception(
                f"Available airports: {df.columns.difference(['day']).values} ")

        df['date'] = pd.to_datetime(df['day']).dt.date
        df.drop('day', inplace=True, axis=1)
        df = df.set_index('date').asfreq('d')
        df = df.drop(df.columns.difference([self.airport]), axis=1)
        self.df1 = df.copy()
        self.df1.name = 'No exogenous'
        self.stationary = adfuller(df[self.airport].dropna())[1] < 0.05

        # Add time as exog
        df['day'] = pd.DatetimeIndex(self.df1.index).weekday
        df['quarter'] = pd.DatetimeIndex(self.df1.index).quarter
        df['year'] = pd.DatetimeIndex(self.df1.index).year
        self.df2 = df.copy()
        self.df2.name='Time as exogenous'

        # Add quarantines as exog
        df = self.df1.copy()
        start_q1 = pd.Timestamp('2020-03-23').date()
        end_q1 = pd.Timestamp('2020-05-4').date()
        dates1 = pd.date_range(start_q1, end_q1).date

        start_q2 = pd.Timestamp('2020-11-14').date()
        end_q2 = pd.Timestamp('2020-12-14').date()
        dates2 = pd.date_range(start_q2, end_q2).date

        start_q3 = pd.Timestamp('2021-02-12').date()
        end_q3 = pd.Timestamp('2021-05-14').date()
        dates3 = pd.date_range(start_q3, end_q3).date

        quarantines = list(dates1)+list(dates2)+list(dates3)

        status = [1 if x in quarantines else 0 for x in df.index]
        df["status"] = status
        self.df3 = df.copy()
        self.df3.name='Quarantines as exogenous'

        # Add covid data as exog
        df = self.df1.copy()

        temp = pd.read_csv('covid_feature.csv')
        temp.date = pd.to_datetime(temp['date']).dt.date
        temp = temp.set_index('date').asfreq('d')

        df = df.join(temp)
        df.interpolate(inplace=True)
        df.fillna(0, inplace=True)
        self.df4 = df
        self.df4.name ='Covid data as exog'

        os.chdir(self.origin)

    def get_dfs(self) -> list:
        """
        Searches all attributes of the class to find those that are Dataframes.

        Returns:
        --------
            list: Contain all Dataframes.
        """
        # Get all attributes of the class
        attributes = inspect.getmembers(self, lambda x:not(inspect.isroutine(x)))
        # Filter the attributes that are DataFrames
        dataframe_attributes = [df[1] for df in attributes if type(df[1]) == pd.DataFrame]

        return dataframe_attributes

    @staticmethod
    def time_function(func):
        def wrapper(*args, **kwargs):
            # Start the timer
            st = datetime.datetime.now()

            # Run the function
            result = func(*args, **kwargs)

            # Calculate the elapsed time
            elapsed_time = datetime.datetime.now() - st
            elapsed_time = str(elapsed_time)[:11]

            # Return the result and the elapsed time
            return result, elapsed_time

        return wrapper

    def add_lags(self, df_:pd.DataFrame, past:int, test_size:int) -> pd.DataFrame:
        
        """
        Creates new features for each column with lags.\n
        New features are created for each column till max number of lags reached ``(t-1, t-2,... t-past)``. \n
        Splits the new features in x and y.
        Splits x and y in training and testing using ``test_size``.

        Args:
        -----
            df_ (pd.DataFrame): Selected dataframe
            past (int): Max number of lags
            test_size (int): Number of rows for testing

        Returns:
        --------
            pd.DataFrames: x_train, y_train, x_test, y_test
        """

        self.test_size = test_size 
        self.truth = self.df1[self.airport].values[-test_size:]
        self.last_train_value = self.df1[self.airport].values[-test_size-1]
        
        df = df_.copy()
        
        for col in df_.columns:
            target_map = df[col].to_dict()
            for p in range(1, past+1):
                df[col+str(p)] = (df.index - pd.Timedelta(p, unit='days')).map(target_map)

        df.dropna(inplace=True)
        
        x_cols = df.columns.difference([self.airport])
        x=df.loc[:, x_cols]

        y = df.loc[:, self.airport]

        x_train = x.iloc[:-test_size]
        y_train = y.iloc[:-test_size]
        x_test = x.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        
        return x_train, y_train, x_test, y_test

    def invert_predictions(self, results:np.array) -> np.array :

        """
        1. Insert in the first position of the predictions, the last non differenced value from the training dataset.
        2. Apply ```cumsum()``` to invert the differencing.
        3. Apply ```round()``` to make predictions integers.
        4. Apply ```abs()``` to make predictions positive.

        Args:
        -----
            results (np.array): Predictions of the differenced timeseries.

        Returns:
        --------
            np.array: Inverted predictions.
        """

        inverted = abs(np.insert(results, 0 , self.last_train_value).cumsum()[1:].round())
        return inverted

    def resid_analysis(self, predicted):
        # Calculate the residuals by subtracting the predicted values from the true values
        resid = pd.DataFrame(self.truth)-pd.DataFrame(predicted)
        print(self.truth.shape[0])
        resid.index = self.df1.index[-self.truth.shape[0]:]

        # Plot the time series of the residuals in the first subplot
        # Add a horizontal line to the first subplot indicating the mean of the residuals
        fig, axis = plt.subplots(3, figsize=(10, 8))
        axis[0].plot(resid, c='royalblue')
        axis[0].hlines(y=resid.mean(), xmin=resid.index.min(), xmax=resid.index.max(), label='Mean')

        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold', c='black')

        # Calculate the p-value of the Augmented Dickey-Fuller (ADF) test for stationarity
        # Set the title of the first subplot based on the results
        check1 = adfuller(resid)[1]
        if check1 > 0.05:
            axis[0].set_title(
                f'Stationarity Status: Not Stationary (ADF Pvalue = {round(check1, 2)})')
        elif check1 < 0.05:
            axis[0].set_title(
                f'Stationarity Status: Stationary (ADF Pvalue = {round(check1, 2)})')
        print('beeee')
        # Plot the histogram and KDE of the residuals on the second subplot

        sns.histplot(data=resid, x=resid[0], kde=True, color='darkblue', stat='frequency', ax=axis[1])
        axis[1].grid(False)

        # Add a title to the second subplot indicating the normality of the residuals
        check2 = kstest(resid, 'norm').pvalue
        if check2 > 0.05:
            axis[1].set_title(
                f'Distribution: Not Normal (Kstest Pvalue = {round(check2, 2)})')
        elif check2 < 0.05:
            axis[1].set_title(
                f'Distribution: Normal (Kstest Pvalue = {round(check2, 2)})')

        plot_acf(resid, lags=40, alpha=0.05, zero=False, ax=axis[2])

        plt.tight_layout()
        axis[0].margins(x=0)
        axis[1].margins(x=0)
        axis[2].margins(x=0)
        axis[0].legend()
        plt.show()


    # Define the results dictionary as a class attribute
    res = {
            "RMSE": [],
            "MAPE": [],
            "MAE": [],
            "Features": [],
            "Differencing": [],
            "Model": []
        }

    def save_results(self, y_test:np.array, y_pred:np.array, features:str, differencing:bool) -> pd.DataFrame:

        """
        Evaluate the predicted values using MAE, MAPE,  RMSE.

        Args:
        -----
            y_test (np.array): True values
            y_pred (np.array): Predicted values
            features (str): Selected Dataframe
            differencing (bool): Boolean value for differencing

        Returns:
        --------
            pd.DataFrame: Metrics and other informations
        """

        y_test=pd.DataFrame(y_test)
        y_pred=pd.DataFrame(y_pred.round())

        self.res['Features'].append(features)
        self.res['Differencing'].append(differencing)

        mae = mean_absolute_error(y_test, y_pred)
        self.res['MAE'].append(mae)

        temp = MeanAbsolutePercentageError(symmetric=False)
        mape = temp(y_test, y_pred)
        self.res['MAPE'].append(mape)

        temp = MeanSquaredError(square_root=True)
        rmse = temp(y_test, y_pred)
        
        self.res['RMSE'].append(rmse)
        self.res['Model'].append(self.model)
        res_ = pd.DataFrame(self.res).set_index(['Model','Features'])
        self.results = res_


    @time_function
    def arima(self, train:pd.DataFrame, test:pd.DataFrame, status:bool):

        """
        Model:
        ------
            Auto Arima

        Args:
        -----
            status (bool): Boolean for differencing
        """

        self.model='Arima'

        if len(train.columns.difference([self.airport])) > 0:
            exog1 = train[train.columns.difference([self.airport])]
            exog2 = test[test.columns.difference([self.airport])]
        else:
            exog1 = None
            exog2 = None

        if status:
            s = False
            dif = 1
        else:
            s = True
            dif = 0

        model = pm.auto_arima(train[self.airport], m=7, seasonal=True, stationary=s, d=dif, max_p=7, max_q=7, max_d=1,
            max_Q=7, max_P=7, max_D=1, maxiter=100, alpha=0.05, max_order=None, X=exog1, error_action="ignore")

        b = model.summary()

        n_periods = len(test)
        arimaResults, confint = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.5,
                                              index=test.index, X=exog2)

        return arimaResults, b

    @time_function
    def mlp(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame, layers: tuple) -> np.array:

        """
        Model:
        ------
            MLP Regressor

        Args:
        -----
            layers (tuple): Tuple for the number of neurons in each layer

        Returns:
        --------
            np.array: Predicted values.
        """
        #Select activation function. If data are differenced set tanh.
        func = 'relu' if y_train.lt(0).sum()==0 else 'tanh'
        self.model='MLP'
        x_train, y_train, x_test = x_train.values, y_train.values.ravel(), x_test.values

        model = MLPRegressor(hidden_layer_sizes=layers, activation=func, alpha=0.001,
                           solver='adam', learning_rate='constant', learning_rate_init=0.05,
                           max_iter=1000, random_state=5, tol=0.001, early_stopping=True, verbose=False,
                           validation_fraction=0.3).fit(x_train, y_train)

        mlpResults = model.predict(x_test)
        return mlpResults
        
    @time_function
    def bdlstm(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame, layers: tuple) -> np.array:

        """
        Model:
        ------ 
            Bidirectional LSTM

        Args:
        -----
            layers (tuple): Tuple for the number of neurons in each layer.

        Raises:
        -------
            ValueError: If the tuple for the neurons doesnt contain two items.

        Returns:
        --------
            np.array: Predicted values.
        """
        
        if len(layers) != 2:
            raise ValueError('The tuple for layers must contain exactly two items')

        self.model='BDLSTM'
        model = Sequential([
            Lambda(lambda x: expand_dims(x, axis=-1), input_shape=[None]),
            Bidirectional(
            layer=LSTM(layers[0], activation='tanh', return_sequences=False),
            backward_layer=LSTM(layers[1], return_sequences=False, activation='relu', use_bias=False, go_backwards=True)),
            Dense(1, activation='linear')
        ])
        opt = RMSprop(learning_rate=0.05, momentum=0.001)
        model.compile(loss=losses.Huber(), optimizer=opt)

        early_stopping = EarlyStopping( monitor='loss', patience=5, min_delta=0.00001, mode='auto')
        model.fit(x_train, y_train, epochs=1000, validation_split=0.3, verbose=0, callbacks=[early_stopping])

        lstmResults = model.predict(x_test, verbose=0)
        lstmResults = lstmResults.ravel()
        return lstmResults

    @time_function
    def tcn(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame, filters: tuple) -> np.array:
        """
        Model:
        ------ 
            Temporal Convolutional Network

        Raises:
        -------
            ValueError: If the tuple for the filters doesnt contain two items.

        Returns:
        --------
            np.array: Predicted values.
        """
        if len(filters) != 2:
            raise ValueError('The tuple for layers must contain exactly two items')

        self.model='TCN'
        i = Input(shape=(x_train.shape[1], 1))
        m = TCN(nb_filters=filters[0], return_sequences=True, activation='tanh')(i)
        m = TCN(nb_filters=filters[1], return_sequences=False, activation='relu')(m)
        m = Dense(1, activation='linear')(m)

        model = Model(inputs=[i], outputs=[m])
        model.compile(optimizer='adam', loss=losses.Huber())
        
        early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00001, mode='auto')
        model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[early_stopping])

        tcnResults = model.predict(x_test, verbose=0)
        tcnResults = tcnResults.ravel()
        return tcnResults

    @time_function
    def hgboost(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame) -> np.array:

        """
        Model: HistGradientBoosting Regressor

        Returns:
            np.array: Predictions
        """

        self.model='HGBoost'
        model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.05, early_stopping="auto")
        model.fit(x_train, y_train.values.ravel())
        hgboostResults = model.predict(x_test)

        return hgboostResults

    @time_function
    def adaboost(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame) -> np.array:

        """
        Model: AdaBoost Regressor

        Returns:
            np.array: Predictions
        """

        self.model='AdaBoost'
        model = AdaBoostRegressor(loss='linear', learning_rate=0.001)
        model.fit(x_train, y_train.values.ravel())
        adaboostResults = model.predict(x_test)

        return adaboostResults
    
    @time_function
    def xgboost(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame) -> np.array:

        """
        Model:
        ------ 
            XGBoost Regressor

        Returns:
        --------
            np.array: Predicted values.
        """

        self.model='XGBoost'
        model = xgb.XGBRegressor(booster='gbtree', learning_rate= 0.35, n_estimators= 50)
        model.fit(x_train, y_train)
        xgboostResults = model.predict(x_test)

        return xgboostResults


class acquire_data:
    def  __init__(self, path:str):

        """
        Create necessary folders if they dont exist.

        Args:
            path (str): Selected folder location.

        Raises:
            Exception: Path must be an existing directory.

        Example:
        >>> from utils import acquire_data
        >>> d = acquire_data(path=r'E:\data')
        >>> d.download_flights(url='https://zenodo.org/record/7323875#.Y6GCi3ZBzIU')
        >>> d.parse(airports=["LGAV","LGRP"])
        >>> d.download_covid_feature()
        >>> d.covid_feature("LGAV")
        """

        if not os.path.exists(path) or not os.path.isdir(path):
            raise Exception("Path must be an existing directory.")

        self.home = os.getcwd()
        self.origin = path
        self.ziped = os.path.join(self.origin, r'ziped')
        if not os.path.exists(self.ziped):
            os.mkdir(self.ziped)

        unziped = os.path.join(self.origin, r'unziped')
        if not os.path.exists(unziped):
            os.mkdir(unziped)
        self.unziped = unziped

    def download_flights(self, url:str):

        """
        - Download all necessary ziped files.
        - Unzip downloaded files.

        Args:
            url (str): OpenSky Network 2020 - zenodo
                Source: https://zenodo.org/record/7323875#.Y6QbBHZBzIV
        """

        self.url = url

        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, "html.parser")

        file_counter = 0
        for link in soup.select("a[href$='gz?download=1']"):

            # Naming the flightlist files using the last portion of each link
            filename = os.path.join(self.ziped, link['href']).split('/')[-1].split("?")[0]

            # Download only new files
            if filename not in os.listdir(self.ziped):
                file_counter += 1

                # Download
                os.chdir(self.ziped)
                with open(filename, 'wb') as f:
                    f.write(requests.get(urljoin(self.url, link['href'])).content)
                # Unzip
                with gzip.open(filename, 'rb') as f_in:

                    os.chdir(self.unziped)

                    with open(filename[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        print('Downloaded', file_counter,' files.')
        os.chdir(self.home)

    def download_covid_feature(self):

        """
        Download Covid data to add as exogenous feature.

            Source: https://ourworldindata.org/coronavirus/country/greece
        """

        os.chdir(self.origin)
        url='https://github.com/owid/covid-19-data/raw/master/public/data/owid-covid-data.csv'
        res = requests.get(url, allow_redirects=True)
        filename = 'owid-covid-data.csv'
        if filename not in os.listdir():
            with open(filename,'wb') as file:
                file.write(res.content)
        os.chdir(self.home)

    def covid_feature(self,airport:str):

        """
        - Find best feature based on correlation.
        - Export selected feature as csv.

        Args:
            airport (str): Selected airport.
        """

        os.chdir(self.origin)
        gr=pd.read_csv('owid-covid-data.csv')
        gr.date=pd.to_datetime(gr.date).dt.date
        gr.set_index('location',inplace=True)
        gr=gr.loc['Greece'].set_index('date')

        df = pd.read_csv('data.csv',usecols=[airport,'day'])
        df.day=pd.to_datetime(df.day).dt.date
        df.set_index('day',inplace=True)
        df.dropna(axis=0, inplace=True)
        temp = df.join(gr)
        temp.index = temp.index.rename('date')

        feature = temp.corr()[airport].nlargest(5).index[1]

        print(f"Selected Feature:{feature}")
        pd.DataFrame(temp[feature]).to_csv('covid_feature.csv')
        os.chdir(self.home)


    def parse(self, airports=["LGAV", "LGRP"]) -> pd.DataFrame:

        """
        - Concatenate all unziped csv files.
        - Add days that do not exist in the index.
        - Inerpolate missing values.
        - Treat sudden drops in values. 
        - Export data to csv.
        """
        
        os.chdir(self.unziped)

        flightlist = pd.concat(
            pd.read_csv(file, usecols = ['callsign','day','destination'], low_memory = True)
            for file in os.listdir()
        )

        data = pd.concat(
            (
                flightlist.query(f'destination == "{airport}"')
                .groupby("day")
                .agg(dict(callsign="count"))
                .rename(columns=dict(callsign=airport))
                for airport in airports
            ),
            axis=1,
        )

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        start_date = pd.to_datetime(data.index.values[0])
        end_date = pd.to_datetime(data.index.values[-1])
        dates=pd.date_range(start_date,end_date).tz_localize('UTC')

        missingDates1=[x for x in dates if x not in data.index]
        missingDates2=[np.nan for x in missingDates1]
        missingDates=pd.DataFrame()
        missingDates['day']=missingDates1

        for port in airports:
            missingDates[port]=missingDates2

        missingDates.set_index('day',inplace=True)
        data = data.append(missingDates).sort_index()
        os.chdir(self.origin)
        temp = data.copy()
        temp.to_csv('raw.csv')
        data = data.interpolate().round()

        for port in airports:
            outliers=[]
            for i in range(0,100):
                for  index, (v0, v1)  in enumerate(zip(data[port].values, data[port].values[1:])):

                    if abs((v1 - v0)/v0) >= 0.8:
                        value = min(v0,v1)
                        if value == v0:
                            outliers.append(data.index.values[index])
                        else:
                            outliers.append(data.index.values[index+1])
                outliers = list(set(outliers))

                if len(outliers) == 0:
                    break

                temp = pd.to_datetime(outliers).tz_localize('UTC')
                data.loc[temp, port]=np.nan
                outliers.clear()
                data[port] = data[port].interpolate().round()

        data.to_csv('data.csv')
        os.chdir(self.home)