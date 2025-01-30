import streamlit as st
st.set_page_config(page_title = "Solar Energy Forecasting Tool", page_icon = ':sun_small_cloud:', layout="wide")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from streamlit_shap import st_shap
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from streamlit_extras.chart_container import chart_container
from streamlit_extras.altex import line_chart
from streamlit_extras.colored_header import colored_header
import pandas as pd
from pandas import read_csv
from datetime import  date, timedelta
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.compose import make_column_transformer 
from sklearn.model_selection import train_test_split
import requests
import numpy as np
from sklearn import metrics
import folium
from streamlit_folium import st_folium, folium_static
#from sklearn.model_selection import GridSearchCV
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import product
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import shap
import joblib
import warnings
from urllib.request import urlopen
import importlib.util
from streamlit_extras.metric_cards import style_metric_cards

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

st.title('SOLAR ENERGY FORECASTING :sun_small_cloud: NOVAERA')


#import main functions    
#@st.cache_resource
def import_functions():
        # Path to the local file
        local_file_path = "./Main.py"  # Update this to the actual path of your file

        # Read the contents of the local file
        with open(local_file_path, "r") as file:
                script_content = file.read()

        # Create a module object
        spec = importlib.util.spec_from_loader("main", loader=None)

        # Create a module from the spec
        module = importlib.util.module_from_spec(spec)

        # Execute the module
        exec(script_content, module.__dict__)

        # Return the module so its functions can be used
        return module

module = import_functions()

config = {
        'file_path': "./Full_SolarDataset.csv",
        'target_variable': 'Active_Power',
        'predictors': ['temperature_2m', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',  'windspeed_10m', 'cloudcover', 'season'],
        'categorical_variables': ['season'],
        'time_intervals': ['first_interval','second_interval','third_interval','fourth_interval','fifth_interval','sixth_interval'],
        'weather_types': ['TypeA', 'TypeB', 'TypeC'],
        'standardize_predictor_list': ['temperature_2m', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',  'windspeed_10m', 'cloudcover']
}
#units of measure
variable_units = {'Active_Power' : ' kwh',
                'temperature_2m' : ' °C',
                'relativehumidity_2m' : ' %',
                'direct_radiation' : ' W/m2',
                'diffuse_radiation' : ' W/m2',
                'windspeed_10m' : ' km/h',
                'cloudcover' : ' %'
                }
#location of the panel
lat = -23.760363
long = 133.874719

@st.cache_data  # 👈 Add the caching decorator
def load_data(path, sep):
        df = pd.read_csv(path, sep=sep ,parse_dates=['timestamp'], index_col='timestamp')
        return df

path = './Full_SolarDataset.csv'

#read dataset
df = load_data(path, '\t')

#subpages 
selected_subpage = option_menu(None, ['Home', 'EDA', 'ML Model Estimation', 'Forecast'], 
        icons=['house', 'bar-chart', "activity", 'graph-up-arrow'], 
        menu_icon="cast", default_index=0, orientation="horizontal")





# ------------------------------------------------------ HOME ---------------------------------------------------------

if selected_subpage == "Home":
        colored_header(
                label="PV Panel Location Explorer",
                description="",
                color_name="blue-80"
                )
        st.write(
        """
        Our home project utilizes data from **Site 18 at the DKA Solar Centre** in Alice Springs, Australia—an area renowned for its exceptional solar potential and advanced research infrastructure.  
        
        This site provides **high-quality, real-world data**, which is essential for developing accurate solar energy forecasting models. Australia's leadership in **renewable energy** and strong **policy support** further enhance the project's relevance.  

        Additionally, the site's accessibility facilitates **seamless collaboration** with stakeholders and instructors, ensuring practical feasibility. By leveraging this strategic location, our project contributes to **optimizing solar energy efficiency** and supporting the **global transition** to sustainable energy solutions.
        """
        )

        
        col1, col2= st.columns(2)
        #image
        with col1:
                #Correct file path
                file_path = "./Site_picture.jpg"  
                try:
                        # Display the image in Streamlit
                        st.image(file_path, caption="PV Panel located in DKA Solar Centre. Alice Springs, Australia", use_container_width=True)
                except FileNotFoundError:
                        st.error(f"File not found: {file_path}")
                except Exception as e:
                        st.error(f"An error occurred: {e}")

        #map
        with col2:
                m = folium.Map(location=[lat, long], 
                        zoom_start=11, control_scale=True)
                html = f'''{'Panel Manufacturer: ' + 'SunPower'}<br>
                        {'Installation Date: ' + "2011"}<br>
                        {'PV Technology: ' + 'Monocrystalline Silicon'}<br>
                        {'Array Rating: ' + "5.24kW"}'''
                #Setup the content of the popup

                iframe = folium.IFrame(html,
                        width=400,
                        height=100)
                #Initialise the popup using the iframe
                popup = folium.Popup(iframe, min_width=500, max_width=800)
                
                #Add popup to the map
                folium.Marker(location=[lat, long],
                        popup = popup, c='SunPower').add_to(m)

                st_data = st_folium(m, height = 546, width=1100)
        st.info(" :information_source:     more panel info can be displayed by clicking on the panel marker")

        colored_header(
                label="PV Panel Dataset Explorer",
                description="",
                color_name="blue-80"
                )
        #Select Date range
        min_date = df.index.min()
        max_date = df.index.max()

        col1, col2= st.columns(2)

        with col1:
                st.write("##### Select a date range")
                date_input = st.date_input(label = "Select date range", value = ((max_date- timedelta(days=31)),max_date),min_value = min_date, max_value = max_date, label_visibility = 'collapsed')
        with col2:
                st.write("##### Select a variable")
                variable_name = st.selectbox('Select variable', df.columns, label_visibility = 'collapsed')   

        #mask to subset dataset based on input date range
        mask = (df.index > pd.to_datetime(date_input[0])) & (df.index <= pd.to_datetime(date_input[1]))

        df_home = df.loc[mask]
        #reset index for the chart
        df_home = df_home.reset_index()


        with chart_container(df):
                st.write("## Here's a cool chart! You can see the data and export it too!")
                line_chart(
                        data=df_home,
                        x='timestamp',
                        y=variable_name,
                        title="Historical Energy Generation Data",
                )

        st.metric(label=f"Average {variable_units[variable_name]} output for {variable_name} in the selected period", value=str(np.mean(df_home[variable_name]).round(3)) + variable_units[variable_name])


        st.info("###### Data Credits: DKA Solar Centre")

# -------------------------------------------------- Exploratory Data Analysis ------------------------------------------------------


elif selected_subpage == 'EDA':
        colored_header(
        label="Exploratory Data Analysis - Get to know the data",
        description="",
        color_name="blue-80"
        )

        #user selects granularity dataset and day/full dataset
        col1, col2 = st.columns(2)
        with col1:
                chosen_granularity = st.selectbox("Choose data granularity", ['Hourly','Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'])
        with col2:
                daytime_or_full_dataset = st.selectbox("Choose Daytime Dataset or Full Dataset", ['Daytime', 'Full'])
        
        #infos
        if chosen_granularity != 'Hourly':
                st.info("Granular dataset is formed from the mean of hourly observations")
        if daytime_or_full_dataset == 'Daytime':
                st.info("Daytime dataset consists of observations from 7 AM to 18 PM")

        
        #prepare day dataset
        mask = (df.index.hour >= 7) & (df.index.hour <= 18)
        daytime_dataset = df[mask]


        if daytime_or_full_dataset == "Daytime":
                chosen_dataset = daytime_dataset.copy()        
        else:
                chosen_dataset = df.copy()


        #prepare granularity datasets
        df_daily = chosen_dataset.resample('D', kind = 'timestamp').mean() 
        df_weekly = chosen_dataset.resample('W', kind = 'timestamp').mean()
        df_monthly = chosen_dataset.resample('M', kind = 'timestamp').mean() 
        df_quarterly = chosen_dataset.resample('Q', kind = 'timestamp').mean() 
        df_yearly = chosen_dataset.resample('Y', kind = 'timestamp').mean()  

        granularities = {'Hourly': chosen_dataset,
                        'Daily': df_daily,
                        'Weekly': df_weekly, 
                        'Monthly': df_monthly, 
                        'Quarterly': df_quarterly, 
                        'Yearly': df_yearly,  
                        }

        granular_df = granularities[chosen_granularity]

        st.write(f"###### Number of observations in {chosen_granularity} dataset: {len(granular_df)}")


        @st.cache_data
        def corr_plot(df):
                fig = plt.figure(figsize = (18,17), clear = True)
                corr = df.corr()
                sns.heatmap(corr, annot=True, cmap='viridis', fmt='.3f')
                st.pyplot(fig)


        @st.cache_data
        def pairplot(df):
                fig = plt.figure(figsize = (18,13), clear = True)
                fig = sns.pairplot(df, diag_kind='kde', corner = True)
                st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
                st.write(f"#### Correlation Plot of the {chosen_granularity} dataframe")
                corr_plot(granular_df)
        with col2:
                st.write(f"#### Pairplot of the {chosen_granularity} dataframe")
                pairplot(granular_df)



        ## add a season and month features

        seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 
                6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
        
        @st.cache_data
        def map_season_month(df):   
        # map season
                df['season'] = df.index.month.map(seasons)
                # map month
                df['month'] = df.index.month_name()
                return df
        
        granular_df = map_season_month(granular_df)


        chosen_variable_EDA = st.selectbox("Choose variable", granular_df.columns)



        # sunburst plot
        @st.cache_data
        def sunburst_plot(df, var):
                fig = px.sunburst(df, path=['season', 'month'], values=var, color = 'season', color_discrete_map={'Summer':'#F8F40E', 'Spring':'#00FF00', 'Winter': '#89CFF0', 'Fall': '#FF4500'})
                fig.update_traces(textfont=dict(family="Arial Black", size=10.5))
                st.plotly_chart(fig, theme="streamlit",use_container_width=True)
        
        # violin plot
        @st.cache_data
        def violin_plot(df, granularity, var):
                if granularity == 'Month':
                        fig = px.violin(granular_df, y=var, x='month', box=True, 
                                hover_data=df.columns, category_orders = {'month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']})
                        st.plotly_chart(fig, theme="streamlit",use_container_width=True)
                else:
                        fig = px.violin(granular_df, y=var, x='season', box=True, 
                                hover_data=df.columns, category_orders = {'season': ['Spring', 'Summer', 'Fall', 'Winter']})
                        st.plotly_chart(fig, theme="streamlit",use_container_width=True)

        @st.cache_data
        def kde_plot(df, granularity, var):
                fig, ax = plt.subplots()
                sns.set(style='whitegrid',palette="deep", font_scale=1.1, rc={"figure.figsize": [22, 10]})
                if granularity == 'season':
                        sns.set(style='whitegrid',palette="deep", font_scale=1.1, rc={"figure.figsize": [22, 10]})
                        colors = {"Winter": "darkblue", "Spring": "green", "Summer": "orange", "Fall": "brown"}
                        for season, color in colors.items():
                                sns.kdeplot(df[df["season"]==season][var], 
                                                fill=True, 
                                                color=color, 
                                                label=season, 
                                                ax=ax,
                                                linewidth=1.2)
                        ax.legend()
                        ax.set_xlabel(var, labelpad=15)
                        ax.set_ylabel("Density", labelpad=15)
                        ax.set_title('Kernel Density Estimation of Active Power by Season', pad=15, fontsize = 20, fontweight = 'bold')
                        st.pyplot(fig)
                else:
                        # Define a color palette with distinct colors for each month
                        colors = {"January": "darkblue", "February": "darkcyan", "March": "darkgreen", 
                                "April": "green", "May": "lime", "June": "gold",
                                "July": "orange", "August": "red", "September": "purple",
                                "October": "pink", "November": "skyblue", "December": "navy"}
                        sns.set(style='whitegrid',palette="deep", font_scale=1.1, rc={"figure.figsize": [22, 10]})
                        for month, color in colors.items():
                                sns.kdeplot(df[df["month"]==month][var], 
                                                fill=True, 
                                                color=color, 
                                                label=month, 
                                                ax=ax,
                                                linewidth=1.2)

                        ax.legend()
                        ax.set_xlabel(var, labelpad=15)
                        ax.set_ylabel("Density", labelpad=15)
                        ax.set_title('Kernel Density Estimation of Active Power by Month', pad=15, fontsize = 20, fontweight = 'bold')

                        st.pyplot(fig)

        @st.cache_data
        def histplot(df, var):
                fig, ax = plt.subplots(figsize = (20,8), clear = True)
                sns.histplot(data=df, x=var, kde=True,  linewidth=2)
                ax.set_title(f'Distribution of {var.replace("_", " ").title()}', fontsize=14)
                ax.set_xlabel(var.replace("_", " ").title(), fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.grid(color='gray', linestyle='--', linewidth=0.5)
                st.pyplot(fig)


        col1, col2 = st.columns(2)
        with col1:
                st.write(f"#### Sunburst Plot of the {chosen_granularity} dataframe")
                sunburst_plot(granular_df, chosen_variable_EDA)
        with col2:
                st.write("#### Histplot of the dataframe")
                histplot(granular_df, chosen_variable_EDA)
        


        col1, col2 = st.columns(2)
        with col1:
                st.write(f"#### KDE of the {chosen_granularity} dataframe - By season or month")
                tab1, tab2 = st.tabs(['Monthly', 'Seasonally'])
                with tab1:
                        kde_plot(granular_df, 'month', chosen_variable_EDA)
                with tab2:
                        kde_plot(granular_df, 'season', chosen_variable_EDA)


        with col2:
                st.write(f"#### Violin Plot of the {chosen_granularity} dataframe - By season or month")
                tab1, tab2 = st.tabs(['Monthly', 'Seasonally'])
                with tab1:
                        violin_plot(granular_df, 'Month', chosen_variable_EDA)
                with tab2:
                        violin_plot(granular_df, 'Season', chosen_variable_EDA)

#brush it up

# -------------------------------------------------- ML Model Estimation ------------------------------------------------------


elif selected_subpage == 'ML Model Estimation':
        
        @st.cache_data
        def main_functions():
                df = module.load_data('./Full_SolarDataset.csv')
                df = module.add_season(df)
                df = module.choose_interval(df)
                train, test = module.split_data(df)
                train = module.detect_time_interval(train)
                test = module.detect_time_interval(test)
                new_train_data = module.create_weather_type(train)
                new_test_data = module.classify_weather_type(new_train_data, test)
                new_stand_train, new_stand_test = module.standardize_data(new_train_data, new_test_data)
                return test, new_test_data, new_stand_train, new_stand_test

        test, new_test_data, new_stand_train, new_stand_test = main_functions()

        # ------------- RF -----------

        colored_header(
                        label="Random Forest Performance",
                        description="",
                        color_name="blue-80"
                        )
        with st.spinner("Running RF model..."):
                forecasted_data_RF = module.train_predict_RF_model(new_stand_train, new_stand_test)
        mae,rmse,r2,smape_ = module.evaluate_model(forecasted_data_RF, test, new_test_data)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(":blue[**MAE**]",round(mae,3))
        col2.metric(":blue[**RMSE**]", round(rmse,3))
        col3.metric(":blue[**R_Squared**]", str(round(r2,2) * 100) + '%')
        col4.metric(":blue[**Scaled mean absolute percentage error**]", str(round(smape_,3)) + '%')
        min_date = new_test_data.index.min()
        max_date = new_test_data.index.max()
        st.write("### Select a date range for Deviations plot")
        date_input = st.date_input(label = "s", value = ((max_date- timedelta(days=90)),max_date),min_value = min_date, max_value = max_date, label_visibility="collapsed", key = 3)

        @st.cache_data
        def deviations_plot(df):
                mask = (df.index > pd.to_datetime(date_input[0])) & (df.index <= pd.to_datetime(date_input[1]))
                deviations = df.loc[mask]
                deviations = deviations[['PredictedTotalPower', 'ActualTotalPower', 'season']].copy()
                deviations = deviations.reset_index()
                fig = px.line(deviations, x="date", y=["ActualTotalPower", "PredictedTotalPower"], title = 'RF Energy Deviations - Predicted Power vs. Actual Power')
                fig['data'][0]['line']['color']='rgb(0, 191, 255)'  #actual total power
                fig['data'][0]['line']['width']=5  
                fig['data'][1]['line']['color']='rgb(8, 35, 161)'  #predicted total power
                fig['data'][1]['line']['width']=5
                # Update layout for title
                fig.update_layout(
                        title={
                        'text': 'RF Energy Deviations - Predicted Power vs. Actual Power',
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                        title_font_size=25,
                        xaxis_title=""
                )
                st.plotly_chart(fig, theme="streamlit",use_container_width=True)

        deviations_plot(forecasted_data_RF)

        #### shap values

        # Define the path for the RF SHAP file
        RF_shap_file = './SHAP_Values/RF_Shap.csv'

        # Check if the file exists
        if os.path.exists(RF_shap_file):
                # Read CSV correctly
                RF_shap_df = pd.read_csv(RF_shap_file, sep=',')  

                # Convert to numeric
                RF_shap_df = RF_shap_df.apply(pd.to_numeric, errors='coerce')

                # Ensure feature alignment
                feature_data = new_test_data.iloc[:, :-2] 

                # Truncate rows to match
                min_rows = min(RF_shap_df.shape[0], feature_data.shape[0])
                RF_shap_df = RF_shap_df.iloc[:min_rows, :]
                feature_data = feature_data.iloc[:min_rows, :]

                # Ensure column alignment
                feature_data = feature_data[RF_shap_df.columns]

                # Ensure correct shape
                assert RF_shap_df.shape == feature_data.shape, \
                        f"Mismatch: SHAP shape {RF_shap_df.shape}, Feature shape {feature_data.shape}"

                # Convert SHAP values to numpy
                shap_values = RF_shap_df.values  
                
                plt.figure(figsize=(10, 5))  # Adjust width & height as needed

                # SHAP visualization
                col1, col2 = st.columns(2)
                with col1:
                        st.write('#### Feature value importance scale')
                        st_shap(shap.summary_plot(shap_values, feature_data, plot_type='violin', color='coolwarm'))
                with col2:
                        st.write('#### Absolute feature importance')
                        st_shap(shap.summary_plot(shap_values, feature_data, plot_type='bar'))
        else:
                st.write("#### RF SHAP values file not found. Please ensure the SHAP values are computed and saved to './SHAP_Values/RF_Shap.csv'.")

        # ------------- MLP -----------

        colored_header(
                        label="Multilayer Perceptron (MLP) Performance",
                        description="",
                        color_name="blue-80"
                        )
        with st.spinner("Running MLP model..."):
                forecasted_data_MLP = module.train_predict_MLP_model(new_stand_train, new_stand_test)
        mae,rmse,r2,smape_ = module.evaluate_model(forecasted_data_MLP, test, new_test_data)
        
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(":blue[**MAE**]",round(mae,3))
        col2.metric(":blue[**RMSE**]", round(rmse,3))
        col3.metric(":blue[**R_Squared**]", str(round(r2,3) * 100) + '%')
        col4.metric(":blue[**Scaled mean absolute percentage error**]", str(round(smape_,3)) + '%')
        style_metric_cards(background_color="#BBDEFB")
        min_date = new_test_data.index.min()
        max_date = new_test_data.index.max()
        st.write("### Select a date range for Deviations plot")
        date_input = st.date_input(label = "s", value = ((max_date- timedelta(days=90)),max_date),min_value = min_date, max_value = max_date, label_visibility="collapsed", key = 1)

        @st.cache_data
        def deviations_plot(df):
                mask = (df.index > pd.to_datetime(date_input[0])) & (df.index <= pd.to_datetime(date_input[1]))
                deviations = df.loc[mask]
                deviations = deviations[['PredictedTotalPower', 'ActualTotalPower', 'season']].copy()
                deviations = deviations.reset_index()
                fig = px.line(deviations, x="date", y=["ActualTotalPower", "PredictedTotalPower"], title = 'MLP Energy Deviations - Predicted Power vs. Actual Power')
                fig['data'][0]['line']['color']='rgb(0, 191, 255)'  #actual total power
                fig['data'][0]['line']['width']=5  
                fig['data'][1]['line']['color']='rgb(8, 35, 161)'  #predicted total power
                fig['data'][1]['line']['width']=5
                # Update layout for title
                fig.update_layout(
                        title={
                        'text': 'MLP Energy Deviations - Predicted Power vs. Actual Power',
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                        title_font_size=25,
                        xaxis_title=""
                )
                st.plotly_chart(fig, theme="streamlit",use_container_width=True)

        deviations_plot(forecasted_data_MLP)
        

        #### shap values

        # Define the path for the RF SHAP file
        MLP_shap_file = './SHAP_Values/MLP_Shap.csv'

        # Check if the file exists
        if os.path.exists(MLP_shap_file):
                # Read CSV correctly
                MLP_shap_df = pd.read_csv(MLP_shap_file, sep='\t')

                # Convert to numeric
                MLP_shap_df = MLP_shap_df.apply(pd.to_numeric, errors='coerce')

                # Ensure feature alignment
                feature_data = new_test_data.iloc[:, :-2]  

                # Truncate rows to match
                min_rows = min(MLP_shap_df.shape[0], feature_data.shape[0])
                MLP_shap_df = MLP_shap_df.iloc[:min_rows, :]
                feature_data = feature_data.iloc[:min_rows, :]

                # Ensure column alignment
                feature_data = feature_data[MLP_shap_df.columns]

                # Ensure correct shape
                assert MLP_shap_df.shape == feature_data.shape, \
                        f"Mismatch: SHAP shape {MLP_shap_df.shape}, Feature shape {feature_data.shape}"

                # Convert SHAP values to numpy
                shap_values = MLP_shap_df.values  
                
                plt.figure(figsize=(10, 5))  # Adjust width & height as needed

                # SHAP visualization
                col1, col2 = st.columns(2)
                with col1:
                        st.write('#### Feature value importance scale')
                        st_shap(shap.summary_plot(shap_values, feature_data, plot_type='violin', color='coolwarm'))
                with col2:
                        st.write('#### Absolute feature importance')
                        st_shap(shap.summary_plot(shap_values, feature_data, plot_type='bar'))
        else:
                st.write("#### MLP SHAP values file not found. Please ensure the SHAP values are computed and saved to './SHAP_Values/MLP_Shap.csv'.")



        # ------------- XGB -----------


        colored_header(
                        label="XGBoost Performance",
                        description="",
                        color_name="blue-80"
                        )
        with st.spinner("Running XGB model..."):
                forecasted_data_XGB = module.train_predict_XGB_model(new_stand_train, new_stand_test)
        mae,rmse,r2,smape_ = module.evaluate_model(forecasted_data_XGB, test, new_test_data)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(":blue[**MAE**]",round(mae,3))
        col2.metric(":blue[**RMSE**]", round(rmse,3))
        col3.metric(":blue[**R_Squared**]", str(round(r2,3) * 100) + '%')
        col4.metric(":blue[**Scaled mean absolute percentage error**]", str(round(smape_,3)) + '%')
        min_date = new_test_data.index.min()
        max_date = new_test_data.index.max()
        st.write("### Select a date range for Deviations plot")
        date_input = st.date_input(label = "s", value = ((max_date- timedelta(days=90)),max_date),min_value = min_date, max_value = max_date, label_visibility="collapsed", key = 2)

        @st.cache_data
        def deviations_plot(df):
                mask = (df.index > pd.to_datetime(date_input[0])) & (df.index <= pd.to_datetime(date_input[1]))
                deviations = df.loc[mask]
                deviations = deviations[['PredictedTotalPower', 'ActualTotalPower', 'season']].copy()
                deviations = deviations.reset_index()
                fig = px.line(deviations, x="date", y=["ActualTotalPower", "PredictedTotalPower"], title = 'XGB Energy Deviations - Predicted Power vs. Actual Power')
                fig['data'][0]['line']['color']='rgb(0, 191, 255)'  #actual total power
                fig['data'][0]['line']['width']=5  
                fig['data'][1]['line']['color']='rgb(8, 35, 161)'  #predicted total power
                fig['data'][1]['line']['width']=5
                # Update layout for title
                fig.update_layout(
                        title={
                        'text': 'XGB Energy Deviations - Predicted Power vs. Actual Power',
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                        title_font_size=25,
                        xaxis_title=""
                )
                st.plotly_chart(fig, theme="streamlit",use_container_width=True)

        deviations_plot(forecasted_data_XGB)

        #### shap values

        # Define the path for the RF SHAP file
        XGB_shap_file = './SHAP_Values/XGB_Shap.csv'

        # Check if the file exists
        if os.path.exists(MLP_shap_file):
                # Read CSV correctly
                XGB_shap_df = pd.read_csv(XGB_shap_file, sep=',')

                # Convert to numeric
                XGB_shap_df = XGB_shap_df.apply(pd.to_numeric, errors='coerce')

                # Ensure feature alignment
                feature_data = new_test_data.iloc[:, :-2]  

                # Truncate rows to match
                min_rows = min(XGB_shap_df.shape[0], feature_data.shape[0])
                XGB_shap_df = XGB_shap_df.iloc[:min_rows, :]
                feature_data = feature_data.iloc[:min_rows, :]

                # Ensure column alignment
                feature_data = feature_data[XGB_shap_df.columns]

                # Ensure correct shape
                assert XGB_shap_df.shape == feature_data.shape, \
                        f"Mismatch: SHAP shape {XGB_shap_df.shape}, Feature shape {feature_data.shape}"

                # Convert SHAP values to numpy
                shap_values = XGB_shap_df.values  
                
                plt.figure(figsize=(10, 5))  # Adjust width & height as needed

                # SHAP visualization
                col1, col2 = st.columns(2)
                with col1:
                        st.write('#### Feature value importance scale')
                        st_shap(shap.summary_plot(shap_values, feature_data, plot_type='violin', color='coolwarm'))
                with col2:
                        st.write('#### Absolute feature importance')
                        st_shap(shap.summary_plot(shap_values, feature_data, plot_type='bar'))
        else:
                st.write("#### XGB SHAP values file not found. Please ensure the SHAP values are computed and saved to './SHAP_Values/XGB_Shap.csv'.")


        
# -------------------------------------------------- Forecast ------------------------------------------------------


elif selected_subpage == 'Forecast':
        @st.cache_data
        def get_weather_forecast_data():
                lat = -23.760363
                long = 133.874719

                Predictors = ['temperature_2m', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',  'windspeed_10m', 'cloudcover']
                start_date = str(date.today()+ timedelta(days=1))
                end_date = str(date.today()+ timedelta(days=4))

                r = requests.get('https://api.open-meteo.com/v1/forecast', params={'latitude':lat, 'longitude': long, 'timezone': 'auto', 'start_date':start_date , 'end_date': end_date , 'hourly' : Predictors}).json() #timezone = auto so that it matches the local timezone

                weather_df = pd.DataFrame(columns = Predictors )
                time = pd.to_datetime(np.array(r['hourly']['time']))
                weather_df['date'] = time
                for p in Predictors:
                        weather_df[p] = np.array(r['hourly'][p])
                weather_df['date'] = pd.to_datetime(weather_df['date'])
                return weather_df
        
        @st.cache_data
        def classify_weather_forecast_type(df):
                new_df = pd.DataFrame()
                for interval in config['time_intervals']:
                        interval_dataset = df[df['time_interval'] == interval].copy()
                        try:
                                grid = joblib.load(f'./ClassifiedWeatherTypes/RF_Weather_{interval}_.pkl')
                                classified_weather_type = module.predict_weather_type(grid, interval_dataset[config['predictors']].copy())
                        except:
                                raise ValueError("Importing weather type classifiers failed.")
                        classified_weather_type['time_interval'] = interval
                        print(f"Weather type Predictions done for {interval}")
                        new_df = pd.concat([new_df, classified_weather_type])
                new_df = new_df.sort_index()
                return new_df
                
        @st.cache_data
        def standardize_data_weather_forecast(df):
                directory_path = './Fitted_Standardizers'
                file_path = os.path.join(directory_path, 'std_scaler.bin')

                X_new_test = df[config['standardize_predictor_list']]
                #save fitted predictor
                predictor_scaler_fit = joblib.load(file_path)
                X_new_test = predictor_scaler_fit.transform(X_new_test)
                
                new_stand_df = pd.DataFrame(X_new_test, index=df[config['standardize_predictor_list']].index, columns=df[config['standardize_predictor_list']].columns)
                new_stand_df = pd.concat([new_stand_df, df[['season','weather_type', 'time_interval']]], axis = 1)
                return  new_stand_df
        @st.cache_data
        def predict_forecast_RF(new_stand_test):
                forecast_test = pd.DataFrame()
                for interval, weather_type in product(config['time_intervals'], config['weather_types']):
                        X_test = new_stand_test[(new_stand_test['time_interval'] == interval) & (new_stand_test['weather_type'] == weather_type)][config['predictors']]
                        if len(X_test != 0):
                                md = joblib.load(f'./Fitted_Models/RF_fitted_{interval}_{weather_type}.pkl')
                                predictions = md.predict(X_test)
                                print(f"Energy Predictions done for {interval, weather_type}")
                                TestingData=pd.DataFrame(data=X_test.copy(), columns=X_test.columns)
                                TestingData['PredictedTotalPower']=predictions
                                forecast_test = pd.concat([forecast_test, TestingData])
                forecast_test = forecast_test.sort_index()
                return forecast_test


        weather_forecast_df = get_weather_forecast_data()
        weather_forecast_df = module.add_season(weather_forecast_df)
        weather_forecast_df = module.choose_interval(weather_forecast_df)
        weather_forecast_df = module.detect_time_interval(weather_forecast_df)
        ord_enc = OrdinalEncoder()
        season = ord_enc.fit_transform(np.array(weather_forecast_df['season']).reshape(-1,1))
        weather_forecast_df['season'] = season
        weather_forecast_df = classify_weather_forecast_type(weather_forecast_df)
        weather_forecast_df_standardized = standardize_data_weather_forecast(weather_forecast_df)
        predicted_forecast = predict_forecast_RF(weather_forecast_df_standardized)
        predicted_forecast = pd.concat([predicted_forecast, weather_forecast_df_standardized[['weather_type']]], axis = 1)


        #unstandardize data

        predictor_scaler_fit = joblib.load(f'./Fitted_Standardizers/std_scaler.bin')

        unst_data = predictor_scaler_fit.inverse_transform(predicted_forecast[config['standardize_predictor_list']])

        predicted_forecast_unst = predicted_forecast.copy()
        predicted_forecast_unst[config['standardize_predictor_list']] = unst_data

        with chart_container(predicted_forecast_unst):
                        line_chart(
                        data=predicted_forecast_unst.reset_index(),
                        x='date',
                        y='PredictedTotalPower',
                        title="Forecast - Energy Generation for the next 3 days",
                )
                        