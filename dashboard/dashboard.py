import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

# Function to create the daily rents dataframe
def create_daily_rents_df(df):
    if df.empty:
        st.error("The dataframe is empty. Please check the date range or data.")
        return None
    
    try:
        # Resampling data by day and aggregating rent count and daily average
        daily_rents_df = df.resample(rule="D", on='dteday').agg({
            "cnt_y": "nunique",  # Assumes cnt_y is a unique rent identifier
            "avg_cnt": "sum"     # Sum of average counts for the day
        })
        
        # Reset the index
        daily_rents_df = daily_rents_df.reset_index()

        # Rename columns
        daily_rents_df.rename(columns={
            "cnt_y": "rent_count",
            "avg_cnt": "daily_avg"
        }, inplace=True)
        
        # Return the final daily rents dataframe
        return daily_rents_df

    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

def create_sum_rent_season(df):
    sum_rent_season_df = df.groupby("season_x").cnt_x.sum().sort_values(ascending=False).reset_index()
    return sum_rent_season_df

def create_bytime_df(df):
    bytime_df = df.groupby(by="time_of_day").dteday.nunique().reset_index()
    bytime_df.rename(columns={
        "dteday":"waktu"
    }, inplace=True)
    return bytime_df

def create_byweather_df(df):
    byweather_df = df.groupby(by="weathersit_x").dteday.nunique().reset_index()
    byweather_df.rename(columns={
        "dteday" : "cuaca"
    }, inplace=True)
    return byweather_df

def create_rfm_df(df):
    # Aggregating the frequency and monetary values
    rfm_df = df.groupby(by="instant_x", as_index=False).agg({
        "dteday": "nunique",  # Frequency: unique days
        "cnt_x": "sum"        # Monetary: total count of transactions
    }).rename(columns={
        "dteday": "frequency",
        "cnt_x": "monetary"
    })

    # Get the maximum date for recency
    max_order_timestamp = df.groupby(by="instant_x")["dteday"].max().reset_index()

    # Merge to include recency information
    rfm_df = rfm_df.merge(max_order_timestamp, on="instant_x", how="left")
    
    # Compute recency
    rfm_df["max_order_timestamp"] = rfm_df["dteday"].dt.date
    recent_date = df["dteday"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    
    # Drop the max order timestamp as it's not needed anymore
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    return rfm_df
# Load the dataset
bikenew_df = pd.read_csv("bike_new.csv")
bikenew_df["dteday"] = pd.to_datetime(bikenew_df["dteday"])


# Get the date range from the dataset
min_date = bikenew_df["dteday"].min()
max_date = bikenew_df["dteday"].max()

# Sidebar with date input
with st.sidebar:
    st.image("logo1.png")
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Filter the main dataframe based on selected dates
main_df = bikenew_df[(bikenew_df["dteday"] >= str(start_date)) & 
                     (bikenew_df["dteday"] <= str(end_date))]

# Debugging: Print the shape of the filtered dataframe
st.write("Shape of the filtered dataframe:", main_df.shape)

# Create daily rents dataframe
daily_rents_df = create_daily_rents_df(main_df)
sum_rent_season_df = create_sum_rent_season(main_df)
bytime_df = create_bytime_df(main_df)
byweather_df = create_byweather_df(main_df) 
rfm_df = create_rfm_df(main_df)

# Check if the daily_rents_df is None (indicating an error)
if daily_rents_df is not None and not daily_rents_df.empty:
    # Header and subheader
    st.header('Bike Sharing Dashboard')
    st.subheader('Daily Rents')

    # Display metrics in columns
    col1, col2 = st.columns(2)

    with col1:
        total_rents = daily_rents_df.rent_count.sum()
        st.metric("Total Rent", value=total_rents)

    with col2:
        daily_avg = daily_rents_df.daily_avg.mean()  # Using mean for daily average
        st.metric("Daily Average", value=daily_avg)

    # Plot the daily rents
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        daily_rents_df["dteday"],
        daily_rents_df["rent_count"],
        marker='o', 
        linewidth=2,
        color="#90CAF9"
    )
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)

    # Show the plot
    st.pyplot(fig)

else:
    st.error("No data available for the selected date range.")

# --------------------------------------------------------------------------
st.subheader("Best & Worst Performing Season")
st.text("the season explain:")
st.text("1 = spring")
st.text("2 = summer")
st.text("3 = fall")
st.text("4 = winter")


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="season_x", y="cnt_x", data=sum_rent_season_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Season", fontsize=30)
ax[0].set_title("Best Performing Season", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)
 
sns.barplot(x="season_x", y="cnt_x", data=sum_rent_season_df.sort_values(by="cnt_x", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Season", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Season", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)
 
st.pyplot(fig)
# --------------------------------------------------------------------------------
st.subheader("Customer Demographics")
st.text("Weather : ")
st.text("1. Clear, Few clouds, Partly cloudy, Partly cloudy")
st.text("2. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist")
st.text("3. Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds")
st.text("4. Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog")
 
col1, col2 = st.columns(2)
 
with col1:
    
    fig, ax = plt.subplots(figsize=(20, 10))
 
    sns.barplot(
        y="waktu", 
        x="time_of_day",
        data=bytime_df.sort_values(by="waktu", ascending=False),
        palette=colors,
        ax=ax
    )
    ax.set_title("Number of Customer by Time of Day", loc="center", fontsize=50)

    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)


 
with col2:


    fig, ax = plt.subplots(figsize=(20, 10))
    
    colors = ["#D3D3D3", "#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
    sns.barplot(
        y="cuaca", 
        x="weathersit_x",
        data=byweather_df.sort_values(by="weathersit_x", ascending=False),
        palette=colors,
        ax=ax
    )
    ax.set_title("Number of Customer by Weather", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

# --------------------------------------------------------------------------
st.subheader("Best Customer Based on RFM Parameters")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)
 
with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)
 
with col3:
    avg_frequency = format_currency(rfm_df.monetary.mean(), "AUD", locale='es_CO') 
    st.metric("Average Monetary", value=avg_frequency)
 
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]
 
sns.barplot(y="recency", x="instant_x", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("instant_x", fontsize=30)
ax[0].set_title("By Recency (days)", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35)
 
sns.barplot(y="frequency", x="instant_x", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("instant_x", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35)
 
sns.barplot(y="monetary", x="instant_x", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("instant_x", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=50)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35)
 
st.pyplot(fig)

st.caption('Copyright (c) rasyidinazhari 2024')
 
# fig, ax = plt.subplots(figsize=(20, 10))
# colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
# sns.barplot(
#     x="cuaca", 
#     y="state",
#     data=bystate_df.sort_values(by="cuaca", ascending=False),
#     palette=colors,
#     ax=ax
# )
# ax.set_title("Number of Customer by States", loc="center", fontsize=30)
# ax.set_ylabel(None)
# ax.set_xlabel(None)
# ax.tick_params(axis='y', labelsize=20)
# ax.tick_params(axis='x', labelsize=15)
# st.pyplot(fig)

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# from babel.numbers import format_currency
# sns.set(style='dark')

# # Function to create the daily rents dataframe
# def create_daily_rents_df(df):
#     daily_rents_df = df.resample(rule="D", on='dteday').agg({
#         "cnt_x" : "nunique",
#         "avg_cnt" : "sum"
#     })
#     daily_rents_df = daily_rents_df.reset_index()
#     daily_rents_df.rename(columns={
#         "cnt_x" : "rent_count",
#         "avg_cnt" : "daily_avg"
#     }, inplace=True)
#     return daily_rents_df  # Return the created dataframe

# # Load the data
# bikenew_df = pd.read_csv("bike_new.csv")
# bikenew_df["dteday"] = pd.to_datetime(bikenew_df["dteday"])

# # Get the date range
# min_date = bikenew_df["dteday"].min()
# max_date = bikenew_df["dteday"].max()

# # Sidebar with date input
# with st.sidebar:
#     st.image("logo1.png")
#     start_date, end_date = st.date_input(
#         label='Rentang Waktu', min_value=min_date,
#         max_value=max_date,
#         value=[min_date, max_date]
#     )

# # Filter the main dataframe based on selected dates
# main_df = bikenew_df[(bikenew_df["dteday"] >= str(start_date)) & 
#                      (bikenew_df["dteday"] <= str(end_date))]

# # Create daily rents dataframe
# daily_rents_df = create_daily_rents_df(main_df)

# # Header and subheader
# st.header('Bike Sharing Dashboard')
# st.subheader('Daily Rents')

# # Display metrics in columns
# col1, col2 = st.columns(2)

# with col1:
#     total_rents = daily_rents_df.rent_count.sum()
#     st.metric("Total Rent", value=total_rents)

# with col2:
#     daily_avg = daily_rents_df.daily_avg.mean()  # Using mean for daily average
#     st.metric("Daily Average", value=daily_avg)

# # Plot the daily rents
# fig, ax = plt.subplots(figsize=(16, 8))
# ax.plot(
#     daily_rents_df["dteday"],
#     daily_rents_df["rent_count"],
#     marker='o', 
#     linewidth=2,
#     color="#90CAF9"
# )
# ax.tick_params(axis='y', labelsize=20)
# ax.tick_params(axis='x', labelsize=15)

# # Show the plot
# st.pyplot(fig)
