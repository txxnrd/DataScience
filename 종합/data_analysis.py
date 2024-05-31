import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define the list of filenames
filenames = [
    "2018_6.csv", "2018_9.csv", "2018_11.csv",
    "2019_6.csv", "2019_9.csv", "2019_11.csv",
    "2020_6.csv", "2020_9.csv", "2020_11.csv",
    "2021_6.csv", "2021_9.csv", "2021_11.csv",
    "2022_6.csv", "2022_9.csv", "2022_11.csv"
]

# Create a dictionary to hold the dataframes
dataframes = {}
standard_max={}
standard_min={}


# Load each file into the dictionary with its filename as the key
for filename in filenames:
    max_and_min=[]
    var_name = filename.split('.')[0]  # Use filename without extension as variable name
    df = pd.read_csv(filename, encoding='utf-8')
    standard_max[var_name] = df['standard_score'].max()
    standard_min[var_name] = df['standard_score'].min()
    dataframes[var_name] = df
# Prepare the data for training
train_data = []
train_target = []

# Determine the global max and min standard scores
global_max_score = min(standard_max.values())
global_min_score = max(standard_min.values())


# Function to normalize scores
def normalize_score(df, global_min, global_max,decimals=0):
    local_min = df['standard_score'].min()
    local_max = df['standard_score'].max()
    df['normalized_score'] = (df['standard_score'] - local_min) / (local_max - local_min) * (global_max - global_min) + global_min
    df['normalized_score'] = df['normalized_score'].round(decimals)

    return df


# Normalize each dataframe to the global max and min standard scores
for key in dataframes.keys():
    df = dataframes[key]
    normalized_df = normalize_score(df, global_min_score, global_max_score)
    dataframes[key] = normalized_df

    # Group by normalized_score and sum male and female columns
    grouped_df = normalized_df.groupby('normalized_score', as_index=False).agg({
        'male': 'sum',
        'female': 'sum'
    })

    # Create a DataFrame with all possible normalized_scores in the range
    all_scores = pd.DataFrame({'normalized_score': range(global_min_score, global_max_score + 1)})

    # Merge the grouped_df with all_scores to fill missing scores with 0 values
    complete_df = pd.merge(all_scores, grouped_df, on='normalized_score', how='left').fillna(0)


    # Update the dataframe in the dictionary
    dataframes[key] = complete_df

    # Print to verify
    print(f'\nDataFrame: {key}')
    print(complete_df)

#
# # Loop through each year and collect the data
# for year in range(2018, 2023):
#     june_data = dataframes[f'{year}_6']
#     september_data = dataframes[f'{year}_9']
#     november_data = dataframes[f'{year}_11']
#
#     # Merge June and September data
#     merged_data = pd.merge(june_data, september_data, on='standard_score', suffixes=('_june', '_sept'))
#
#     # Append the features and target data
#     train_data.append(merged_data[['male_june', 'female_june', 'male_sept', 'female_sept']])
#     train_target.append(november_data[['male', 'female']])
#
# # Concatenate all years' data
# X_train = pd.concat(train_data)
# y_train = pd.concat(train_target)
# print(X_train)
# print(y_train)
#
#
#
# print("------------------------------------------")
#
# print(standard_max)
# print(standard_min)
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
#
# # Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')
