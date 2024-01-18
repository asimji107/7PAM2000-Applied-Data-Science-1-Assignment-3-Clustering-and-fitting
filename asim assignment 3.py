import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Model
import matplotlib.font_manager as fm

def read_clean_transpose_csv_and_impute_missing(csv_file_path):
    """
    Reads data from a CSV file, cleans the data, and returns the original,
    cleaned, and transposed data. It also imputes missing values using the mean.

    Parameters:
    - csv_file_path (str): Path to the CSV file.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV.
    - cleaned_data (pd.DataFrame): Data after cleaning and imputation.
    - transposed_data (pd.DataFrame): Transposed data.
    """

    # Read the data from the CSV file
    original_data = pd.read_csv(csv_file_path)

    # Replace non-numeric values with NaN
    original_data.replace('..', np.nan, inplace=True)

    # Select relevant columns
    relevant_columns = [
        "CO2 emissions (metric tons per capita)",
        "CO2 emissions from solid fuel consumption (% of total)",
        "CO2 emissions from liquid fuel consumption (% of total)",
        "GDP per capita growth (annual %)"
    ]

    # Create a SimpleImputer instance with strategy='mean'
    mean_imputer = SimpleImputer(strategy='mean')

    # Apply imputer to fill missing values
    cleaned_data = original_data.copy()
    cleaned_data[relevant_columns] = mean_imputer.fit_transform(cleaned_data[relevant_columns])

    # Transpose the data
    transposed_data = cleaned_data.transpose()

    return original_data, cleaned_data, transposed_data


def fit_exponential_growth_model(time_points, actual_values):
    """
    Fits an exponential growth model to the provided time points and actual values.

    Parameters:
    - time_points (array-like): Time points.
    - actual_values (array-like): Actual data values.

    Returns:
    - fitting_result (lmfit.model.ModelResult): Result of the curve fitting.
    """

    # Define the exponential growth model function
    def exponential_growth_model(x, amplitude, growth_rate):
        return amplitude * np.exp(growth_rate * np.array(x))

    # Create a model and set initial parameters
    model = Model(exponential_growth_model)
    initial_params = model.make_params(amplitude=1, growth_rate=0.001)

    # Fit the model to the data
    fitting_result = model.fit(actual_values, x=time_points, params=initial_params)

    return fitting_result


def plot_curve_fit_with_confidence_interval(time_points, actual_values, fitting_result):
    """
    Plot the actual data, fitted curve, and confidence interval.

    Parameters:
    - time_points (array-like): Time points.
    - actual_values (array-like): Actual data values.
    - fitting_result (lmfit.model.ModelResult): Result of the curve fitting.

    Returns:
    None
    """

    # Set custom font to a system font
    custom_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='arial')))

    # Set a beautiful style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Scatter plot for actual data
    sns.scatterplot(x=time_points, y=actual_values, label='Actual Data', color='#4C72B0', s=80)

    # Line plot for the exponential growth fit
    sns.lineplot(x=time_points, y=fitting_result.best_fit, label='Exponential Growth Fit', color='#55A868', linewidth=2)

    # Confidence interval plot
    plt.fill_between(time_points, fitting_result.best_fit - fitting_result.eval_uncertainty(),
                     fitting_result.best_fit + fitting_result.eval_uncertainty(),
                     color='#55A868', alpha=0.2, label='Confidence Interval')

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('CO2 Emissions (metric tons per capita)', fontproperties=custom_font, fontsize=14)
    plt.ylim(0, 30)
    plt.title('Curve Fit for CO2 Emissions Over Time', fontproperties=custom_font, fontsize=16)
    plt.legend(prop=custom_font)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set background color
    plt.gca().set_facecolor('#F0F0F0')

    # Set tick color
    plt.tick_params(axis='both', colors='black')

    # Set spines color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#000000')

    plt.show()
def plot_cluster_with_centers(data, cluster_centers):
    """
    Plot clustering results with cluster centers.

    Parameters:
    - data (DataFrame): The input data.
    - cluster_centers (array-like): Coordinates of cluster centers.

    Returns:
    None
    """
    
    # Set custom font to a system font
    custom_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='arial')))

    # Set a beautiful style
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(12, 8))

    # Scatter plot for data points
    sns.scatterplot(x="GDP per capita growth (annual %)", y="CO2 emissions (metric tons per capita)",
                    hue="Cluster", palette="viridis", data=data, s=80, edgecolor='w', linewidth=0.5)

    # Scatter plot for cluster centers
    sns.scatterplot(x=cluster_centers[:, 1], y=cluster_centers[:, 3], marker='o', s=200, color='red', label='Cluster Centers',
                    edgecolor='k', linewidth=1.5)

    plt.title('Clustering of Countries with Cluster Centers', fontproperties=custom_font, fontsize=16)
    plt.xlabel('GDP per capita growth (annual %)', fontproperties=custom_font, fontsize=14)
    plt.ylabel('CO2 emissions (metric tons per capita)', fontproperties=custom_font, fontsize=14)
    plt.legend(prop=custom_font)

    # Set background color
    plt.gca().set_facecolor('#F0F0F0')

    # Set tick color
    plt.tick_params(axis='both', colors='black')

    # Set spines color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#000000')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()
 
def plot_co2_emissions_over_time(time_points, co2_emissions_data, future_years, predicted_values):
    """
    Plot CO2 emissions over time with predictions.

    Parameters:
    - time_points (array-like): Time points for actual data.
    - co2_emissions_data (array-like): Actual CO2 emissions data.
    - future_years (array-like): Time points for future predictions.
    - predicted_values (array-like): Predicted CO2 emissions values.

    Returns:
    None
    """

    # Set custom font to a system font
    custom_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='arial')))

    # Set a beautiful style
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(12, 8))

    # Line plot for actual CO2 emissions
    sns.lineplot(x=time_points, y=co2_emissions_data, label='CO2 Emissions (Actual)', color='#4C72B0', linewidth=2)

    # Line plot for predicted values
    plt.plot(future_years, predicted_values, label='Predicted Values', color='#55A868', linestyle='--', linewidth=2)

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('CO2 Emissions (metric tons per capita)', fontproperties=custom_font, fontsize=14)
    plt.title('CO2 Emissions Over Time with Predictions', fontproperties=custom_font, fontsize=16)
    plt.legend(prop=custom_font)

    # Set background color
    plt.gca().set_facecolor('#F0F0F0')

    # Set tick color
    plt.tick_params(axis='both', colors='black')

    # Set spines color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#000000')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add a note about extrapolation
    plt.annotate('Note: Future predictions beyond observed data are extrapolations. Exercise caution in interpretation.',
                 xy=(0.5, -0.1), xycoords="axes fraction",
                 ha="center", va="center", fontsize=10, color='gray', style='italic')

    plt.show()   
def plot_co2_emissions_australia(time_points, co2_emissions_data, future_years, predicted_values):
    """
    Plot CO2 emissions for Australia over time with predictions.

    Parameters:
    - time_points (array-like): Time points for actual data.
    - co2_emissions_data (array-like): Actual CO2 emissions data for Australia.
    - future_years (array-like): Time points for future predictions.
    - predicted_values (array-like): Predicted CO2 emissions values for Australia.

    Returns:
    None
    """

    # Set custom font to Arial
    custom_font = fm.FontProperties(family='Arial')

    # Set a beautiful style
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(12, 8))

    # Line plot for actual CO2 emissions
    sns.lineplot(x=time_points, y=co2_emissions_data, label='CO2 Emissions for Australia (Actual)', color='#007F00', linewidth=2)

    # Line plot for predicted values
    plt.plot(future_years, predicted_values, label='Predicted Values', color='#FF0000', linestyle='--', linewidth=2)

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('CO2 Emissions (metric tons per capita)', fontproperties=custom_font, fontsize=14)
    plt.title('CO2 Emissions for Australia Over Time with Predictions (1990-2040)', fontproperties=custom_font, fontsize=16)
    plt.legend(prop=custom_font)

    # Set background color
    plt.gca().set_facecolor('#F0F0F0')

    # Set tick color
    plt.tick_params(axis='both', colors='black')

    # Set spines color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#000000')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show() 
def plot_co2_emissions_germany(time_points, co2_emissions_data, future_years, predicted_values):
    """
    Plot CO2 emissions for Germany over time with predictions.

    Parameters:
    - time_points (array-like): Time points for actual data.
    - co2_emissions_data (array-like): Actual CO2 emissions data for Germany.
    - future_years (array-like): Time points for future predictions.
    - predicted_values (array-like): Predicted CO2 emissions values for Germany.

    Returns:
    None
    """

    # Set custom font to Arial
    custom_font = fm.FontProperties(family='Arial')

    # Set a beautiful style
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(12, 8))

    # Line plot for actual CO2 emissions
    sns.lineplot(x=time_points, y=co2_emissions_data, label='CO2 Emissions for Germany (Actual)', color='#6A0DAD', linewidth=2)

    # Line plot for predicted values
    plt.plot(future_years, predicted_values, label='Predicted Values', color='#FFA500', linestyle='--', linewidth=2)

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('CO2 Emissions (metric tons per capita)', fontproperties=custom_font, fontsize=14)
    plt.title('CO2 Emissions for Germany Over Time with Predictions (1990-2040)', fontproperties=custom_font, fontsize=16)
    plt.legend(prop=custom_font)

    # Set background color
    plt.gca().set_facecolor('#F0F0F0')

    # Set tick color
    plt.tick_params(axis='both', colors='black')

    # Set spines color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#000000')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()
# Main Code

csv_file_path = "asim 3 data.csv"
original_data, cleaned_data, transposed_data = read_clean_transpose_csv_and_impute_missing(csv_file_path)

scaler = StandardScaler()
relevant_columns = [
    "CO2 emissions (metric tons per capita)",
    "CO2 emissions from solid fuel consumption (% of total)",
    "CO2 emissions from liquid fuel consumption (% of total)",
    "GDP per capita growth (annual %)"
]
normalized_data = scaler.fit_transform(cleaned_data[relevant_columns])

kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(normalized_data)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

silhouette_avg = silhouette_score(normalized_data, cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

plot_cluster_with_centers(cleaned_data, cluster_centers)

time_points = cleaned_data['Time']
co2_emissions_data = cleaned_data['CO2 emissions (metric tons per capita)']

fitting_result = fit_exponential_growth_model(time_points, co2_emissions_data)

# Plot the curve fit for the entire dataset
plot_curve_fit_with_confidence_interval(time_points, co2_emissions_data, fitting_result)

# Generate time points for prediction for the entire dataset
future_years = [2025, 2030, 2035,2040,2045]

# Predict values for the future years using the fitted model for the entire dataset
predicted_values = fitting_result.eval(x=np.array(future_years))

# Display the predicted values for the entire dataset
for year, value in zip(future_years, predicted_values):
    print(f"Predicted value for {year} is : {value:.2f}")

# Filter data for Australia
australia_data = cleaned_data[cleaned_data['Country Name'] == 'Australia']

# Extract relevant data for Australia
time_points_australia = australia_data['Time']
co2_emissions_australia = australia_data['CO2 emissions (metric tons per capita)']

future_years_australia = list(range(1990, 2040))
predicted_values_australia = fitting_result.eval(x=np.array(future_years_australia))

plot_co2_emissions_over_time(time_points, co2_emissions_data, future_years, predicted_values)

plot_co2_emissions_australia(time_points_australia, co2_emissions_australia, future_years_australia, predicted_values_australia)



# Filter data for Germany
germany_data = cleaned_data[cleaned_data['Country Name'] == 'Germany']

# Extract relevant data for Germany
time_points_germany = germany_data['Time']
co2_emissions_germany = germany_data['CO2 emissions (metric tons per capita)']

predicted_values_germany = fitting_result.eval(x=np.array(future_years_australia))
plot_co2_emissions_germany(time_points_germany, co2_emissions_germany, future_years_australia, predicted_values_germany)

