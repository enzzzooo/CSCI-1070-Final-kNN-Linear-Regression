# note: use "scratchmodule rather than scikit, pandas or numpy"
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
sys.path.insert(0, "./data-science-from-scratch-master")
from scratch.linear_algebra import distance
from scratch.probability import normal_pdf
from scratch.simple_linear_regression import *
from scratch.statistics import *
import math
from typing import NamedTuple
import random
from typing import List

# --------------------- Prior Exploration ---------------------

def feature_correlations_by_proximity(proximity_type, max_value=False):
    """
    Compute correlations between all numeric features and median_house_value for a specified proximity  
    """
    if max_value != False: 
        print("\n--- Feature Correlations (Filtered for", proximity_type, "with max value filter) ---")
    else:
        print("\n--- Feature Correlations (Filtered for", proximity_type, ") ---") 

    # STEP ONE: Load dataset
    features = {
        "longitude": [],
        "latitude": [],
        "housing_median_age": [],
        "total_rooms": [],
        "total_bedrooms": [],
        "population": [],
        "households": [],
        "median_income": [],
        "average_bedrooms_per_house": [],
    }
    median_house_value = []

    with open("./housing.csv", "r") as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            if row[9] != proximity_type:
                continue

            # Skip rows with missing values in numeric columns
            if '' in row[0:8]:
                continue

            # Apply max value filter if specified, this is because I wanted to compare results so I added this boolean flag 
            if max_value != False:
                if float(row[8]) >= 500000:
                    continue 

            households = float(row[6])
            if households == 0:
                continue  # avoid division by zero just in case

            # Append the data to respective feature lists  
            features["longitude"].append(float(row[0]))
            features["latitude"].append(float(row[1]))
            features["housing_median_age"].append(float(row[2]))
            features["total_rooms"].append(float(row[3]))
            features["total_bedrooms"].append(float(row[4]))
            features["population"].append(float(row[5]))
            features["households"].append(float(row[6]))
            features["median_income"].append(float(row[7]))
            features["average_bedrooms_per_house"].append(float(row[4]) / households)
            median_house_value.append(float(row[8]))

    # Print correlations
    print(f"\n--- Correlations with Median House Value ({proximity_type}) ---")
    for feature_name, values in features.items():
        # from the scratch.statistics module 
        corr = correlation(values, median_house_value)
        print(f"{feature_name}: {corr:.3f}")


# --------------------- Problem 1: kNN Regression ---------------------

# For kNN regression model I tried to follow the example in the sratch module as closely as possible
# adapting it as needed for regression rather than classification 

# Adapt scratch LabeledPoint class
class LabeledPoint(NamedTuple):
    point: Vector
    label: float

def z_score_scale(values):
    mu = mean(values)
    sigma = standard_deviation(values)
    # avoid division by zero, returns all 0s if sigma is 0 
    if sigma == 0:
        return [0.0 for _ in values]
    # otherwise apply z-score formula to each value 
    return [(v - mu) / sigma for v in values]

# instead of voting for classification, we do regression by averaging the k nearest labels to output a continous numeric value 
# I just used the def knn_classify structure from scratch module as a guide
# all we had to adapt was the return statement to average the k nearest labels rather than majority vote 
def knn_regression(k: int, train_points: List[Vector], train_targets: List[float], new_point: Vector) -> float:
    """Predict numeric label using kNN: average of k nearest labels"""
    # Order points by distance
    by_distance = sorted(
        zip(train_points, train_targets),
        key=lambda pair: distance(pair[0], new_point)
    )
    # Take labels of k nearest
    k_labels = [label for _, label in by_distance[:k]]
    # Regression = average of k labels
    return mean(k_labels)

# this is my function to tune k and plot RMSE vs k for kNN regression model 
# my goal is the find the best k value based on RMSE 
def tune_knn_and_plot(k_range=range(1, 56, 2),max_value=False):
    """
    Elbow Plot to "tune" k for kNN regression model (near ocean)
    """

    # allow the flag to filter out capped values for comparison purposes, again 
    if max_value != False:
        print("\n--- Tuning kNN (Filtered for NEAR OCEAN with max value filter) ---")       
    else:
        print("\n--- Tuning kNN (Filtered for NEAR OCEAN) ---")
   
    median_income = []
    average_rooms_per_house = []
    median_house_value = []

    # --- A. DATA ACQUISITION (filter NEAR OCEAN) ---
    with open("./housing.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            # only use data points NEAR OCEAN
            if row[9] != "NEAR OCEAN":
                continue

            # ensure no division by 0
            households = float(row[6])
            if households == 0:
                continue  # skip invalid rows

            # if the flag is set, fitler out capped values
            if max_value != False:
                if float(row[8]) >= 500000:
                    continue  # skip capped valuess
            # append featurs 
            median_income.append(float(row[7]))
            average_rooms_per_house.append(float(row[3]) / households)
            median_house_value.append(float(row[8]))  # target

    # --- B. NORMALIZATION (Z-score) ---
    median_income_scaled = z_score_scale(median_income)
    avg_rooms_scaled = z_score_scale(average_rooms_per_house)

    # combine features into 2D vectors
    points = [[median_income_scaled[i], avg_rooms_scaled[i]] for i in range(len(median_house_value))]

    # --- C. TRAIN / TEST SPLIT (80/20) ---
    # 80% train so we need to slice at 0.8* of the length of the points 
    split_index = int(0.8 * len(points))
    train_points = points[:split_index]
    train_targets = median_house_value[:split_index]
    test_points = points[split_index:]
    test_targets = median_house_value[split_index:]

    # --- D. EVALUATE kNN over range of k ---
    rmse_values = []
    for k in k_range:
        predictions = [knn_regression(k, train_points, train_targets, p) for p in test_points]
        squared_errors = [(predictions[i] - test_targets[i])**2 for i in range(len(test_targets))]
        rmse = math.sqrt(sum(squared_errors)/len(test_targets))
        rmse_values.append(rmse)
        print(f"k = {k} | RMSE = {rmse:.2f}")

    # --- E. PLOT RMSE vs k ---
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), rmse_values, marker='o')
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("RMSE")
    plt.title("kNN Regression (NEAR OCEAN): RMSE vs k")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("KNN_RMSE_vs_k.png")
    plt.show()

    # print the k that is associated with the smallest RMSE
    # note this isnt a tpical grab from the elbow way 
    # I decided to go for the k associated withthe smallest RMSE because I wanted to optimize the models results or fit as much as possible
    # because with k=10 i got R^2: 0.3253127651171194 vs k=55 R^2: 0.38779783793630496
    best_index = rmse_values.index(min(rmse_values))
    best_k = list(k_range)[best_index]
    print(f"\nBest k based on RMSE: {best_k}")
    return best_k


def run_knn_regression_P1(k=5, max_value=False):
    """
    Problem 1: kNN model using median_income + average_rooms_per_house
    """
    # allow the flag to filter out capped values for comparison purposes, again
    if max_value != False:
        print("\n--- Problem 1. kNN Regression (Filtered for NEAR OCEAN with max value filter) ---")
    else:
        print("\n--- Problem 1. kNN Regression (Filtered for NEAR OCEAN) ---")
        
    median_income = []
    average_rooms_per_house = []
    median_house_value = []

    # --- A. DATA ACQUISITION ---
    with open("./housing.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            ocean = row[9]
            if ocean != "NEAR OCEAN":
                continue

            households = float(row[6])
            if households == 0:
                continue

            if max_value != False:
                if float(row[8]) >= 500000:
                    continue  # skip capped valuess

            median_income.append(float(row[7]))
            average_rooms_per_house.append(float(row[3]) / households)
            median_house_value.append(float(row[8]))

    print("\n--- Dataset Loaded (NEAR OCEAN) ---")
    print("Total data points:", len(median_house_value))

    # --- B. STANDARDIZATION ---
    median_income_scaled = z_score_scale(median_income)
    avg_rooms_scaled = z_score_scale(average_rooms_per_house)
    points = [[median_income_scaled[i], avg_rooms_scaled[i]] for i in range(len(median_house_value))]

    # --- C. TRAIN / TEST SPLIT ---
    # 80% train so we need to slice at 0.8* of the length of the points 
    split_index = int(0.8 * len(points))
    train_points = points[:split_index]
    train_targets = median_house_value[:split_index]
    test_points = points[split_index:]
    test_targets = median_house_value[split_index:]

    # --- D. PREDICTIONS ---
    predictions = [knn_regression(k, train_points, train_targets, p) for p in test_points]

    # --- E. PERFORMANCE METRICS ---
    # the sum of the squared errors is calculaed by squaring the difference between each predicted and actual value
    # summing these squared errors as the name indicates gives us the SSE 
    squared_errors = [(predictions[i] - test_targets[i])**2 for i in range(len(test_targets))]
    sse = sum(squared_errors)
    rmse = math.sqrt(sse / len(test_targets))
    mean_y = mean(test_targets)
    ss_total = sum((y - mean_y)**2 for y in test_targets)
    r_sq = 1 - sse / ss_total

    print("\n--- kNN Regression Performance (NEAR OCEAN) ---")
    print(f"k = {k}")
    print(f"SSE: {sse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r_sq}")

    # --- F. SCATTER PLOT ---
    plt.figure(figsize=(8, 6))
    plt.scatter(test_targets, predictions, alpha=0.3, edgecolors="black")
    plt.xlabel("True Median House Value")
    plt.ylabel("Predicted Median House Value (kNN)")
    plt.title(f"kNN Regression (NEAR OCEAN), k={k}")
    plt.grid(True, linestyle="--", alpha=0.5)

    # using built in function from matplotlib to format axis with commas
    comma_formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(comma_formatter)
    ax.yaxis.set_major_formatter(comma_formatter)

    plt.savefig("KNN_True_vs_Predicted2.png")
    plt.show()

# --------------------- Problem 2: Simple Linear Regression ---------------------

# for this problem I followed the structure of the scratch module linear regression example very closely 
def run_linear_regression_P2(max_value=False):
    """
    Simple Linear Regression (NEAR OCEAN)
    Predict median house value using a single feature: median_income
    """

    # again for comparison purposes I added the max_value boolean flag 
    if max_value != False:
        print("\n--- Problem 2. Linear Regression (Filtered for NEAR OCEAN with max value filter) ---")
    else:
        print("\n--- Problem 2. Linear Regression (NEAR OCEAN) ---")

    median_income = []         # Independent variable
    median_house_value = []    # Dependent variable (target)

    # A. DATA ACQUISITION (filter NEAR OCEAN)
    with open('./housing.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:

            # only use data points NEAR OCEAN 
            if row[9] != "NEAR OCEAN":
                continue  

            # skip rows with missing values    
            if '' in row[7:9]:
                continue  

            if max_value != False:
                if float(row[8]) >= 500000:
                    continue  # skip capped valuess
                
            median_income.append(float(row[7]))           
            median_house_value.append(float(row[8]))       
    print("--- Dataset Loaded ---")
    print("Total data points:", len(median_house_value))

    # B. CORRELATION
    corr = correlation(median_income, median_house_value)
    print(f"Correlation with median_house_value: {corr:.3f}")

    # C. LINEAR REGRESSION MODEL TRAINING
    alpha, beta = least_squares_fit(median_income, median_house_value)

    # D. PERFORMANCE METRICS (using scratch module functions)
    sse = sum_of_sqerrors(alpha, beta, median_income, median_house_value)
    r_sq = r_squared(alpha, beta, median_income, median_house_value)
    rmse = math.sqrt(sse / len(median_house_value))

    print("\n--- Linear Regression Performance ---")
    print(f"Alpha (intercept): {alpha}")
    print(f"Beta (slope): {beta}")
    print(f"SSE: {sse}")
    print(f"RMSE: {rmse}")
    print(f"R^2: {r_sq}")

    # E. PLOT: Distribution of house values
    plt.figure(figsize=(12, 6))
    # divide the histogram into 30 bins
    plt.hist(median_house_value, bins=30, density=True, alpha=0.5, edgecolor='black')
    # get avg house value and standard deviation (spread of the data is from mean)
    # i wanted to plot a normal distribution curve over the histogram to visualize how closely the data follows a normal distribution
    # but i didnt see anything in the scratch modeule to do so 
    plt.title("Histogram: Median House Values (NEAR OCEAN)")
    plt.xlabel("Median House Value")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("Histogram_Median_House_Values.png")
    plt.show()

    # F. PLOT: Linear regression
    plt.figure(figsize=(14, 7))
    plt.scatter(median_income, median_house_value, edgecolors="black", zorder=3)
    x_line = [min(median_income), max(median_income)]
    y_line = [predict(alpha, beta, x) for x in x_line]
    plt.ylim(0, 510000)  
    plt.plot(x_line, y_line, color="red", linewidth=2, zorder=2)
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")
    plt.title("Linear Regression (NEAR OCEAN) - Median Income")
    plt.grid(True, linestyle="--", alpha=0.5)
    # using built in function from matplotlib to format axis with commas 
    comma_formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(comma_formatter)
    ax.yaxis.set_major_formatter(comma_formatter)
    plt.savefig("LinearRegression_MedianIncome.png")
    plt.show()




def main():
    # best_k = tune_knn_and_plot()

    # feature_correlations_by_proximity("NEAR OCEAN", max_value=True)
    # run_knn_regression_P1(k=55, max_value=True)
    # run_linear_regression_P2(max_value=True)

    # feature_correlations_by_proximity("NEAR OCEAN")
    run_knn_regression_P1(k=55)
    run_knn_regression_P1(k=10)

    run_linear_regression_P2()


if __name__ == "__main__":
    main()