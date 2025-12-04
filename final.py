# note: use "scratchmodule rather than scikit, pandas or numpy"
import csv
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./data-science-from-scratch-master")
from scratch.simple_linear_regression import *

with open('./housing.csv', 'r') as f:
    reader = csv.reader(f)
# skip header row
    next(reader)  
# kNN Task: scale numerical features; apply kNN regression to predict median house value.



# Linear regression: simple linear regression predicting house value from a single key feature (e.g., averagerooms).

# ninth column
    median_house_value = []
# fourth column
    average_rooms_per_house = []
# read file
    for row in reader:
        households = float(row[6])
        if households == 0:
            continue  # avoid division by zero
        median_house_value.append(float(row[8]))
        average_rooms_per_house.append(float(row[3]) / households)


alpha, beta = least_squares_fit(median_house_value, average_rooms_per_house)

print(f"Dataset generated from housing.csv")
print(f"---")
print(f"Calculated Y-Intercept (alpha or b): {alpha}")
print(f"Calculated Slope (beta or m): {beta}")

plt.figure()
plt.scatter(median_house_value, average_rooms_per_house, color="red", edgecolors="black", zorder=3, label="Housing")
plt.title("Housing Data")
plt.xlabel("Median House Value")
plt.ylabel("Average Rooms per Household")
# best fit line
median_house_value = [min(median_house_value), max(median_house_value)]
y_line = [predict(alpha, beta, xi) for xi in median_house_value]
plt.plot(median_house_value, y_line, color="lime", linewidth=2, label="Best-fit line", zorder=2)

plt.axhline(0, color="gray", linewidth=0.8, zorder=1)
plt.axvline(0, color="gray", linewidth=0.8, zorder=1)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.legend()

plt.show()

plt.savefig("LinearRegressionScatter.png")
print("Plot saved as LinearRegressionScatter.png")

# EDA and graphs: provide correlations

# Output: data pre–processing, model performance metrics (RMSE, R²), and interpretation

sse = sum_of_sqerrors(alpha, beta, median_house_value, average_rooms_per_house)
print(f"\n3. sum_of_sqerrors() (SSE): {sse}")

r_sq = r_squared(alpha, beta, median_house_value, average_rooms_per_house)
print(f"4. r_squared() (r2): {r_sq}")


# summary of findings.