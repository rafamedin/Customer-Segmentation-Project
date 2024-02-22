import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(0)
n_samples = 1000

# Age between 18 and 70
age = np.random.randint(18, 71, size=n_samples)

# Income follows a normal distribution with mean 50,000 and standard deviation 20,000
income = np.random.normal(loc=50000, scale=20000, size=n_samples).astype(int)

# Spending Score between 1 and 100
spending_score = np.random.randint(1, 101, size=n_samples)

# Gender: 0 for male, 1 for female
gender = np.random.randint(0, 2, size=n_samples)

# Create DataFrame
customer_data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Spending Score': spending_score,
    'Gender': gender
})

# Save DataFrame to CSV
customer_data.to_csv('customer_data.csv', index=False)

print("Sample customer_data.csv file created successfully!")
