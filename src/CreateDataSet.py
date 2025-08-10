# This is a pre-processing step, typically run once
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

# Save the preprocessed data to the 'data' directory
df.to_csv('data/iris.csv', index=False)
print("Dataset saved to data/iris.csv")
