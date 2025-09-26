import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

# generating simulation data from the paper(gul18)
"""
In the simulation data set, we have generated two classes.
One with 20 informative features are generated from a (correlated) 
multi-variate normal(2, sigma) distribution, the other one with 20 informative 
features are generated from independent normal(1,1) distribution.
User could tune the dataset by adding non-informative features (noises)

parameters:
variable_size: 
num_variables: the number of (informative) features added into dataset
num_noises: default = 0, the number of non-informative features added into dataset
"""
def get_data(variable_size = None, num_variables = None, num_noises = 0, omega = 1):
    # Mean vector
    mean = np.zeros(num_variables) + 2

    # Covariance matrix
    sig = np.zeros((num_variables,num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            sig[i][j] = (0.9)**np.abs(i-j)

    # Cholesky decomposition
    L = np.linalg.cholesky(omega * sig)

    # Generate independent standard normal variables
    std_normal_vars = np.random.normal(0, 1, size=(variable_size, num_variables))

    # Transform variables using the Cholesky decomposition
    correlated_vars = mean + np.dot(std_normal_vars, L.T)

    # Generate variables of size 500 from a standard normal distribution
    variables = np.zeros((num_variables, variable_size))
    for i in range(num_variables):
        variables[i] = np.random.normal(1, 1, size=variable_size)

    data_0 = pd.DataFrame(correlated_vars, columns=['feature_'+str(i+1) for i in range(num_variables)])
    data_1 = pd.DataFrame(variables.T, columns=['feature_'+str(i+1) for i in range(num_variables)])
    res_0 = pd.Series(np.zeros(variable_size))
    res_1 = pd.Series(np.zeros(variable_size) + 1)
    informative = pd.concat([data_0,data_1],ignore_index=True)
    response = pd.concat([res_0,res_1],ignore_index=True)
    response = pd.DataFrame(response).astype(int)
    size = (2*variable_size, num_noises)

    # Create random noises
    noises = pd.DataFrame(np.random.randn(*size), columns=['noises_'+str(i+1) for i in range(num_noises)])
    final = informative.join(noises)

    return final,response


def generate_simulation_data(n_samples=1000, n_irrelevant=100, n_weak_relevant=10,
                             n_strong_relevant=2):
    """
    Generates binary simulation data with irrelevant, weakly relevant,
    and strongly relevant features, including implicit interaction effects.

    Parameters:
    - n_samples (int): Number of samples.
    - n_irrelevant (int): Number of irrelevant (noisy) features.
    - n_weak_relevant (int): Number of weakly relevant features.
    - n_strong_relevant (int): Number of strongly relevant features.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - X (pd.DataFrame): DataFrame of generated features.
    - y (np.array): Binary target variable.
    """
    # Set random seed for reproducibility

    # 1. Generate Irrelevant (Noisy) Features
    irrelevant_features = np.random.normal(0, 1, (n_samples, n_irrelevant))

    # 2. Generate Weakly Relevant but Non-Redundant Features
    weak_relevant_features = np.array([
        np.random.normal(0, 1, n_samples) + np.random.normal(0, 0.5, n_samples) for _ in range(n_weak_relevant)
    ]).T

    # 3. Generate Strongly Relevant Features
    strong_relevant_features = np.array([
        np.random.normal(0, 1, n_samples) + np.random.normal(0, 0.1, n_samples) for _ in range(n_strong_relevant)
    ]).T

    # 4. Define Target Variable with Implicit Interaction Influence
    # Compute the logit with implicit interaction effects
    logit = sum(0.8 * strong_relevant_features[:, i] for i in range(n_strong_relevant))
    if n_strong_relevant > 1:
        logit += 0.5 * (strong_relevant_features[:, 0] - strong_relevant_features[:, 1])

    # Add influence from the first two weak relevant features (if available)
    if n_weak_relevant > 1:
        logit += 0.2 * weak_relevant_features[:, 0] + 0.2 * weak_relevant_features[:, 1]

    # Convert logit to a binary target using a sigmoid function
    probability = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, probability)

    # Combine all features into a single dataset without explicit interactions
    X = np.hstack([irrelevant_features, weak_relevant_features, strong_relevant_features])

    # Create a DataFrame with meaningful column names
    columns = [f'irrelevant_{i + 1}' for i in range(n_irrelevant)] + \
              [f'weak_relevant_{i + 1}' for i in range(n_weak_relevant)] + \
              [f'strong_relevant_{i + 1}' for i in range(n_strong_relevant)]

    X_df = pd.DataFrame(X, columns=columns)

    return X_df, y

def compute_error(data, label, rand_state, model, prob = False):
    sim_train_X, sim_test_X, sim_train_y, sim_test_y = train_test_split(data, label, test_size=0.2,
                                                                        random_state=rand_state)
    model.fit(sim_train_X,sim_train_y)
    if prob == True:
        y_pred = (model.predict_proba(sim_test_X))[:,1]
    else:
        y_pred = model.predict(sim_test_X)
    res = mean_squared_error(sim_test_y, y_pred)
    return res

def get_data3(variable_size = None, num_variables = None, num_noises = 0, omega = 1, scale3 = 1):
    # Mean vector
    mean = np.zeros(num_variables) + 2

    # Covariance matrix
    sig = np.zeros((num_variables,num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            sig[i][j] = (0.5)**np.abs(i-j)

    # Cholesky decomposition
    L = np.linalg.cholesky(omega * sig)

    # Generate independent standard normal variables
    std_normal_vars = np.random.normal(0, 1, size=(variable_size, num_variables))

    # Transform variables using the Cholesky decomposition
    correlated_vars = mean + np.dot(std_normal_vars, L.T)

    # Generate variables of size 500 from a standard normal distribution
    variables = np.zeros((num_variables, variable_size))
    for i in range(num_variables):
        variables[i] = np.random.normal(1, 1, size=variable_size)

    variables2 = np.zeros((num_variables, variable_size))
    for i in range(num_variables):
        variables2[i] = np.random.normal(3, scale = scale3, size=variable_size)

    data_0 = pd.DataFrame(correlated_vars, columns=['feature_'+str(i+1) for i in range(num_variables)])
    data_1 = pd.DataFrame(variables.T, columns=['feature_'+str(i+1) for i in range(num_variables)])
    data_2 = pd.DataFrame(variables2.T, columns=['feature_'+str(i+1) for i in range(num_variables)])
    res_0 = pd.Series(np.zeros(variable_size))
    res_1 = pd.Series(np.zeros(variable_size) + 1)
    res_2 = pd.Series(np.zeros(variable_size) + 2)
    informative = pd.concat([data_0,data_1,data_2],ignore_index=True)
    response = pd.concat([res_0,res_1, res_2],ignore_index=True)
    response = pd.DataFrame(response)
    size = (3*variable_size, num_noises)

    # Create random noises
    noises = pd.DataFrame(np.random.randn(*size), columns=['noises_'+str(i+1) for i in range(num_noises)])
    final = informative.join(noises)

    return final,response


def generate_linear_data(n, p, q):
    # generate data matrix for informative features
    X_rel = np.random.normal(0, 1, (n, q))
    w = np.random.normal(0, 4, q)
    z = X_rel @ w + np.random.normal(0, 0.1, n)
    y = (z > 0).astype(int)

    # generate data matrix for noisy features
    X_irr = np.random.normal(0, 1, (n, p - q))

    # concat two parts
    X = np.hstack((X_rel, X_irr))

    # convert to pd dataframe
    data = pd.DataFrame(X, columns=['feature_'+str(i+1) for i in range(q)]+ ['noises_'+str(i+1) for i in range(p-q)])
    response = pd.DataFrame(y)

    return data, response


def generate_nonlinear_data(n, p, q):
    # generate data matrix for informative features
    X_rel = np.random.uniform(0, 1, (n, q))
    z = np.sum(np.sin(X_rel) + X_rel ** 2, axis=1)
    y = (z > np.mean(z)).astype(int)

    # generate data matrix for noisy features
    X_irr = np.random.uniform(0, 1, (n, p - q))

    # concat two parts
    X = np.hstack((X_rel, X_irr))

    # convert to pd dataframe
    data = pd.DataFrame(X, columns=['feature_' + str(i + 1) for i in range(q)] + ['noises_' + str(i + 1) for i in
                                                                                  range(p - q)])
    response = pd.DataFrame(y)
    return data, response


def generate_mixed_data(n, p, q):
    # generate data matrix for informative features
    X_rel_part1 = np.random.normal(0, 1, (n, q // 2))
    X_rel_part2 = np.random.uniform(0, 1, (n, q // 2))
    X_rel = np.hstack((X_rel_part1, X_rel_part2))
    w = np.random.normal(0, 1, q)
    z = np.sum(np.sin(X_rel) + X_rel ** 2, axis=1)
    y = (z > 0).astype(int)

    # generate data matrix for noisy features
    X_irr = np.random.normal(0, 1, (n, p - q))

    # concat two parts
    X = np.hstack((X_rel, X_irr))

    # convert to pd dataframe
    data = pd.DataFrame(X, columns=['feature_' + str(i + 1) for i in range(q)] + ['noises_' + str(i + 1) for i in
                                                                                  range(p - q)])
    response = pd.DataFrame(y)
    return data, response


import numpy as np
import pandas as pd


def generate_independent_data(n, p, q):
    # generate data matrix for informative features
    X_rel = np.random.normal(0, 1, (n, q))
    z = np.zeros(n)

    for i in range(q):
        f_i = np.sin  # alter: lambda x: x**2
        z += f_i(X_rel[:, i])

    y = (z > np.mean(z)).astype(int)

    # generate data matrix for noisy features
    X_irr = np.random.normal(0, 1, (n, p - q))

    # concat two parts
    X = np.hstack((X_rel, X_irr))

    # column names
    feature_names = [f'feature {i + 1}' for i in range(q)] + [f'noisy {i + 1}' for i in range(p - q)]

    # convert to pandas Dataframe
    df = pd.DataFrame(X, columns=feature_names)
    y = pd.DataFrame(y)

    return df, y


from sklearn.datasets import make_classification

def generate_rf_data(n, p, q):

    X, y = make_classification(n_samples=n,
                               n_features=p, n_informative=q, n_redundant=0,
                               random_state=42)

    # column names
    feature_names = [f'feature {i + 1}' for i in range(q)] + [f'noisy {i + 1}' for i in range(p - q)]

    # convert to pandas Dataframe
    df = pd.DataFrame(X, columns=feature_names)
    y = pd.DataFrame(y)

    return df, y

def distance_from_center(x, center):
    return np.linalg.norm(x - center)


def calculate_probability(r):
    if r < 47:
        return 1
    elif 47 <= r < 65:
        return (65-r) / 20
    else:
        return 0


def add_non_informative_features(x, n_non_informative):
    non_informative_features = np.random.uniform(0, 100, size=(x.shape[0], n_non_informative))
    return np.hstack((x, non_informative_features))

'''
def generate_simulation_data(n_samples=100, n_informative_features=4,
                             non_informative_features_list=[50, 100, 200, 500], seed=42):

    np.random.seed(seed)

    center = np.array([50, 50, 50, 50])

    # Generate the base data with 4 informative features
    x_base = np.random.uniform(0, 100, size=(n_samples, n_informative_features))

    # Calculate distances
    distances = np.apply_along_axis(distance_from_center, 1, x_base, center)

    # Calculate probabilities
    probabilities = np.array([calculate_probability(r) for r in distances])

    # Generate responses
    y = np.random.binomial(1, probabilities, n_samples)

    # Generate datasets with different number of non-informative features
    datasets = {}
    for n_non_informative in non_informative_features_list:
        x_extended = add_non_informative_features(x_base, n_non_informative)
        dataset = pd.DataFrame(x_extended, columns=[f'feature_{i}' for i in range(n_informative_features)]+
                                                   [f'noisy_{i}' for i in range(x_extended.shape[1]-n_informative_features)])
        dataset['y'] = y
        datasets[n_non_informative] = dataset

    return datasets'''

'''
# Example usage
datasets = generate_simulation_data()

# Access the dataset with 100 non-informative features
dataset_100_non_informative = datasets[100]
print(dataset_100_non_informative.head())

# Save the datasets to CSV files if needed
for n_non_informative, dataset in datasets.items():
    dataset.to_csv(f'dataset_{n_non_informative}_non_informative_features.csv', index=False)
'''