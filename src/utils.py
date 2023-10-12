import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder


def find_missing_values(dataframe: DataFrame) -> DataFrame:
    """
    Find and report missing values in the DataFrame.

    Parameters:
    - dataframe: DataFrame
        The input DataFrame to analyze for missing values.

    Returns:
    - DataFrame or str
        If there are missing values, returns a DataFrame containing the total count
        and percentage of missing values for each column. If there are no missing
        values, returns a message indicating that there are no missing values.
    
    """
    total_missing: DataFrame = dataframe.isnull().sum()
    
    if total_missing.sum() == 0:
        return "There are no missing values in the DataFrame."
    
    percent_missing: DataFrame = (total_missing / len(dataframe)) * 100
    missing_values_df: DataFrame = pd.concat([total_missing, percent_missing], axis=1, keys=["Total", "Percentage"])
    missing_values_df = missing_values_df[missing_values_df["Total"] > 0]  # Filter out columns with no missing values
    
    return missing_values_df


def drop_columns_with_high_missing_values(dataframe: DataFrame, threshold=0.7) -> DataFrame:
    """
    Drop columns with missing values exceeding the specified threshold.
    
    Parameters:
    - dataframe: DataFrame
        The input DataFrame.
    - threshold: float (default: 0.7)
        The threshold for the percentage of missing values to drop a column. Columns with
        missing values exceeding this threshold will be dropped.

    Returns:
    - DataFrame
        The DataFrame with columns containing more than the threshold percentage of missing values removed.
    """
    total_missing: DataFrame = dataframe.isnull().sum()
    percent_missing: DataFrame = (total_missing / len(dataframe)) * 100
    columns_to_drop = percent_missing[percent_missing > threshold].index
    
    dataframe = dataframe.drop(columns=columns_to_drop)
    return dataframe


def separate_categorical_numerical(dataframe: DataFrame) -> tuple:
    """
    Separate categorical and numerical variables in the DataFrame.

    Parameters:
    - dataframe: DataFrame
        The input DataFrame to be separated.

    Returns:
    - tuple
        A tuple containing two DataFrames:
        - The first DataFrame contains categorical variables.
        - The second DataFrame contains numerical variables.
    """
    # Separate columns into categorical and numerical
    categorical_columns = dataframe.select_dtypes(include=['object', 'category'])
    numerical_columns = dataframe.select_dtypes(exclude=['object', 'category'])

    return categorical_columns, numerical_columns

def label_encode_categorical_variables(dataframe: DataFrame, categorical_columns: list) -> DataFrame:
    """
    Label encode categorical variables in a DataFrame.

    Parameters:
    - dataframe: DataFrame
        The input DataFrame.
    - categorical_columns: list
        A list of column names that contain categorical variables to be label encoded.

    Returns:
    - DataFrame
        A new DataFrame with categorical variables label encoded.
    """
    # Create a copy of the input DataFrame to avoid modifying the original data
    encoded_dataframe = dataframe.copy()
    
    label_encoder = LabelEncoder()

    for column in categorical_columns:
        if column in encoded_dataframe.columns:
            encoded_dataframe[column] = label_encoder.fit_transform(encoded_dataframe[column])

    return encoded_dataframe


def drop_highly_correlated_variables(dataframe: DataFrame, threshold: float = 0.7) -> DataFrame:
    """
    Drop highly correlated variables from a DataFrame based on a specified threshold.

    Parameters:
    - dataframe: DataFrame
        The input DataFrame containing numerical variables.
    - threshold: float (default: 0.7)
        The correlation threshold above which variables are considered highly correlated and dropped.

    Returns:
    - DataFrame
        A new DataFrame with highly correlated variables removed.
    """
    corr_matrix = dataframe.corr().abs()

    mask = corr_matrix >= threshold

    # Find column pairs with high correlation
    high_corr_vars = set()
    for col in mask.columns:
        correlated_columns = mask.columns[mask[col]]
        for col2 in correlated_columns:
            if col != col2:
                high_corr_vars.add(col2)

    # Drop highly correlated variables
    reduced_dataframe = dataframe.drop(columns=high_corr_vars)
    
    return reduced_dataframe