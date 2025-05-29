from xml.dom.minidom import Document
from a_preprocessing import *
import numpy as np
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def print_descriptive_statistics_catvar(df, target_L1):
    # General summary for categorical columns
    print(f"\n{'=' * 80}")
    print(f"\nDESCRIPTIVE STATISTICS FOR L1: {target_L1}")
    print("\nCATEGORICAL VARIABLES:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())


def print_descriptive_statistics_numvar(df, target_L1):
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    # General summary for numerical columns
    print("\nNUMERICAL VARIABLES")
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    # Create comprehensive statistics table
    stats_table = numerical_df.describe(percentiles=[.25, .5, .75]).transpose()
    stats_table['IQR'] = stats_table['75%'] - stats_table['25%']
    print(df.describe())
    print(stats_table)


if __name__ == '__main__':
    target_file_directory = 'C:\\Users\\ASUS\\Documents\\PYTHON_CODE\\Congreso_Granaslavic_June25\\sources\\texts.csv'
    target_variable_names = [
        'Placement test score (%)', 'Proficiency', 'Sex', 'Age', 'L1',
        'Stay abroad in Spanish speaking country (>= 1 month)',
        'Proficiency (self-assessment) speaking', 'Proficiency (self-assessment) listening',
        'Proficiency (self-assessment) reading', 'Proficiency (self-assessment) writing',
        'Proficiency (self-assessment)', 'Medium', 'Task number', 'Writting/audio details', 'Where the task was done',
    ]
    target_variables_mapping = {
        'Proficiency (self-assessment)': {'1 / 6': 1, '1.25 / 6': 1.25, '1.5 / 6': 1.5, '1.75 / 6': 1.75,
                                          '2 / 6': 2, '2.25 / 6': 2.25, '2.5 / 6': 2.5, '2.75 / 6': 2.75,
                                          '3 / 6': 3, '3.25 / 6': 3.25, '3.5 / 6': 3.5, '3.75 / 6': 3.75,
                                          '4 / 6': 4, '4.25 / 6': 4.25, '4.5 / 6': 4.5, '4.75 / 6': 4.75,
                                          '5 / 6': 5, '5.25 / 6': 5.25, '5.5 / 6': 5.5, '5.75 / 6': 5.75,
                                          '6 / 6': 6}
    }
    # LOAD CEDEL2 CSV FILE TO DATAFRAME
    dataframe = load_file_CEDEL2_as_dataframe(target_file_directory)
    # SELECT TARGET VARIABLES FROM CEDEL2 DATAFRAME
    selected_vars_dataframe = select_dataframe_variables(dataframe, target_variable_names)
    # DROP NANS FROM DATAFRAME
    cleaned_dataframe = clean_variables_lost_invalid_cases(selected_vars_dataframe)
    # SAVE CLEANED DATAFRAME
    save_cleaned_dataframe(
        cleaned_dataframe,
        'C:\\Users\\ASUS\\Documents\\PYTHON_CODE\\Congreso_Granaslavic_June25\\output\\cleaned_data.csv'
    )
    # ENCODE STRING VARIABLES FROM DATAFRAME
    encoded_vars_dataframe = encode_var_string_to_numeric(cleaned_dataframe, target_variables_mapping)
    # SAVE ENCODED DATAFRAME
    save_encoded_dataframe(encoded_vars_dataframe, '..\\output\\encoded_data.csv')
    # NORMALIZE NUMERIC VARIABLES FROM DATAFRAME
    norm_vars = normalize_numeric_vars(encoded_vars_dataframe, ['Proficiency (self-assessment)'], [(0, 100)])
    numeric_vars = ['Placement test score (%)', 'Proficiency (self-assessment)', 'Age']
    # FILTER AND SAVE L1 RUSSIAN DATA
    ruso_data = filter_by_L1(encoded_vars_dataframe, 'Russian')
    save_cleaned_dataframe(ruso_data, 'C:\\Users\\ASUS\\Documents\\PYTHON_CODE\\Congreso_Granaslavic_June25\\output\\ruso_data.csv')
    # FILTER AND SAVE L1 NON-RUSSIAN DATA
    data_without_russian = exclude_by_L1(encoded_vars_dataframe, 'Russian')
    data_without_russian.to_csv('..\\output\\data_without_russian.csv', index=False)
    # PRINT DESCRIPTIVE ANALYSIS RESULTS
    print(print_descriptive_statistics_catvar(ruso_data, 'Russian'))
    print(print_descriptive_statistics_numvar(ruso_data, 'Russian'))
    print(print_descriptive_statistics_catvar(data_without_russian, 'all_L1'))
    print(print_descriptive_statistics_numvar(data_without_russian, 'all_l1'))
