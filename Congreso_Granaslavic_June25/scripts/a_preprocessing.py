import pandas as pd
from sklearn.preprocessing import minmax_scale


def load_file_CEDEL2_as_dataframe(str_file_directory, enc='utf8', dlm='\t'):
    print('Loading CEDEL2 file')
    df = pd.read_csv(str_file_directory, encoding=enc, delimiter=dlm)
    print('CEDEL2 file loaded')
    print(df.shape)
    print(df.head(5))
    return df


def select_dataframe_variables(data_frame, list_of_variable_names):
    print('DataFrame variables selected')
    return data_frame[list_of_variable_names]


def clean_variables_lost_invalid_cases(df_selected_var):
    print('Cleaning nans from DataFrame')
    df_clean = df_selected_var.dropna()
    result_df = df_clean.reset_index(drop=True)
    print('DataFrame cleaned')
    print("Number of cases:", len(result_df))
    return result_df


def save_cleaned_dataframe(df, output_path):
    print('Saving cleaned DataFrame')
    df.to_csv(output_path, index=False)
    print('Clean DataFrame saved')
    return True


def encode_var_string_to_numeric(df_encode, mappings):
    print('Encoding string variables')
    for col, mapping in mappings.items():
        df_encode[col] = df_encode[col].map(mapping)
    print('String variables encoded')
    return df_encode


def save_encoded_dataframe(df, output_path):
    print('Saving encoded DataFrame')
    df.to_csv(output_path, index=False)
    print('Encoded DataFrame saved')
    return True


def normalize_numeric_vars(in_dataframe, listofvars, listofscales):
    print('Normalizing numeric variables')
    for var, scale in zip(listofvars, listofscales):
        in_dataframe[var] = minmax_scale(
            in_dataframe[var],
            feature_range=scale,
            axis=0,
            copy=True
        )
    print('Numeric variables normalized')
    print(in_dataframe.head(5))
    return in_dataframe


def filter_by_L1(df, target_L1):
    print(f'Filtering data for L1: {target_L1}')
    filtered_df = df[df['L1'] == target_L1].reset_index(drop=True)
    print(f'Filtered DataFrame has {len(filtered_df)} rows')
    return filtered_df


def exclude_by_L1(df, l1_value):
    print(f'Filtering data for other L1s excluding: {l1_value}')
    excluded_df = df[df['L1'] != l1_value]
    print(f'Excluded DataFrame has {len(excluded_df)} rows')
    return excluded_df


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
