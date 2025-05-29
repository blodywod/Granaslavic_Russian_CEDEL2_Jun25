from b_descriptive import *
from scipy import stats
import pandas as pd
from scipy.stats import mannwhitneyu
import pingouin as pg


def normality_test(df, target_L1):
    print(f"\n{'=' * 80}")
    print(f"\nNORMALITY TESTS FOR L1: {target_L1}")
    numerical_df = df.select_dtypes(include=[np.number])
    normality_tests = pd.DataFrame({
        'shapiro_p': [stats.shapiro(numerical_df[col])[1] for col in numerical_df],
        'jarque_bera_p': [stats.jarque_bera(numerical_df[col])[1] for col in numerical_df]
    }, index=numerical_df.columns)
    print(normality_tests)


def compare_mannwhitney_effectsize(df_russian, df_non_russian, variable):
    # Extract and clean data
    rus_data = df_russian[variable].dropna()
    non_rus_data = df_non_russian[variable].dropna()

    # Perform Mann-Whitney test with Pingouin
    res = pg.mwu(rus_data, non_rus_data)

    # Perform Mann-Whitney test
    stat, p = mannwhitneyu(rus_data, non_rus_data, alternative='two-sided')

    # Calculate Rosenthal's r effect size (r)
    # N = len(rus_data) + len(non_rus_data)
    # Z = (stat - (len(rus_data) * len(non_rus_data) / 2)) / np.sqrt((len(rus_data) * len(non_rus_data) * (N + 1)) / 12)
    # r = abs(Z) / np.sqrt(N)  # Absolute value for magnitude

    # Prepare results
    results = {
        'variable': variable,
        'U': stat,
        'p_value': p,
        'effect_size': {
            'RBC': res['RBC'].values[0],  # Rank-Biserial Correlation (equivalent to Rosenthal's r)
            'CLES': res['CLES'].values[0],  # Common Language Effect Size
        },
        'russian_stats': {
            'n': len(rus_data),
            'median': rus_data.median(),
            'IQR': rus_data.quantile(0.75) - rus_data.quantile(0.25)
        },
        'non_russian_stats': {
            'n': len(non_rus_data),
            'median': non_rus_data.median(),
            'IQR': non_rus_data.quantile(0.75) - non_rus_data.quantile(0.25)
        }
    }

    return results


def print_results_tests(results):
    """Print Mann-Whitney test results in a readable format"""
    if results is None:
        return

    print(f"\n{'═' * 50}")
    print(f"COMPARISON: {results['variable']}")
    print(f"Russian (n={results['russian_stats']['n']}) vs Non-Russian (n={results['non_russian_stats']['n']})")

    print("\nDESCRIPTIVE STATISTICS:")
    print(
        f"Russian Median: {results['russian_stats']['median']:.2f} (IQR: {results['russian_stats']['IQR']:.2f})")
    print(
        f"Non-Russian Median: {results['non_russian_stats']['median']:.2f} (IQR: {results['non_russian_stats']['IQR']:.2f})")

    print("\nMANN-WHITNEY TEST RESULTS:")
    print(f"U = {results['U']:.1f}, p = {results['p_value']:.4f}")
    # print(f"Effect size (r) = {results['effect_size']:.3f}")
    print(f"Effect size (RBC) = {results['effect_size']['RBC']:.3f}")
    print(f"Probability Group A > Group B (CLES) = {results['effect_size']['CLES']:.3f}")

    print("\nINTERPRETATION:")
    significance = "SIGNIFICANT" if results['p_value'] < 0.05 else "not significant"
    print(f"Differences are {significance} (p < 0.05)")
    print("═" * 50)


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
    # INFERENTIAL ANALYSIS
    normality_test(ruso_data, 'Russian')
    normality_test(data_without_russian, 'all_L1')
    results_placement = compare_mannwhitney_effectsize(ruso_data, data_without_russian, 'Placement test score (%)')
    results_age = compare_mannwhitney_effectsize(ruso_data, data_without_russian, 'Age')
    results_selfassess = compare_mannwhitney_effectsize(ruso_data, data_without_russian, 'Proficiency (self-assessment)')
    print_results_tests(results_placement)
    print_results_tests(results_age)
    print_results_tests(results_selfassess)
