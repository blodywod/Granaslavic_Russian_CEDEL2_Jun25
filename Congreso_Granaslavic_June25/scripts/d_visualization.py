from c_inference import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm


def barplot_relative_frequencies_cat(df):
    for var in df:
        plt.figure(figsize=(8, 6))
        relative_frq = df[var].value_counts(normalize=True) * 100

        sns.barplot(x=[str(i) for i in relative_frq.index], y=relative_frq.values, color='grey', edgecolor='None')
        plt.title(f'Relative frequencies of \"{var}\"')

        for i, value in enumerate(relative_frq.values):
            plt.text(i, value + 0.5, f'{value:.2f}%', ha='center', va='bottom')
        sns.despine()
        plt.show()


def barplot_relative_frequencies_cat_sorted(var, labels):
    plt.figure(figsize=(8, 6))
    relative_frq = var.value_counts(normalize=True) * 100

    relative_frq = relative_frq.reindex(labels)

    sns.barplot(x=relative_frq.index, y=relative_frq.values, color='grey', edgecolor='None')
    plt.title(f'Relative frequencies of \"{var}\"')

    for i, value in enumerate(relative_frq.values):
        plt.text(i, value + 0.5, f'{value:.2f}%', ha='center', va='bottom')
    sns.despine()
    plt.show()


def barplot_relative_frq_num_with_norm(df, var, minimo=13, maximo=89, step=5):
    # Generate some data for this demonstration.
    data = df[var]
    bins = np.arange(minimo, maximo, step)

    # Fit a normal distribution to the data: mean and standard deviation
    mu, std = norm.fit(data)

    # Plot the histogram with relative frequencies.
    counts, bins, _ = plt.hist(data, bins=bins, density=False, alpha=0.6, color='#3A3A3A', rwidth=0.9)

    # Calculate relative frequencies.
    relative_frq = counts / sum(counts) * 100

    # Plot the PDF (Probability Density Function).
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) * sum(counts) * (bins[1] - bins[0])  # Scale PDF to histogram

    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)

    # Print relative frequencies.
    print("Relative Frequencies:")
    for bin_start, bin_end, frq in zip(bins[:-1], bins[1:], relative_frq):
        print(f"{bin_start:.2f} - {bin_end:.2f}: {frq:.2f}%")

    # Set y-axis label and adjust ticks to show percentages
    plt.xlabel(f'Relative frequencies of \"{var}\"')
    plt.ylabel('Relative Frequency (%)')
    plt.xticks(bins)
    plt.gca().set_yticklabels(['{:.0f}%'.format(100 * x/sum(counts)) for x in plt.gca().get_yticks()])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


def boxplot_comparison(data1, data2, label1='Group 1', label2='Group 2', title='', xlabel='Group', ylabel='Value'):
    # Create combined DataFrame
    combined = pd.concat([
        pd.DataFrame({xlabel: label1, ylabel: data1}),
        pd.DataFrame({xlabel: label2, ylabel: data2})
    ])
    # Create boxplot with no fill
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=xlabel, y=ylabel, data=combined,
                color='white', showmeans=True,
                meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'),
                boxprops=dict(facecolor='white', edgecolor='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='black')
                )
    # Reduce jitter and point size (jitter=less horizontal spread, alpha=points semitransparent)
    sns.stripplot(x=xlabel, y=ylabel, data=combined, color='gray', alpha=0.3, jitter=0.2, size=3)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


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
    # # PLOTTING STUFFS FOR L1_RUSSIAN
    # barplot_relative_frequencies_cat(ruso_data)
    # barplot_relative_frequencies_cat_sorted(
    #     ruso_data['Proficiency'],
    #     ['Lower beginner', 'Upper beginner', 'Lower intermediate', 'Upper intermediate', 'Lower advanced',
    #      'Upper advanced']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     ruso_data['Proficiency (self-assessment) speaking'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     ruso_data['Proficiency (self-assessment) listening'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     ruso_data['Proficiency (self-assessment) reading'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     ruso_data['Proficiency (self-assessment) writing'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frq_num_with_norm(ruso_data, 'Age')
    # barplot_relative_frq_num_with_norm(ruso_data, 'Placement test score (%)', 0, 110, 10)
    # barplot_relative_frq_num_with_norm(ruso_data, 'Proficiency (self-assessment)', 0, 110, 10)
    #
    # # PLOTTING STUFFS FOR DATA WITHOUT RUSSIAN
    # print('PLOTTING STUFFS FOR DATA WITHOUT RUSSIAN')
    # barplot_relative_frequencies_cat(data_without_russian)
    # barplot_relative_frequencies_cat_sorted(
    #     data_without_russian['Proficiency'],
    #     ['Lower beginner', 'Upper beginner', 'Lower intermediate', 'Upper intermediate', 'Lower advanced',
    #      'Upper advanced']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     data_without_russian['Proficiency (self-assessment) speaking'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     data_without_russian['Proficiency (self-assessment) listening'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     data_without_russian['Proficiency (self-assessment) reading'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frequencies_cat_sorted(
    #     data_without_russian['Proficiency (self-assessment) writing'],
    #     ['Lower beginner (A1)', 'Upper beginner (A2)', 'Lower intermediate (B1)', 'Upper intermediate (B2)',
    #      'Lower advanced (C1)', 'Upper advanced (C2)']
    # )
    # barplot_relative_frq_num_with_norm(data_without_russian, 'Age')
    # barplot_relative_frq_num_with_norm(data_without_russian, 'Placement test score (%)', 0, 110, 10)
    # barplot_relative_frq_num_with_norm(data_without_russian, 'Proficiency (self-assessment)', 0, 110, 10)

    # PLOTTING FOR RUSSIAN vs NON-RUSSIAN COMPARISON
    boxplot_comparison(ruso_data['Placement test score (%)'], data_without_russian['Placement test score (%)'],
                       label1='Russian L1 learners', label2='non-Russian L1 learners', title='Distribution of Placement test score (%) across L1 groups', ylabel='Score')
