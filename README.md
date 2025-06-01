# Who's there? Uncovering learners behind data
## CEDEL2 Learner Corpus Analysis – L1 Russian Learners of L2 Spanish
### Granaslavic Congress 2025
This repository contains the code and analysis for the study titled "Who's There? Uncovering Learners Behind Data", presented at the Granaslavic Congress 2025. The research focuses on examining the learner profiles, Spanish proficiency, and written productions of L1 Russian learners of L2 Spanish within the CEDEL2 corpus. The study combines statistical analysis with learner corpus research (LCR). 
  
## Corpus: CEDEL2 v2.1 (accessed 2024-04-23)
The Corpus Escrito del Español como L2 (L2 Spanish Written Corpus, CEDEL2) is a large-scale, multi-L1 learner corpus of Spanish as a foreign language. It includes spoken and written compositions from learners at various proficiency levels and native control subcorpora. The corpus contains 4,334 compositions, 25 variables and 16 subcorpora—11 learner subcorpora and 5 native control subcorpora. 
#### CEDEL2 learner corpus
The learner portion of the CEDEL2 corpus includes 3,034 files with 41 variables. After preprocessing (e.g., removing missing values, invalid cases and irrelevant variables), we retained 3,022 files with 15 variables for analysis.
#### Russian subset (post-preprocessing)
- 101 learners (3.3% of the total CEDEL2 corpus)
- Data collected in 2020
- Compared against 2,921 learners of other ten L1s

## Objective
To describe and analyze the learner profiles, Spanish proficiency and written productions of Russian L1 learners in the CEDEL2 corpus, examining how their linguistic background relates to text length and performance, and exploring differences with non-Russian learners through descriptive and inferential statistics.

## Methodology
Data Preprocessing: Selection and preparation of data from L1 Russian learners and relevant variables.
Text Processing: Tokenization, removal of punctuation and stopwords, and computation of token and word counts.
Descriptive Statistics: Analysis of token/word counts, frequency distributions, and measures such as mean, median, IQR, and range.
Normality Testing: Shapiro-Wilk test to assess distribution normality.
Inferential Statistics: Mann–Whitney U test to compare Russian vs. non-Russian learners, along with effect size metrics RBC (Rank-Biserial Correlation) and CLES (Common Language Effect Size)

## Repository Structure
- sources/: Original CEDEL2 learner corpus file.
- output/: Processed subsets of the CEDEL2 corpus used in the analysis.
- scripts/: Python scripts for data processing, statistical analysis, and visualization.

## CONTACT
- Author: Thuy Huong Nguyen; 
Affiliation: Universidad de Granada;
Email: huong.traductora@gmail.com
- Co-author: Dr.Benamí Barros García;
Affiliation: Universidad de Granada;
Email: bbarros@ugr.es
