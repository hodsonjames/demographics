# Gender Classification Algorithm

This algorithm attempts to classify gender (M/F) from employment profiles using the given first name and country data. Created by Kunal Adhia. Advisors: James Hodson, Anastassia Fedyk.

# Instructions
To run, please ensure that all necessary libraries (numpy, pandas, sklearn, xpinyn) are installed. Python 3 was used for this project. Simply clone this repository and run "python3 gender_classifier.py".

# Approach

When performing introductory searches and analyses on the input data set, available data sets online, and the capabilities of simple classifier models, the following observations were made:

- The input data, stored in "gender-sample.jsonl", is very unclean. Some of the noticeable outliers (and there are many) include:
	- Names that begin with a prefix (such as CA, Mr, and Dr).
	- Names with anglicized versions in parentheses.
	- Names in different languages (such as Russian and Chinese).
	- Names with only the first initial (such as "V. Ross")
	- Names with special characters
	- Incomplete or null first names (such as "ee" and "-")

- Five of the six provided reference data sets, as well as the outside data set, do not contain breakdowns of name distributions by country.

Due to the high frequency and variance of outliers, a more general approach was taken to clean the input data and extract the first name. The following approach was taken:
- Common prefixes were removed.
 - If available, anglicized names were used instead of birth names, as these provided a higher likelihood of exact matches against reference data sets.
- Names in other languages (especially in Chinese, which had the highest frequency) were translated using libraries such as xpinyin.
- Blank names were ignored and a blind guess was made by the model.
- For the classification model, all special characters were replaced by a ? using a regex substitution.
- The first name was considered to be anything before the first space after prefix removal. 

This approach, although not perfect, captured most of the sample data and almost all outliers.

Nevertheless, these issues still make it very difficult to construct an accurate model or exact-match every name to the available data. Therefore, in order to maximize total accuracy, two separate approaches were implemented:
 - First, an exact match algorithm was performed against six different  reference datasets. These datasets provided different types of information, but the main focus was the name, classified gender, country information (if provided), and confidence. If conflicting results arose, priority was given based on the below order of datasets. The six datasets are as follows:
	- The U.S. SSN datasets from 1880 to 2019 (the algorithm prioritized recent data)
	- The most common male/female names in the US in 2019 (tiebreakers by rank)
	- Classified names with frequency by country (author: astro.joerg@googlemail.com)
	- A much more comprehensive dataset of names (data.world author: howarder)
	- A public dataset of primarily Indian and Middle Eastern names
	- A public dataset of primarily East Asian names
- For the names which were unable to be exact-matched, two models were considered: a model using both the first name and country data (only dataset 3 was used for training/testing), and a model where only the first name was used (all six datasets were used). Due to the lack of sufficient training data, the latter option was used for classification. A Ridge Classfier was used with an 80/20 train-test split and automated hyperparemeter selection.
