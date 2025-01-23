# 👋🏼 Introduction
Web-scraped datasets are often messy and require thorough cleaning before they can be analyzed. Starting with an unclean dataset that was web-scraped from Glassdoor's website, I worked on exploring and cleaning it in order to transform it into a tidy, well-structured dataset ready for further analysis. The dataset contains information on data science job postings.

# 🙋🏻‍♀️ Questions
Throughout the cleaning process, I asked these questions:
1. What are the basic cleaning steps I need to do to make this dataset more structured and understandable?
2. What information can I extract out of the existing columns that will create more useful columns for analysis?
3. How will the cleaned dataset possibly be used for future analysis?

# 🔧 Tools I Used
For this project, I used these key tools:

- Python: This allowed me to clean the dataset, mainly using these libraries:
    - Pandas Library: For exploring and cleaning data
    - NLTK Library: For tokenization
- Visual Studio Code: This made executing Python scripts more convenient and efficient.
- Git & Github: These allowed me to share my Python scripts and analysis, and made sure versions were kept up to date.

# 🧽 The Process
### Messy Columns
After exploring the data, I started by tidying up columns that were clearly in messy formats. The Salary Estimate column wasn't in a number format and contained unnecessary characters. I cleaned up this column and created new ones for minimum, maximum, and average salary.
```python
# Treating Salary Estimate column

# Remove "Glassdoor est." and other unwanted characters
df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'\(Glassdoor est.\)', '', regex=True)
df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'[^\d\-]', '', regex=True)

# Split the range into two parts
df[['Salary_Min', 'Salary_Max']] = df['Salary Estimate'].str.split('-', expand=True)

# Convert to integers
df['Salary_Min'] = pd.to_numeric(df['Salary_Min'], errors='coerce')
df['Salary_Max'] = pd.to_numeric(df['Salary_Max'], errors='coerce')

# Calculate the average salary
df['Salary_Average'] = ((df['Salary_Min'] + df['Salary_Max']) / 2).astype(int)
```

I also needed to clean up the Company Name column, which contained company ratings as well.
```python
df['Company_Name'] = df['Company Name'].str.replace(r'\n.*', '', regex=True)
```

Lastly, I cleaned up the job description column and created a cleaned version for possible future keyword analysis.
```python
# Treating Job Description column

nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Remove unnecessary symbols and whitespace
df['Job_Description_Cleaned'] = df['Job Description'].str.replace(r'\n', ' ', regex=True)  # Remove newlines
df['Job_Description_Cleaned'] = df['Job_Description_Cleaned'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
df['Job_Description_Cleaned'] = df['Job_Description_Cleaned'].str.strip()  # Remove leading/trailing spaces

# Step 2: Convert to lowercase
df['Job_Description_Cleaned'] = df['Job_Description_Cleaned'].str.lower()

# Step 3: Remove stop words
stop_words = set(stopwords.words('english'))
extras = {"description"}
all_stop_words = stop_words.union(extras)
def remove_stop_words(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in all_stop_words]
    return ' '.join(cleaned_tokens)

df['Job_Description_Cleaned'] = df['Job_Description_Cleaned'].apply(remove_stop_words)
```

### Extracting Information
I wanted to create a new column that was True if the job location and company headquarters were in the same state/country, and False if not.
```python
# Checking if location and headquarters are in the same state/country

# Extract the state/country from location and headquarters
df['Location_State'] = df['Location'].str.split(',').str[1].str.strip()
df['Headquarters_State'] = df['Headquarters'].str.split(',').str[1].str.strip()

# Handle missing or invalid headquarters values
df['Headquarters_State'] = df['Headquarters_State'].replace("-1", None)

# Compare the states/countries and create 'Same_State' column
df['Same_State'] = df['Location_State'] == df['Headquarters_State']

# Check the results
df[['Location', 'Headquarters', 'Location_State', 'Headquarters_State', 'Same_State']]
```

I used the Founded column to calculate the company age, as this would be more helpful for analysis.
```python
# Calculating company age

from datetime import datetime

current_year = datetime.now().year

# Replace invalid values (-1) with NaN
df['Founded'] = df['Founded'].replace(-1, pd.NA)

# Calculate company age only for non-missing values
df['Company_Age'] = df['Founded'].apply(lambda x: current_year - x if pd.notna(x) else pd.NA)

# Check results
df['Company_Age']
```

I also wanted to create skill columns that would have a value of 0 or 1 if it was mentioned/not in the job description.
```python
# Checking for skills mentioned in cleaned job descrip

df['python'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['excel'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df['hadoop'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)
df['spark'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['aws'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['tableau'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
df['big_data'] = df['Job_Description_Cleaned'].apply(lambda x: 1 if 'big data' in x.lower() else 0)
```

Lastly, I created a column for the simplified version of the job titles, just so there would be an easier way of checking job types.
```python
# Creating simple job titles from long ones

def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
df['Job_Simp'] = df['Job Title'].apply(title_simplifier)

df.Job_Simp.value_counts()
```
### Refining the Dataset
Finally, with the cleaned up data and new columns, I had a more useful dataset on hand. I just needed to drop unnecessary columns and reorder the columns in a more structured format.
```python
# Dropping unnecessary columns
df.drop(['index', 'Company Name', 'Founded', 'Competitors', 'Location_State', 'Headquarters_State'], axis=1,inplace=True)
df.head()

# Reordering columns

# Define new column order
column_order = ['Job Title', 'Job_Simp', 'Salary Estimate', 'Salary_Min', 'Salary_Max', 'Salary_Average',
                'Company_Name', 'Job Description', 'Rating', 'Location', 'Headquarters', 'Same_State', 
                'Size', 'Company_Age', 'Type of ownership', 'Industry', 'Sector', 'Revenue',
                'python', 'excel', 'hadoop', 'spark', 'aws', 'tableau', 'big_data', 'Job_Description_Cleaned']

# Reorder the columns
df = df[column_order]

# Display results
df.head()
```

# ✅ Results
After going through the entire cleaning process, I had a new dataset that was cleaned and structured, ready for meaningful analysis. I saved this new dataset in case I wanted to use it to analyze data science job postings on Glassdoor.
```python
df.to_csv('cleaned_DS_jobs.csv', index=False)
```
You can check out the resulting dataset [here](cleaned_DS_jobs.csv).

# 📝 Conclusions
Working on this project made me realize a few things:
- **Web-scraped data in its raw form is usually messy to begin with.** This is one of the cons of working with web-scraped data. You have to go through a rigorous cleaning process before you can actually start to analyze it.
- **Data cleaning is not just about fixing errors but about making the data reliable and structured for analysis.** The initial data cleaning steps usually just target simple errors and inconsistencies. But you have to look beyond that, and explore if the data can be reliable even after doing some initial cleaning.
- **Cleaning and preparing a dataset effectively requires a clear understanding of its intended use.** It's important to understand what the dataset will be used for. In this case, the Glassdoor job postings dataset would likely be used by someone looking for a new job or helping someone find one. I kept that in mind while making sure the dataset had the most essential information for this type of analysis.

## Closing Thoughts
As a data analyst, I know that data cleaning is a crucial and time-consuming step in the analysis process. Unreliable and messy data could lead to an entirely new set of problems. Focusing on data cleaning in this project allowed me to practice essential skills that a data analyst should have. Although, I still always hold out hope that the datasets I'll deal with in my work are already as clean as they possibly can be. 😄