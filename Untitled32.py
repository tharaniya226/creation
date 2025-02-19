#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

# Load dataset
def load_data(file_path):
    df = pd.read_csv('students.csv.csv')  # Assuming CSV format, modify for Excel
    return df

# Identify strengths and weaknesses using K-Means clustering
def analyze_performance(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Performance Cluster'] = kmeans.fit_predict(df[['Mark']])
    performance_labels = {0: 'Poor', 1: 'Medium', 2: 'High'}
    df['Performance Level'] = df['Performance Cluster'].map(performance_labels)
    return df

# Recommend To-Do Courses using Decision Tree
def recommend_courses(df):
    courses = {
        'Poor': ['Basic Programming', 'Logic Building'],
        'Medium': ['Data Structures', 'Algorithms'],
        'High': ['Advanced ML', 'Competitive Programming']
    }
    df['Recommended Courses'] = df['Performance Level'].map(lambda x: courses[x])
    return df

# Identify poor performers and suggest mentors
def pair_students(df):
    poor = df[df['Performance Level'] == 'Poor']
    high = df[df['Performance Level'] == 'High']
    pairs = list(zip(poor['Candidate Name'], np.random.choice(high['Candidate Name'], len(poor), replace=True)))
    return pairs

# Flask App Setup
app = Flask(__name__)

data_file = 'students.csv.csv'
df = load_data(data_file)
df = analyze_performance(df)
df = recommend_courses(df)
mentorship_pairs = pair_students(df)

@app.route('/')
def dashboard():
    return render_template('dashboard.html', tables=[df.to_html(classes='data')], mentorship=mentorship_pairs)

if __name__ == '__main__':
    app.run(debug=True)


# In[3]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

# Generate synthetic dataset
def generate_synthetic_data():
    np.random.seed(42)
    data = {
        'Candidate Name': [f'Student_{i}' for i in range(1, 101)],
        'Mark': np.random.randint(30, 100, 100)
    }
    df = pd.DataFrame(data)
    return df

# Identify strengths and weaknesses using K-Means clustering
def analyze_performance(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Performance Cluster'] = kmeans.fit_predict(df[['Mark']])
    performance_labels = {0: 'Poor', 1: 'Medium', 2: 'High'}
    df['Performance Level'] = df['Performance Cluster'].map(performance_labels)
    return df

# Recommend To-Do Courses using Decision Tree
def recommend_courses(df):
    courses = {
        'Poor': ['Basic Programming', 'Logic Building'],
        'Medium': ['Data Structures', 'Algorithms'],
        'High': ['Advanced ML', 'Competitive Programming']
    }
    df['Recommended Courses'] = df['Performance Level'].map(lambda x: courses[x])
    return df

# Identify poor performers and suggest mentors
def pair_students(df):
    poor = df[df['Performance Level'] == 'Poor']
    high = df[df['Performance Level'] == 'High']
    pairs = list(zip(poor['Candidate Name'], np.random.choice(high['Candidate Name'], len(poor), replace=True)))
    return pairs

# Flask App Setup
app = Flask(__name__)

df = generate_synthetic_data()
df = analyze_performance(df)
df = recommend_courses(df)
mentorship_pairs = pair_students(df)

@app.route('/')
def dashboard():
    return render_template('dashboard.html', tables=[df.to_html(classes='data')], mentorship=mentorship_pairs)

if __name__ == '__main__':
    app.run(debug=True)


# In[4]:


import random
import pandas as pd
import numpy as np

# Load names and email addresses from the namelist
namelist_df = pd.read_excel('section A name list 1 year Ai&Ds.xlsx')  # Adjust path if necessary
print(namelist_df.columns)
 
# Extract the names and emails
names_emails = namelist_df[['Name ', 'MAIL']].values.tolist()

# Define the courses and course IDs
courses = {
    'Engineering Mathematics': 'U20BST215',
    'Computer Programming': 'U20EST243',
    'Data Structure and Applications': 'U20EST245',
    'Object Oriented Programming': 'U20EST247',
    'Computer and Communication Networks': 'U20ADT202'
}

# Grading function based on marks
def get_grade(mark):
    if pd.isna(mark):
        return None
    if mark < 35:
        return 'F'
    elif 35 <= mark < 45:
        return 'E'
    elif 45 <= mark < 55:
        return 'D'
    elif 55 <= mark < 65:
        return 'C'
    elif 65 <= mark < 75:
        return 'B'
    elif 75 <= mark < 85:
        return 'A'
    else:
        return 'S'

# Generate data
data = []

# Number of candidates (using the length of your namelist for the number of records)
num_candidates = len(names_emails)

for i in range(num_candidates):
    candidate_name, candidate_email = names_emails[i]  # Use names and emails from the namelist
    
    # Each candidate attempts all courses
    for course_name, course_id in courses.items():
        # Initialize mark and attempt
        mark = random.randint(25, 97)
        attempt = 1
        
        while attempt <= 3:
            # Add record for current attempt
            grade = get_grade(mark)
            
            data.append({
                'course_name': course_name,
                'course_id': course_id,
                'attempt': attempt,
                'candidate_name': candidate_name,
                'candidate_email': candidate_email,
                'mark': mark,
                'grade': grade
            })
            
            if mark > 70 and mark <= 100:  # If mark > 70%, stop attempts
                break
            else:
                if attempt == 1:
                    # Increase mark by 10-20% for second attempt
                    mark += mark * (random.uniform(0.10, 0.20))
                    mark = np.round(min(mark, 100))  # Ensure mark doesn't go above 100           
                    attempt = 2
                elif attempt == 2:
                    # If 2nd attempt is still ≤ 70%, go for a 3rd attempt
                    mark += mark * (random.uniform(0.10, 0.20))
                    mark = np.round(min(mark, 100))  # Ensure mark doesn't go above 100
                    attempt = 3
                else:
                    # For attempt 3, mark doesn't matter anymore, we're done
                    break

# Convert to DataFrame
df = pd.DataFrame(data)

# Introduce blank cells in 'mark' column randomly
def introduce_blank_marks(df, blank_prob=0.1):
    for i in range(len(df)):
        if random.random() < blank_prob:  # 10% chance to introduce blank mark
            df.at[i, 'mark'] = None
    return df

# Introduce blank cells in 'candidate_name' and 'candidate_email' columns randomly
def introduce_blank_names_emails(df, blank_prob=0.1):
    for i in range(len(df)):
        if random.random() < blank_prob:  # 10% chance to introduce blank name
            df.at[i, 'candidate_name'] = None
        if random.random() < blank_prob:  # 10% chance to introduce blank email
            df.at[i, 'candidate_email'] = None
    return df

# Introduce negative values in the 'mark' column randomly (maximum 20 negative values)
def introduce_negative_marks(df, max_negative=20):
    neg_count = 0
    for i in range(len(df)):
        if neg_count >= max_negative:
            break
        if random.random() < 0.05:  # 5% chance to introduce negative mark
            df.at[i, 'mark'] = -abs(random.randint(1, 30))  # Negative mark between -1 and -30
            neg_count += 1
    return df

# Apply the functions to introduce inconsistencies
df_with_blank_marks = introduce_blank_marks(df)
df_with_blank_marks = introduce_blank_names_emails(df_with_blank_marks)
df_with_blank_marks = introduce_negative_marks(df_with_blank_marks)

# Shuffle the dataset to mix records
df_with_blank_marks = df_with_blank_marks.sample(frac=1).reset_index(drop=True)

# Display the first few rows to verify
print(df_with_blank_marks.head())

# Export to CSV if needed
df_with_blank_marks.to_csv('name.csv', index=False)


# In[ ]:




