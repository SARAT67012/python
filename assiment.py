# import pandas as pd
# import numpy as np

# print("Pandas and NumPy are successfully imported!")

# import pandas as pd
# import numpy as np

# # Creating the DataFrame
# data = {
#     'CustomerID': [1, 2, 3, 4, 5],
#     'Age': [45, 0, 72, 122, 44],
#     'Subscription Months': [6, 3, 14, np.nan, 8],
#     'Satisfaction Score': [8, np.nan, 5, 7, 3],
#     'Feedback': [
#         'Good quality, needs more variety',
#         'Delivery was late',
#         'Tasty but too expensive',
#         'Fine',
#         'Not happy with the portion sizes'
#     ]
# }

# df = pd.DataFrame(data)

# # Identifying NaN values in numeric fields
# print("NaN Values in Each Column:")
# print(df.isna().sum())

# # Handling Age column
# # Replace 0 and values above 100 with the median of valid ages
# valid_ages = df.loc[(df['Age'] > 0) & (df['Age'] <= 100), 'Age']
# median_age = valid_ages.median()
# df.loc[df['Age'] == 0, 'Age'] = median_age
# df.loc[df['Age'] > 100, 'Age'] = median_age

# # Handling Subscription Months column
# # Impute missing values with median
# median_subscription = df['Subscription Months'].median()
# df['Subscription Months'] = df['Subscription Months'].fillna(median_subscription)

# # Handling Satisfaction Score column
# # Ensure scores are within 1-10 range and impute missing values with median
# valid_scores = df.loc[(df['Satisfaction Score'] >= 1) & (df['Satisfaction Score'] <= 10), 'Satisfaction Score']
# median_score = valid_scores.median()
# df.loc[df['Satisfaction Score'] < 1, 'Satisfaction Score'] = median_score
# df.loc[df['Satisfaction Score'] > 10, 'Satisfaction Score'] = median_score
# df['Satisfaction Score'] = df['Satisfaction Score'].fillna(median_score)

# print("\nCleaned Data:")
# print(df)


# #q2.

# # Detect outliers using IQR method
# Q1_age = df['Age'].quantile(0.25)
# Q3_age = df['Age'].quantile(0.75)
# IQR_age = Q3_age - Q1_age

# # Define outlier boundaries
# lower_bound_age = Q1_age - 1.5 * IQR_age
# upper_bound_age = Q3_age + 1.5 * IQR_age

# # Cap Age values
# df['Age'] = df['Age'].apply(lambda x: median_age if x < lower_bound_age or x > upper_bound_age else x)

# # Detect outliers in Subscription Months
# Q1_sub = df['Subscription Months'].quantile(0.25)
# Q3_sub = df['Subscription Months'].quantile(0.75)
# IQR_sub = Q3_sub - Q1_sub

# # Define outlier boundaries for Subscription Months
# lower_bound_sub = Q1_sub - 1.5 * IQR_sub
# upper_bound_sub = Q3_sub + 1.5 * IQR_sub

# # Cap Subscription Months values
# df['Subscription Months'] = df['Subscription Months'].apply(lambda x: median_subscription if x < lower_bound_sub or x > upper_bound_sub else x)

# print("\nCleaned Data After Outlier Treatment:")
# print(df)

# #q.3



# from nltk.sentiment import SentimentIntensityAnalyzer
# import nltk

# nltk.download('vader_lexicon')
# sia = SentimentIntensityAnalyzer()

# # Function to classify sentiment
# def classify_sentiment(text):
#     score = sia.polarity_scores(text)['compound']
#     if score > 0:
#         return 'Positive'
#     elif score < 0:
#         return 'Negative'
#     else:
#         return 'Neutral'

# # Apply sentiment analysis
# df['Sentiment'] = df['Feedback'].apply(classify_sentiment)

# # Extract most common pain points using word frequency
# from collections import Counter
# import re

# all_feedback = ' '.join(df['Feedback']).lower()
# words = re.findall(r'\b\w+\b', all_feedback)  # Extract words
# common_words = Counter(words).most_common(10)  # Find 10 most common words

# print("\nSentiment Analysis Result:")
# print(df[['Feedback', 'Sentiment']])
# print("\nCommon Pain Points:", common_words)
