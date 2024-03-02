import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as StrLimit
import matplotlib

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

df = pd.read_csv('news-classification.csv')

df = df[['category_level_1', 'content']]

texts = df['content']
labels = df['category_level_1']

stop_words = set(stopwords.words('english')) 

def clean (text):
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words] 
    text = ' '.join(filtered_words)
    return text

texts = texts.apply(clean)

# print('Some insights about the dataset: ')
# print(df.describe()) 

plt.figure(figsize=(10, 5))
sns.countplot(x='category_level_1', data=df)
labels = [label[:10] for label in df['category_level_1'].unique()]  # Limit the length of labels to 10
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)  # Set the xticks with the limited labels
plt.grid()
plt.plot()
plt.tight_layout()
plt.show()

# min_count = df['category_level_1'].value_counts().min()
texts = texts.values.reshape(-1, 1)
ros = RandomOverSampler(random_state=777)
X_ROS, Y_ROS = ros.fit_resample(texts, labels)

X_ROS = pd.DataFrame(X_ROS)
# X_ROS = pd.DataFrame(X_ROS, columns=['content'])

print(X_ROS.describe())

plt.figure(figsize=(10, 5))
sns.countplot(x=Y_ROS)
plt.title('Distribution of the categories after resampling')
# plt.xticks(rotation=45)
# Optionally, you can also adjust the alignment and the spacing
labels = [label[:10] for label in df['category_level_1'].unique()]  # Limit the length of labels to 10
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)  # Set the xticks with the limited labels
plt.tight_layout()
plt.grid()
plt.plot()
plt.show()
