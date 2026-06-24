import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# STEP 1 - DOWNLOAD STOPWORDS
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))


# STEP 2 - LOAD DATA
df=pd.read_csv('spam.csv')



#Remove duplicates
df=df.drop_duplicates()
print(df.shape)



# STEP 3 - TEXT CLEANING
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)

    #Remove stopwords
    words=text.split()
    words=[word for word in words if word not in stop_words]

    return ' '.join(words)

df['cleaned_msg']=df['message'].fillna('').apply(clean_text)

print(df.head())




# STEP 4 - FEATURES AND TARGET
X=df['cleaned_msg']
Y=df['label']


# STEP 5 - STRATIFIED TRAIN TEST SPLIT
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# STEP 6 - MODEL PIPELINES
models={
    'Naive Bayes':Pipeline([
        ('tfidf',TfidfVectorizer(max_features=3000)),
        ('model',MultinomialNB())
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('model', LogisticRegression(max_iter=1000))
    ]),

    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('model', RandomForestClassifier(random_state=42))
    ])
}


# STEP 7 - CROSS VALIDATION
for name,model in models.items():
    scores=cross_val_score(
        model,
        X_train,
        Y_train,
        scoring='f1_weighted',
        cv=5
    )
    print(f"{name}")
    print(f"F1 Scores : {scores}")
    print(f"Mean F1   : {scores.mean():.4f}")
    print(f"Std Dev   : {scores.std():.4f}")
    print()
