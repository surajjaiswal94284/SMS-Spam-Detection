import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import( # For test set evaluation
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_FILE='spam_model.pkl'
VECTORIZER_FILE='tfidf.pkl'

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)

    #Remove stopwords
    words=text.split()
    words=[word for word in words if word not in stop_words]

    return ' '.join(words)

if not os.path.exists(MODEL_FILE):
    # Traning Phase

    #Load Dataset
    df=pd.read_csv('spam.csv')

    #Remove duplicates
    df=df.drop_duplicates()
    
    df['cleaned_msg'] = df['message'].fillna('').apply(clean_text)
    
    X=df['cleaned_msg']
    Y=df['label']


    
    X_train,X_test,Y_train,Y_test=train_test_split(
        X,Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    #Vectorize
    tfidf=TfidfVectorizer(max_features=3000)
    X_train_tfidf=tfidf.fit_transform(X_train)
    X_test_tfidf=tfidf.transform(X_test)

    #Final Model
    model=MultinomialNB()
    
    #Train
    model.fit(X_train_tfidf,Y_train)


    # TEST SET EVALUATION
    Y_pred=model.predict(X_test_tfidf)

    print(f"Accuracy: {accuracy_score(Y_test,Y_pred):.4f}")

    print(
        f"F1 Score: "
        f"{f1_score(Y_test,Y_pred,pos_label='spam'):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test,Y_pred))

    print("\nClassification Report:")
    print(classification_report(Y_test,Y_pred))


    #Save Model
    joblib.dump(model,MODEL_FILE)
    joblib.dump(tfidf,VECTORIZER_FILE)
    print("Model Saved Successfully")

else:
    # INFERENCE PHASE
    model=joblib.load(MODEL_FILE)
    tfidf=joblib.load(VECTORIZER_FILE)

    while True:
        message = input( "\nEnter Message (type exit to quit): " )
        if message.lower()=='exit':
            break
        cleaned=clean_text(message)
        vector=tfidf.transform([cleaned])
        prediction=model.predict(vector)
        print(f"Predicton:{prediction[0]}")