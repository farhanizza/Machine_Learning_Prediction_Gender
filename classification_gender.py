import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from cleantext import clean

def clean_text(text):
    cleaned_text = clean(text, 
                         fix_unicode=True, 
                         to_ascii=True, 
                         lower=True, 
                         no_line_breaks=True, 
                         no_urls=True, 
                         no_emails=True, 
                         no_phone_numbers=True, 
                         no_numbers=True, 
                         no_digits=True, 
                         no_currency_symbols=True, 
                         no_punct=True, 
                         replace_with_punct="")
    
    joined_text = ''.join(cleaned_text.split())

    return joined_text

dataset = pd.read_csv('name_gender_all.csv')

vectorizer = CountVectorizer()

name = dataset['name'].apply(lambda text: ''.join(text.split()).lower())

x = vectorizer.fit_transform(name)

Lb = LabelEncoder()

gender = dataset['gender'].apply(lambda text: text.lower())

y = Lb.fit_transform(gender)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model = MultinomialNB()

model.fit(x_train, y_train)

print(f"How many rows the datasets {dataset.shape[0]}")

print(f"The accuracy of the model on the test set is {round(model.score(x_train, y_train) * 100, 2)}%.")

def predict_gender(input_name, model, vectorizer):
    input_name_lower = clean_text(input_name)
    input_name_vectorized = vectorizer.transform([input_name_lower])
    prediction = model.predict(input_name_vectorized)[0]
    probabilities = model.predict_proba(input_name_vectorized)[0]
    gender = "Female" if prediction == 0 else "Male"
    prob_female = round(probabilities[0] * 100, 2)
    prob_male = round(probabilities[1] * 100, 2)
    
    return gender, prob_female, prob_male

input_name = input("Masukkan nama untuk diprediksi gender-nya: ")

predicted_gender, prob_female, prob_male = predict_gender(input_name, model, vectorizer)

print(f"Prediksi gender untuk nama '{input_name}': {predicted_gender}")

print(f"Probabilitas Female: {prob_female}%")

print(f"Probabilitas Male: {prob_male}%")