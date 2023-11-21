# Module ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from cleantext import clean

# Module Web (FLASK)
from flask import Flask, render_template, request

app = Flask(__name__)

def clean_text(text):
    cleaned_text = clean(text, 
                         fix_unicode=True, 
                         to_ascii=True, 
                         lower=False, 
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

y = Lb.fit_transform(dataset.gender)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model = MultinomialNB()

model.fit(x_train, y_train)

def predict_gender(input_name, model, vectorizer):
    input_name_lower = clean_text(input_name)
    input_name_vectorized = vectorizer.transform([input_name_lower])
    prediction = model.predict(input_name_vectorized)[0]
    probabilities = model.predict_proba(input_name_vectorized)[0]
    gender = "Female" if prediction == 0 else "Male"
    prob_female = round(probabilities[0] * 100, 2)
    prob_male = round(probabilities[1] * 100, 2)
    
    return gender, prob_female, prob_male

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        input_name = request.form['name']
        gender, prob_female, prob_male = predict_gender(input_name, model, vectorizer)
        return render_template('predict.html',
        gender=gender, 
        prob_female=prob_female, 
        prob_male=prob_male, 
        dataset=dataset.shape[0], 
        accuracy=round(model.score(x_train, y_train) * 100, 2),
        name=input_name)

if __name__ == '__main__':
    app.run(debug=True)
