from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Sample dataset with links
links = [
    {"title": "10 Impressive Health Benefits of Apples 10 benefits FAQs Takeaway This nutritious fruit offers multiple health benefits. Apples may lower your chance of developing cancer, diabetes, and heart disease. Research says apples may also help you lose weight while improving your gut and brain health.From sweet red varieties, like Red Delicious, Fuji, or Gala, to tangy green ones, like Granny Smith — my favorite with lime juice and a little salt when I want a savory snack — there is an apple for everyone. They’re commonly used in recipes like pies, cookies, muffins, jam, salads, oatmeal, or smoothies. They also make a great snack or wedged and smeared with nut butter. In addition to their culinary versatility and numerous colors and flavors, apples are an exceptionally healthy fruit with many research-backed benefits. Here are eight impressive health benefits of apples.", "url": "https://www.healthline.com/nutrition/10-health-benefits-of-apples"},
    {"title": "Bananas contain fiber as well as vitamins and minerals that may provide health benefits. Bananas are among the most important food crops on the planet. They come from a family of plants called Musa that are native to Southeast Asia and grown in many of the warmer areas of the world. Bananas are a healthy source of fiber, potassium, vitamin B6, vitamin C, and various antioxidants and phytonutrients. Many types and sizes exist. Their color usually ranges from green to yellow, but some varieties are red. This article tells you everything you need to know about bananas.", "url": "https://www.healthline.com/nutrition/foods/bananas"},
    {"title": "Oranges: Nutrients, Benefits, Juice, and More  Nutrition Beneficial plant compoundsBenefitsWhole oranges vs. orange juice Adverse effects The bottom line Many types of oranges are high in fiber and beneficial vitamins, like vitamin C. They also contain antioxidants which can have various health benefits, including supporting immune function. If you’re a fan of citrus fruits, you’ve probably enjoyed your fair share of oranges. Citrus sinensis, or the sweet orange, is the type people typically enjoy fresh and in juice form. Sweet orange trees originated in China thousands of years ago and are now grown in many areas around the world, including the United States, Mexico, and Spain (1Trusted Source, 2Trusted Source). Oranges are a treasure trove of nutrients and protective plant compounds, including vitamins, minerals, and antioxidants. Studies show that consuming oranges regularly may benefit your health in several ways.", "url": "https://www.healthline.com/nutrition/oranges"},
    {"title": "Grapes Health", "url": "https://www.healthline.com/nutrition/benefits-of-grapes"}
]

# Extract titles for TF-IDF
documents = [link["title"] for link in links]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

@app.route('/', methods=['GET', 'POST'])
def search():
    query = request.form.get('query', '').lower()
    
    if query:
        query_vector = tfidf_vectorizer.transform([query])
        scores = (tfidf_matrix * query_vector.T).toarray().flatten()
        results = [
            {"title": links[i]["title"], "url": links[i]["url"], "score": scores[i]} 
            for i in range(len(scores))
        ]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
    else:
        results = []
    
    return render_template('search.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
