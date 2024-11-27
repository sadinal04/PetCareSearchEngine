from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load data
df = pd.read_csv("cleaned_hewania_articles_tokenized.csv")
df.fillna("", inplace=True)

# Gabungkan kolom judul dan konten untuk perhitungan
df["combined"] = df["Judul_Cleaned"] + " " + df["Content_Cleaned"]

# Load stopwords
with open("stopword.txt", "r") as file:
    stopwords = set(file.read().splitlines())

# Fungsi preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Hilangkan angka dan tanda baca
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenisasi dan hapus stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return " ".join(tokens)

# Fungsi untuk melakukan stemming pada query menggunakan Sastrawi
def stem_query(query):
    return stemmer.stem(query)  # Terapkan stemming pada query

# Fungsi Cosine Similarity
def get_cosine_similarity(query, top_n=20):
    query = preprocess_text(query)
    query = stem_query(query)  # Terapkan stemming pada query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + df["combined"].tolist())
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    results = []
    for idx, cosine_score in enumerate(cosine_sim):
        results.append({
            "index": idx,
            "title": df.loc[idx, "Judul"],
            "link": df.loc[idx, "Link"],
            "date": df.loc[idx, "Tanggal"],
            "content": df.loc[idx, "Content"][:200] + "...",
            "image": df.loc[idx, "Image URL"],
            "cosine_similarity": cosine_score,
        })
    
    results = sorted(results, key=lambda x: x["cosine_similarity"], reverse=True)
    return results[:top_n]

# Fungsi Jaccard Similarity
def get_jaccard_similarity(query, top_n=20):
    query = preprocess_text(query)  # Preprocessing untuk query
    query = stem_query(query)  # Terapkan stemming pada query
    query_tokens = set(query.split())  # Tokenkan query
    
    results = []
    for idx, row in df.iterrows():
        # Preprocessing hanya dilakukan di Jaccard
        doc_tokens = set(preprocess_text(row["combined"]).split())  # Preprocess langsung
        
        # Hitung Jaccard Similarity
        intersection = len(query_tokens & doc_tokens)
        union = len(query_tokens | doc_tokens)
        jaccard_score = intersection / union if union != 0 else 0
        
        results.append({
            "index": idx,
            "title": row["Judul"],
            "link": row["Link"],
            "date": row["Tanggal"],
            "content": row["Content"][:200] + "...",
            "image": row["Image URL"],
            "jaccard_similarity": jaccard_score,
        })
    
    results = sorted(results, key=lambda x: x["jaccard_similarity"], reverse=True)
    return results[:top_n]

# Route index - Menampilkan artikel terbaru
@app.route('/')
def index():
    # Ambil artikel terbaru (misalnya 3 artikel terbaru berdasarkan tanggal)
    latest_articles = df.sort_values(by='Tanggal', ascending=False).head(3)
    
    # Konversi DataFrame menjadi list of dictionaries
    latest_articles_dict = latest_articles.to_dict(orient='records')
    
    # Render template dengan data artikel terbaru
    return render_template('index.html', latest_articles=latest_articles_dict)

# Route search
@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query', '') if request.method == 'POST' else request.args.get('query', '')
    algorithm = request.form.get('algorithm', 'cosine') if request.method == 'POST' else request.args.get('algorithm', 'cosine')
    page = int(request.args.get('page', 1))
    per_page = 10
    
    if algorithm == 'cosine':
        all_results = get_cosine_similarity(query, top_n=30)
    elif algorithm == 'jaccard':
        all_results = get_jaccard_similarity(query, top_n=30)
    else:
        all_results = []
    
    # Filter out results with a score of 0
    filtered_results = [result for result in all_results if (result.get('cosine_similarity', 0) > 0 or result.get('jaccard_similarity', 0) > 0)]
    
    total_results = len(filtered_results)
    total_pages = (total_results + per_page - 1) // per_page
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    results = filtered_results[start:end]
    
    return render_template(
        'results.html',
        query=query,
        results=results,
        algorithm=algorithm,
        page=page,
        total_pages=total_pages,
        total_results=total_results
    )

if __name__ == '__main__':
    app.run(debug=True)
