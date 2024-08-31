import requests
import re
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template, request, url_for, session, send_file, send_from_directory, Response
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import random
import io
import secrets
import tempfile
import os
from collections import Counter

nltk.download("stop_words")
nltk.download("punkt")
nltk.download("punkt_tab")

secret_key = secrets.token_hex(16)  # Generate a secure random key

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; AS; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:85.0) Gecko/20100101 Firefox/85.0"
]

######     CLASSES     ######

class AppException(Exception):
    def __init__(self, value):
        self.__value = value
    def toString(self):
        return self.__value

######     FUNCTIONS     #######

def getProductDetails(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    check = re.search("www.amazon.co.uk", url)
    if not check:
        raise AppException("Invalid Amazon Link")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise AppException("There is an issue with this link")
    else:
        soup = BeautifulSoup(response.content, "lxml")
        title = soup.find("span", {"id": "productTitle"}).get_text(strip=True)
        rating = soup.find("span", {"class": "a-icon-alt"}).get_text(strip=True)
        image_url = soup.find("img", {"id": "landingImage"})["src"]
        return [title, rating, image_url]
    
def getReviewsFromPage(soup):
    review_titles = []
    review_ratings = []
    review_texts = []
    reviews = soup.find_all("div", {"class": "a-section review aok-relative"})
    for review in reviews:
        title = review.find("a", {"data-hook": "review-title"}).get_text(strip=True)
        rating = review.find("i", {"data-hook": "review-star-rating"}).get_text(strip=True)
        text = review.find("span", {"data-hook": "review-body"}).get_text(strip=True)
        if title.startswith(rating):
            title = title[len(rating):].strip()
        review_titles.append(title)
        review_ratings.append(rating)
        review_texts.append(text)
    return review_titles, review_ratings, review_texts

def get_random_user_agent():
    return random.choice(user_agents)

def getAllReviews(url, ASIN, num):
    all_titles = []
    all_ratings = []
    all_texts = []
    while num < 50:
        headers = {"User-Agent": get_random_user_agent()}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "lxml")
        titles, ratings, texts = getReviewsFromPage(soup)
        all_titles.extend(titles)
        all_ratings.extend(ratings)
        all_texts.extend(texts)
        next_page = soup.find("li", {"class": "a-last"})
        if not next_page or not next_page.find("a"):
            break
        num = str(num)
        next_page_url = f"https://www.amazon.co.uk/product-reviews/{ASIN}/?pageNumber={num}"
        num = int(num)
        num += 1
        url = next_page_url
    df = pd.DataFrame({"Title": all_titles, "Rating": all_ratings, "Review": all_texts})
    return df

def extractASIN(url):
    asin_pattern = r"/(?:dp|gp|product)/([A-Z0-9]{10})"
    match = re.search(asin_pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def clean_text(text):
    text = BeautifulSoup(text,"html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    return text

def makeWordCloud(df):
    if df.empty:
        raise AppException("Amazon has blocked this request. Please try again")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["final"]))
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return buffer

def makePieChart(df):
    if df.empty:
        raise AppException("Amazon has blocked this request. Please try again")
    satisfied_words = {"excellent", "great", "amazing", "good", "wonderful", "comfortable"}
    unsatisfied_words = {"terrible", "bad", "awful", "uncomfortable", "poor", "delayed"}
    
    def classify_review(text, satisfied, unsatisfied):
        words = set(text.split())
        if words.intersection(satisfied):
            return "Satisfied"
        elif words.intersection(unsatisfied):
            return "Unsatisfied"
        else:
            return "Neutral"
    
    df["classification"] = df["final"].apply(classify_review, args=(satisfied_words, unsatisfied_words))
    classification_counts = df["classification"].value_counts()
    
    buffer = io.BytesIO()
    plt.figure(figsize=(8, 8))
    plt.pie(classification_counts, labels=classification_counts.index, autopct="%1.1f%%", startangle=140, colors=["lightgreen", "lightcoral", "lightblue"])
    plt.title("Classification of Reviews")
    plt.axis('equal')
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer

def makeHistogram(df):
    if df.empty:
        raise AppException("Amazon has blocked this request. Please try again")
    df['review_length'] = df['final'].apply(lambda x: len(x.split()))
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.hist(df['review_length'], bins=30, color='lightgreen', edgecolor='black')
    plt.xlabel('Review Length (words)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Review Lengths')
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer

def makeBarChart(df):
    if df.empty:
        raise AppException("Amazon has blocked this request. Please try again")
    # Combine all the reviews into a single string and split into words
    all_words = ' '.join(df["final"]).split()
    
    # Get the frequency of the most common 15 words
    word_freq = Counter(all_words).most_common(15)
    words, freqs = zip(*word_freq)
    
    # Create a BytesIO buffer to save the plot
    buffer = io.BytesIO()
    
    # Generate the bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs, color='skyblue')
    plt.xticks(rotation=90)
    plt.title('Top 15 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    
    # Save the plot to the buffer
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    
    # Seek to the beginning of the buffer
    buffer.seek(0)
    
    return buffer

######     APP     ######

webpage = Flask(__name__)
webpage.secret_key = secret_key

@webpage.route("/")
@webpage.route("/home", methods=["GET"])
def home():
    if "df" in session:
        session.pop("df", None)
    return render_template("homepage.html")

@webpage.route("/product", methods=["GET", "POST"])
def product():
    if request.method == "POST":
        url = request.form["url"]
        ASIN = extractASIN(url)
        reviewURL = f"https://www.amazon.co.uk/product-reviews/{ASIN}/?pageNumber=1"
        try:
            details = getProductDetails(url)
            df = getAllReviews(reviewURL, ASIN, 1)
        except AppException as e:
            error = e.toString()
            return render_template("homepage.html", error=error)
        
        reviews = df["Review"]
        df["clean_reviews"] = reviews.apply(clean_text)
        stop_words = set([
            "not", "and", "but", "there", "or", "ok", "the", "i", "was", "with", "in", "on", "to", "am", "me", "of", "what", "were", "s",
            "check", "u", "us", "t", "a", "re", "as", "we", "at", "an", "our", "it", "my", "they", "had", "that", "have", "which", "her", "him",
            "their", "for", "is", "this", "from", "so", "very", "you", "be", "are", "when", "all", "by", "after", "would", "get", "only", "no",
            "out", "them", "been", "up", "then", "just", "did", "if", "even", "could", "can", "again", "other", "more", "first", "any", "will",
            "about", "has", "over", "before", "off", "than", "do", "because", "got", "who", "never", "new", "another", "due", "still", "like",
            "two", "also", "some", "now", "always", "money", "bought"
        ])
        df["tokenized"] = df["clean_reviews"].apply(nltk.word_tokenize)
        df["cleaned_tokenized"] = df["tokenized"].apply(lambda tokens: [word for word in tokens if word not in stop_words])
        df["final"] = df["cleaned_tokenized"].apply(lambda tokens: " ".join(tokens))
        try:
            wordcloud_image = makeWordCloud(df)
            piechart_image = makePieChart(df)
            histogram_image = makeHistogram(df)
            barchart_image = makeBarChart(df)
        except AppException as e:
            error = e.toString()
            return render_template("homepage.html", error=error)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="static") as temp_file_wc:
            temp_file_wc.write(wordcloud_image.getvalue())
            temp_file_wc.seek(0)
            wordcloud_file_path = os.path.basename(temp_file_wc.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="static") as temp_file_pc:
            temp_file_pc.write(piechart_image.getvalue())
            temp_file_pc.seek(0)
            piechart_file_path = os.path.basename(temp_file_pc.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="static") as temp_file_hg:
            temp_file_hg.write(histogram_image.getvalue())
            temp_file_hg.seek(0)
            histogram_file_path = os.path.basename(temp_file_hg.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="static") as temp_file_bc:
            temp_file_bc.write(barchart_image.getvalue())
            temp_file_bc.seek(0)
            barchart_file_path = os.path.basename(temp_file_bc.name)
        print(df)
        return render_template("product.html", name=details[0], rating=details[1], imageLink=details[2], wordcloud_url=wordcloud_file_path, piechart_url=piechart_file_path, histogram_url=histogram_file_path, barchart_url=barchart_file_path)

    return render_template("homepage.html")

if __name__ == "__main__":
    webpage.run(debug=True)
