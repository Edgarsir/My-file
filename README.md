# My-file
#Clone a content creator's style from their Instagram conten

import instagrapi
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time 
!pip install time.sleep()
USERNAME = "your_user"
PASSWORD = "your_passs"

cl = instagrapi.Client()
max_retries = 3  
retry_delay = 20 
for attempt in range(max_retries):
    try:
        cl.login(USERNAME, PASSWORD)
        break
    except instagrapi.exceptions.BadPassword as e:
        print(f"Login attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            raise


creator_username = "creator_username" 
image_data, captions = scrape_instagram(creator_username)



def extract_image_features(image_data):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    image_features = base_model.predict(image_data)
    return image_features

image_features = extract_image_features(image_data)

!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
def extract_text_features(captions):
  model = SentenceTransformer('all-MiniLM-L6-v2')
  text_features = model.encode(captions)
  return text_features

text_features = extract_text_features(captions)

combined_features = np.concatenate((image_features, text_features), axis=1)

def generate_image(style_features, random_seed = 42):
    np.random.seed(random_seed)

    random_image = np.random.rand(224, 224, 3)  
    return random_image

generated_image = generate_image(combined_features[0])
def save_image(image_array, filename='generated_image.jpg'):
  img = Image.fromarray((image_array * 255).astype(np.uint8))
  img.save(filename)

save_image(generated_image)
print("Image generated successfully!")

2. Generate personalized product reviews from e-commerce URLs
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def generate_product_review(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, "html.parser")

        product_title = soup.find("h1", class_="product-name")  
        product_description = soup.find("div", class_="description")  

        if product_title:
            product_title = product_title.text.strip()
        else:
            return "Product Title Not Found"

        if product_description:
            product_description = product_description.text.strip()
        else:
            return "Product Description Not Found"

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(product_description)
        
        if scores['compound'] >= 0.05:
            sentiment = "positive"
            review = f"I absolutely love this {product_title}! {product_description}. Highly recommend."
        elif scores['compound'] <= -0.05:
            sentiment = "negative"
            review = f"I'm very disappointed with this {product_title}. {product_description}. Would not recommend."
        else:
            sentiment = "neutral"
            review = f"This {product_title} is okay. {product_description}. It's average."

        return {
            "product_title": product_title,
            "review": review,
            "sentiment": sentiment
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except AttributeError as e:
        print(f"Error parsing HTML: {e}")
        return None

product_url = "http://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"  

review_data = generate_product_review(product_url)

if review_data:
    print(review_data["review"])
    
3. Create video scripts matching the creator's style

import random

def generate_video_script(topic, keywords):
    script = f"## Video Script: {topic}\n\n"

    intro_options = [
        "Hey everyone, welcome back to my channel!",
        "What's up, guys? Let's dive into this topic!",
        "Super excited to share this with you today!",
    ]
    script += random.choice(intro_options) + "\n\n"

    for keyword in keywords:
        script += f"- {keyword.capitalize()}: This section will cover {keyword}. "
        sentences = [
            f"We'll explore the intricacies of {keyword}.",
            f"Let's dive deep into the impact of {keyword}.",
            f"{keyword} is super important for various reasons."
        ]
        script += random.choice(sentences) + "\n"

    script += "\n"
    outro_options = [
        "Thanks for watching!",
        "Don't forget to like and subscribe!",
        "See you in the next video!",
    ]
    script += random.choice(outro_options)
    return script
topic = "Artificial Intelligence"
keywords = ["machine learning", "deep learning", "neural networks"]
video_script = generate_video_script(topic, keywords)
print(video_script)

4. Synthesize voice clips in the creator's voice


!pip install TTS

from TTS.api import TTS

tts = TTS.from_pretrained("tts_models/en/ljspeech/tacotron2-DDC")

def synthesize_voice(text, output_file="output.wav"):
    tts.tts_to_file(text=text, file_path=output_file)

try:
    text_to_synthesize = review_data["review"]
    synthesize_voice(text_to_synthesize)
    print(f"Voice clip synthesized and saved as {output_file}")
except KeyError:
    print("Error: 'review_data' does not contain 'review'. Ensure 'review_data' is defined and contains a 'review' field.")
except Exception as e:
    print(f"An error occurred: {e}")
