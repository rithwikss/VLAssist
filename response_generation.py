import cohere
from config import cohere_api_key
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

co = cohere.Client(cohere_api_key)

def check_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(vectorizer)
    return similarity[0][1]

def generate_response(caption, previous_response, previous_responses):
    prompt = f"""You are an AI assistant that describes images based on captions.
Given a caption, provide a clear and objective description of what the image likely contains.
Focus on the main elements and details mentioned in the caption.
Avoid personal interpretations, jokes, or sarcasm. Just describe what you see.

For example:
Caption: A man wearing a black t-shirt
Response: The image shows a man wearing a black t-shirt.

Caption: A table and a computer with code
Response: The image depicts a table with a computer on it. The computer screen displays code.

Caption: A group of people playing soccer
Response: The image shows a group of people playing soccer.

Caption: Sunrise from a rooftop
Response: The image captures a sunrise view from a rooftop perspective.

Caption: A person holding a water bottle
Response: The image shows a person holding a water bottle.

Caption: '{caption}'"""

    if previous_response:
        prompt += f"\n\nPrevious response = '{previous_response}'"

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=50,
        temperature=0.3,
        k=0,
        stop_sequences=[],
        return_likelihoods="NONE"
    )
    new_response = response.generations[0].text.strip()

    similarity_threshold = 0.7
    for past_response in previous_responses:
        if check_similarity(new_response, past_response) > similarity_threshold:
            return generate_response(caption, previous_response, previous_responses)

    return new_response