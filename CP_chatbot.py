from flask import Flask, request
from mistralai import Mistral
import requests
import qdrant_client
from sentence_transformers import SentenceTransformer

api_key = "uAyQH2qjnN5Batgf1YDp24KB3BLKLHBK"
mistral_client = Mistral(api_key=api_key)
qdrant_client = qdrant_client.QdrantClient(url="http://localhost:6333")

model = SentenceTransformer('all-MiniLM-L6-v2')

chat_history = {}
def update_chat_history(sender, query, response):
    # Store chat history for the sender
    if sender not in chat_history:
        chat_history[sender] = []
    chat_history[sender].append({"query": query, "response": response})

def is_exit_query(query):
    prompt = f"Does the following sentence indicate that the user wants to exit the conversation? Please respond with True or False only.\n\nSentence: \"{query}\""
    try:
        chat_response = mistral_client.chat.complete(
            model="open-mistral-7b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response = chat_response.choices[0].message.content.strip()
        return response.lower() == "true"
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def is_greeting(query):
    prompt = f"Is the following sentence a greeting? Please respond with True or False only.\n\nSentence: \"{query}\""
    try:
        chat_response = mistral_client.chat.complete(
            model="open-mistral-7b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response = chat_response.choices[0].message.content.strip()
        return response.lower() == "true"
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

collection_name = "cp_profile"    
def search_pdf(query):
    query_embedding = model.encode(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=1
    )
    return search_result

def generate_response_mistral(query, history, context):
    system_prompt = (
        "You are an AI assistant for 'Clouxi Plexi'."
        "Provide factual, concise, and natural responses to queries about the company."
        "Do not answer queries that are unrelated to Clouxi Plexi."
        "If the information isn't available, simply state that you don't have information on that topic and provide the contact details of the company Email:info@clouxiplexi.com"
        "Never mention the source of the information (e.g., 'PDF' or 'document')."
        "Do not tell user that you hve been trained on any information."
    )
    # Convert chat history to a readable format
    history_text = "\n".join([f"Q: {entry['query']}\nA: {entry['response']}" for entry in history])
    # Rename 'context' to something more natural
    company_context = context
    # prromt history or new query dono p based hai
    prompt = (
        f"Context from chat history:\n{history_text}\n\n"
        f"Relevant Company Information:\n{company_context}\n\n"
        f"User Query:\n{query}\n\nResponse:"
    )
    response = ""
    try:
        chat_response = mistral_client.chat.complete(
            model="open-mistral-7b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        response = chat_response.choices[0].message.content.strip()
        unwanted_phrases = ["PDF", "document", "source", "provided","context"] #post processing to remove unwanted mentions of context or PDF
        for phrase in unwanted_phrases:
            if phrase in response:
                response = response.replace(phrase, "")
        return response
    except Exception as e:
        return f"An error occurred: {e}"

def send_whatsapp_message(to, body):
    ultramsg_instance_id = "instance93703"
    ultramsg_token = "klecvfzsqyippgue"
    url = f"https://api.ultramsg.com/{ultramsg_instance_id}/messages/chat"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "token": ultramsg_token,
        "to": to,
        "body": body
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")

app = Flask(__name__)
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message_data = data.get('data', {})
    if not message_data:
        print("No message data found in the payload")
        return
    query = message_data.get('body', '')
    sender = message_data.get('from', '')
    if not query or not sender:
        print("Query or sender information is missing")
        return
    if '@g.us' in sender:
        return "Message received from a group chat. Ignoring."
    if is_exit_query(query):
        send_whatsapp_message(to=sender, body="OK bye! Have a good day :)")
        chat_history.pop(sender, None)
        return
    if is_greeting(query):
        send_whatsapp_message(to=sender, body="Hello! How can I assist you?")
        return

    search_result = search_pdf(query)
    if not search_result or len(search_result) == 0:
        send_whatsapp_message(to=sender, body="Sorry, I couldn't find relevant information.")
        return "No relevant information found."

    history = chat_history.get(sender, [])
    response = generate_response_mistral(query, history, search_result)
    update_chat_history(sender, query, response)
    send_whatsapp_message(to=sender, body=response)
    return "OK"
   

if __name__ == '__main__':
    app.run(port=5000)
