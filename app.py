
import pandas as pd
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# --------- Load Spacy NLP and Transformers Once --------- #
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------- Functions --------- #
def summarize_dialog(dialogue_text):
    summary = summarizer(dialogue_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    return summary

def generate_professional_reply(summary):
    return f"Thank you for bringing this to our attention. Based on your concern — {summary.lower()} — we’ll take immediate action to resolve this. Please let us know if you experience any further issues."

def extract_actions(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "TIME", "DATE"]]

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def recommend_resolution(query, ticket_texts, resolutions):
    query = preprocess_text(query)
    ticket_texts = [preprocess_text(ticket) for ticket in ticket_texts]

    query_emb = sentence_model.encode(query, convert_to_tensor=True)
    ticket_embs = sentence_model.encode(ticket_texts, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_emb, ticket_embs)
    best_match_idx = int(similarities.argmax())
    return resolutions[best_match_idx]

def train_router_model(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = MultinomialNB()
    clf.fit(X, labels)
    return vectorizer, clf

def predict_routing(text, vectorizer, clf):
    X_test = vectorizer.transform([text])
    return clf.predict(X_test)[0]

def train_time_predictor(df):
    X = df[['ticket_length', 'has_attachment']]
    y = df['expected_resolution_time']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def predict_resolution_time(model, scaler, ticket_length, has_attachment):
    new_ticket = [[ticket_length, has_attachment]]
    scaled_ticket = scaler.transform(new_ticket)
    return round(model.predict(scaled_ticket)[0], 2)

# --------- Load Data --------- #
summarizer_df = pd.read_csv(r"C:\Users\pranali ajit jamdade\OneDrive\Desktop\CRM\summarizer_data.csv")
actions_df = pd.read_csv(r"C:\Users\pranali ajit jamdade\OneDrive\Desktop\CRM\action_data.csv")
resolution_df = pd.read_csv(r"C:\Users\pranali ajit jamdade\OneDrive\Desktop\CRM\resolution_data.csv")
routing_df = pd.read_csv(r"C:\Users\pranali ajit jamdade\OneDrive\Desktop\CRM\routing_data.csv")
time_df = pd.read_csv(r"C:\Users\pranali ajit jamdade\OneDrive\Desktop\CRM\time_estimator_data.csv")

# Train routing and time models once
vectorizer, clf = train_router_model(routing_df['ticket_text'], routing_df['team_label'])
time_model, scaler = train_time_predictor(time_df)

# --------- Streamlit UI --------- #
st.set_page_config(page_title="AI Customer Support", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f4f6f9; padding: 2rem; border-radius: 1rem;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; background-color: #1363DF; color: white;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Smart AI Assistant for Customer Support")
st.caption("Empowered with NLP, ML, and Transformers")

with st.container():
    st.subheader("1. Summarize Conversation & Generate Agent Reply")
    sample_dialogue = """Customer: Hi, my internet keeps dropping every few minutes and it's really frustrating.  
Agent: I'm sorry to hear that. Have you tried restarting your router?  
Customer: Yes, multiple times. It's still happening.  
Agent: I understand. Let me check if there are any outages in your area."""

    dialogue_input = st.text_area("Enter Customer-Agent Conversation", sample_dialogue, height=200)
    if st.button("Generate Summary and Response"):
        with st.spinner("Analyzing conversation..."):
            summary = summarize_dialog(dialogue_input)
            agent_reply = generate_professional_reply(summary)
            st.success("Summary of Conversation")
            st.write(summary)
            st.success("Professional Agent Reply")
            st.write(agent_reply)

    add_vertical_space(1)

    st.subheader("2. Extract Actions")
    action_text = st.text_area("📌 Input for action extraction", "Please escalate to engineering and arrange a callback for Saturday.")
    if st.button("Extract Entities"):
        st.success(extract_actions(action_text))

    add_vertical_space(1)

    st.subheader("3. Recommend Resolution")
    resolution_query = st.text_area("🔍 Describe your issue", "Getting payment failure issue.")
    if st.button("Get Recommended Resolution"):
        recommendation = recommend_resolution(resolution_query, resolution_df['ticket_text'], resolution_df['resolution_summary'])
        st.success(recommendation)

    add_vertical_space(1)

    st.subheader("4. Predict Routing Team")
    routing_query = st.text_area("📨 Ticket description", "There is a load on Payment Gateway.")
    if st.button("Predict Responsible Team"):
        team = predict_routing(routing_query, vectorizer, clf)
        st.success(f"Route to: {team}")

    add_vertical_space(1)

    st.subheader("5. Estimate Resolution Time")
    col1, col2 = st.columns(2)
    with col1:
        ticket_length = st.number_input("📝 Ticket length (words)", min_value=1, max_value=1000, value=25)
    with col2:
        has_attachment = st.selectbox("📎 Has attachment?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    if st.button("Estimate Time"):
        time = predict_resolution_time(time_model, scaler, ticket_length, has_attachment)
        st.success(f"⏱️ Estimated Resolution Time: {time} hours")

st.markdown("---")
st.markdown("Made with ❤️ by your AI Assistant")