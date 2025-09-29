import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import random
from collections import defaultdict
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

# Spotify API credentials
CLIENT_ID = "7842cb1b5ec04b1fa16278f9fd712a0b"
CLIENT_SECRET = "0ea669da75654ac6b432699fd233590b"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define RL Agent
class MusicRecommenderAgent:
    def __init__(self, song_list, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.song_list = song_list
        self.q_table = defaultdict(lambda: np.zeros(len(song_list)))  # Q-table initialized to zeros
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.song_list)))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

# Fetch album cover URL
def get_song_album_cover_url(song_name, artist_name):
    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track")
        if results and results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            album_cover_url = track["album"]["images"][0]["url"]
            return album_cover_url
    except Exception as e:
        logging.warning(f"Failed to fetch album cover for {song_name}: {e}")
    return "no-music-icon.png"

# Recommendation function
def recommend(song, exclude_list=None):
    exclude_list = exclude_list or []
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_music_names = []
    recommended_music_posters = []

    for i in distances[1:]:
        try:
            song_name = music.iloc[i[0]].song
            artist = music.iloc[i[0]].artist
            if song_name not in exclude_list and song_name not in recommended_music_names:
                recommended_music_posters.append(get_song_album_cover_url(song_name, artist))
                recommended_music_names.append(song_name)
            if len(recommended_music_names) == 5:
                break
        except Exception as e:
            logging.warning(f"Failed to recommend song: {e}")
            continue

    return recommended_music_names, recommended_music_posters

# Streamlit UI
st.set_page_config(
    page_title="Music Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🎵 Music Recommender")
st.sidebar.write("Navigate through the app:")
page = st.sidebar.radio("", ["Home", "Recommendations", "About"])

# Load data
music = pickle.load(open('C:\\Users\\sarth\\Desktop\\app\\df.pkl', 'rb'))
similarity = pickle.load(open('C:\\Users\\sarth\\Desktop\\app\\similarity.pkl', 'rb'))

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = MusicRecommenderAgent(music['song'].values.tolist())
    st.session_state.user_state = 0
    st.session_state.recommendations = []
    st.session_state.exclude_list = []

agent = st.session_state.agent
user_state = st.session_state.user_state

if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎵 Welcome to the Music Recommender System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Discover personalized music recommendations based on your preferences!</h3>", unsafe_allow_html=True)
    with st.expander("How It Works"):
        st.write("""
        1. Select a song from the dropdown menu.
        2. View recommendations based on lyrics similarity.
        3. Provide feedback on recommendations by liking or skipping songs.
        4. The system dynamically learns your preferences and improves recommendations!
        """)
elif page == "Recommendations":
    st.title("🎧 Personalized Recommendations")
    selected_song = st.selectbox(
        "Type or select a song from the dropdown",
        music['song'].values,
        help="Start typing to search for a song."
    )
    if st.button("Show Recommendations"):
        recommended_music_names, recommended_music_posters = recommend(selected_song, exclude_list=st.session_state.exclude_list)
        st.session_state.recommendations = list(zip(recommended_music_names, recommended_music_posters))

    if st.session_state.recommendations:
        st.write("### Recommended Songs:")
        cols = st.columns(5)
        feedback = []
        for i, (song, poster) in enumerate(st.session_state.recommendations):
            with cols[i % 5]:
                st.image(poster, use_column_width=True)
                st.text(song)
                feedback.append(st.radio(f"Feedback for {song}", ['Like', 'Skip'], key=f"feedback_{i}"))

        for i, (song, user_feedback) in enumerate(zip(st.session_state.recommendations, feedback)):
            song_name = song[0]
            try:
                action = music[music['song'] == song_name].index[0]
                reward = 1 if user_feedback == "Like" else -1
                next_state = action
                agent.update_q_table(user_state, action, reward, next_state)
                st.session_state.user_state = next_state
                if user_feedback == "Skip":
                    st.session_state.exclude_list.append(song_name)
            except Exception as e:
                logging.error(f"Error processing feedback for {song_name}: {e}")

        next_recommendations, next_posters = recommend(selected_song, exclude_list=st.session_state.exclude_list)
        if next_recommendations:
            st.session_state.recommendations = list(zip(next_recommendations, next_posters))
        else:
            st.error("No next recommendations available.")
elif page == "About":
    st.title("📖 About This Project")
    st.write("""
    This music recommender system uses:
    - **Lyrics similarity (TF-IDF vectorization and cosine similarity)** for initial recommendations.
    - **Reinforcement Learning** to dynamically adapt to user feedback and personalize recommendations.
    - Spotify API integration for album covers and audio previews.
    """)

    # Feedback Chart Example
    st.write("### User Feedback Visualization")
    songs = ["Song A", "Song B", "Song C"]
    likes = [10, 5, 3]
    fig, ax = plt.subplots()
    ax.bar(songs, likes)
    st.pyplot(fig)
