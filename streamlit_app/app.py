import os
import streamlit as st
import requests
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="🎥 Movie Recommender", layout="wide")
BASE_URL = os.getenv("API_URL", "http://localhost:8000")
# Styled Title Block
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B; font-size: 3em;'>
        🍿 <span style='color: white;'>BingeMax</span>: Movie Recommendation Engine 🎬
    </h1>
 """, unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    ":mag: Search",
    ":bar_chart: Recommend",
    ":star: Top Picks",
    "👤 User Recommendations"
])

def show_recommendation_graph(center_node, recommendations_df):
    try:
        G = nx.Graph()
        G.add_node(center_node)

        for _, row in recommendations_df.iterrows():
            title = row.get("title", "Unknown")
            G.add_node(title)
            G.add_edge(center_node, title)

        pos = nx.spring_layout(G)

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'), mode='lines'))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                 marker=dict(size=20, color='skyblue'),
                                 text=node_text, textposition="bottom center"))
        fig.update_layout(title=f"Recommendation Network for '{center_node}'",
                          showlegend=False, height=600, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Graph rendering failed: {e}")

@st.cache_data(show_spinner=False)
def fetch_titles():
    try:
        res = requests.get(f"{BASE_URL}/titles")
        return res.json() if res.ok else []
    except:
        return []

all_titles = fetch_titles()

with tab1:
    st.subheader(":mag: Search Movies by Title")
    if all_titles:
        movie_name = st.selectbox("Select or search movie title:", all_titles)
        if movie_name:
            response = requests.get(f"{BASE_URL}/search", params={"title": movie_name, "max_results": 10})
            if response.ok:
                df = pd.DataFrame(response.json()).drop(columns=["tconst"], errors="ignore")
                st.dataframe(df)
            else:
                st.error(response.json().get("detail", "Error during search"))
    else:
        st.warning("⚠️ No titles available. Ensure API is running and returns data.")

with tab2:
    st.subheader(":dart: Movie-Based Recommendations")
    if all_titles:
        movie_dropdown = st.selectbox("Select a movie to get recommendations:", all_titles)
        if st.button("Show Recommendations"):
            try:
                response = requests.get(f"{BASE_URL}/recommend", params={"title": movie_dropdown})
                if response.ok:
                    df = pd.DataFrame(response.json()).drop(columns=["tconst"], errors="ignore")
                    if df.empty:
                        st.warning("No recommendations found.")
                    else:
                        show_recommendation_graph(movie_dropdown, df)
                        st.dataframe(df)
                else:
                    st.error(response.json().get("detail", "Error fetching recommendations"))
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.warning("⚠️ No movie titles found. Cannot generate recommendations.")

with tab3:
    st.subheader(":fire: Explore Movies")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Show Top Rated"):
            res = requests.get(f"{BASE_URL}/top_rated?limit=20")
            if res.ok:
                df = pd.DataFrame(res.json()).drop(columns=["tconst"], errors="ignore")
                st.dataframe(df)
                fig = px.bar(df, x="title", y="rating", title="Top Rated Movies", labels={"title": "Movie", "rating": "Rating"})
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to fetch top rated movies.")

        if st.button("Show Random Picks"):
            res = requests.get(f"{BASE_URL}/random?limit=20")
            if res.ok:
                df = pd.DataFrame(res.json()).drop(columns=["tconst"], errors="ignore")
                st.dataframe(df)
            else:
                st.error("Failed to fetch random picks.")

        try:
            genres_res = requests.get(f"{BASE_URL}/genres")
            genre_list = genres_res.json() if genres_res.ok else []
        except Exception as e:
            st.error(f"Could not load genres: {e}")
            genre_list = []

        selected_genres = st.multiselect("Select genre(s):", genre_list)
        if selected_genres:
            all_movies = []
            for genre in selected_genres:
                res = requests.get(f"{BASE_URL}/genre/{genre}?limit=25")
                if res.ok:
                    all_movies.extend(res.json())

            if all_movies:
                df = pd.DataFrame(all_movies).drop_duplicates(subset="tconst", keep="first").drop(columns=["tconst"], errors="ignore")
                st.dataframe(df)

                genre_counts = df["genres"].str.split(",").explode().str.strip().value_counts()
                st.markdown("### 🎬 Genre Distribution (Filtered Results)")
                st.bar_chart(genre_counts)

                fig = px.bar(df.head(10), x="title", y="rating", title="Top Movies by Selected Genres")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No movies found for selected genres.")

    with col2:
        st.markdown("### ℹ️ Instructions")
        st.write("""
        - Select one or more genres to view matching movies.
        - You can also explore top-rated or random picks.
        - A genre frequency chart and bar plot will appear if data is available.
        """)

with tab4:
    st.subheader("👤 Get Personalized Recommendations by User ID")
    user_id = st.text_input("Enter your User ID:", value="", max_chars=10)
    if user_id:
        try:
            response = requests.get(f"{BASE_URL}/recommend_user", params={"user_id": user_id})
            if response.ok:
                recommendations = response.json()
                if recommendations:
                    st.success(f"Top {len(recommendations)} Recommendations for User {user_id}:")
                    df = pd.DataFrame(recommendations).drop(columns=["tconst"], errors="ignore")
                    show_recommendation_graph(f"User {user_id}", df)
                    st.dataframe(df)
                else:
                    st.warning(f"No recommendations found for User ID {user_id}.")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to fetch recommendations: {e}")
