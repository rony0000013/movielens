import streamlit as st
import requests

st.title('🎬 MovieLens 📸')

st.header('Upload Movie')
movie_file = st.file_uploader('Choose a movie file', type=['mp4', 'avi', 'mkv', 'mp3', 'wav', 'ogg'])

if st.button('Analyze Movie'):
    if movie_file is not None:
        files = {'movie_file': movie_file.getvalue()}
        response = requests.post(f'{st.secrets["SERVER_URL"]}/analyze_movie', files=files)
        if response.status_code == 200:
            st.success('Movie analyzed successfully!')
            st.json(response.json())
        else:
            st.error('Failed to analyze movie')
    else:
        st.warning('Please upload a movie file')

st.header('Query Movie')
query = st.text_input('Ask about the movie...')

if st.button('Get Insights'):
    if query:
        response = requests.post(f'{st.secrets["SERVER_URL"]}/query_movie', params={'query': query})
        if response.status_code == 200:
            st.success('Query successful!')
            answer = response.json()['llm_response']
            st.write(answer)
        else:
            st.error('Failed to query movie')
    else:
        st.warning('Please enter a query')

st.header('Voice Query')
audio_file = st.audio_input('Record your voice query')

if audio_file is not None:
    files = {'audio_file': audio_file}
    response = requests.post(f'{st.secrets["SERVER_URL"]}/query_movie', files=files)
    if response.status_code == 200:
        st.success('Audio query successful!')
        st.write(response.json()["llm_response"])
    else:
        st.error('Failed to query movie with audio')
