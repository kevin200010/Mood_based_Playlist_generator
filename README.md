# Mood_based_Playlist_generator
It is basically program , who gives song playlist based on your mood and your listening history

The research paper, "Song Playlist Generator System Based on Facial Expression and Song Mood," presents an innovative approach to playlist generation by integrating machine learning techniques. It leverages Convolutional Neural Networks (CNN) for real-time emotion detection and Artificial Neural Networks (ANN) for song classification. 

The CNN is used to analyze facial expressions and determine the user's emotional state. With an impressive accuracy of 84%, this emotion detection model has been trained and tested on a substantial dataset comprising around 14,000 facial images, particularly using the FER-13 dataset. The high accuracy demonstrates the efficacy of the CNN in accurately recognizing user emotions based on facial cues.

Simultaneously, an ANN is employed for song classification, categorizing songs into mood-specific groups. This model has been trained using song features extracted from the Spotify music player. The song classification model achieves an accuracy of 82%, underscoring its proficiency in aligning songs with the user's emotional state and personal preferences.

The comparison of results indicates that the combined system effectively provides highly accurate and personalized playlist recommendations. The research's emphasis on accuracy and data-driven decision-making sets it apart from previous methods of playlist generation. This approach enhances user engagement and satisfaction by ensuring that the recommended songs resonate with the user's current mood, ultimately creating a more enjoyable and immersive listening experience.

In summary, the paper's utilization of machine learning techniques, the impressive accuracy of the emotion detection and song classification models, the extensive dataset used, and the notable improvement in playlist generation when compared to conventional methods highlight the research's contributions to the field of personalized music recommendation.

Of course, here's the description of each component using "I":

**1. Developing Model For Facial Emotion Detection:**
   - First, I developed a model for facial emotion detection, which involved two crucial steps:
     - Step 1: I utilized a Haar Cascade feature-based classifier to detect faces in images. This initial step allowed me to locate and isolate facial regions accurately.
     - Step 2: I employed a Convolutional Neural Network (CNN) for emotion detection. My CNN model architecture comprised six convolutional layers followed by three dense layers. These convolutional layers had varying feature sizes, and the dense layers included different numbers of nodes.
   - For training this model, I utilized the FER-13 dataset, which contains data for four facial expressions: Angry, Happy, Sad, and Surprised.
   - The outcome of my efforts was an 83% accuracy in facial emotion recognition, with a loss of 0.4532.

**2. Collecting User Past History Using APIs:**
   - In the second part of the project, I collected user data and their historical music listening records using Application Programming Interfaces (APIs). I set up connections with these APIs, likely from music streaming platforms or databases, to gather user-specific information.

**3. Song Mood Classification:**
   - Moving on to the third component, I designed a system for classifying the mood of songs. The process involved:
     - Creating a Sequential Model for song mood classification, following common deep learning practices.
     - Utilizing the KerasClassifier to encapsulate my Sequential Model as a function.
     - Selecting ten relevant attributes, including song features like Length, Danceability, Acousticness, Energy, Instrumentalness, Liveness, Valence, Loudness, Speechiness, and Tempo, which I found useful for mood prediction.
     - Designing a Sequential Neural Network with four layers, each with various neuron configurations.
   - The model I created achieved an accuracy of 83% for song mood classification, considering four distinct mood classes: Calm Song, Happy Song, Energetic Song, and Sad Song. The dataset I used for this task was sourced from Spotify and contained over 800 songs with diverse attributes.

In summary, I divided the project into well-defined components. In the first part, I focused on facial emotion detection using a CNN, and I achieved an 83% accuracy. The second part involved collecting user data through APIs, and the third part concentrated on song mood classification with a deep learning approach, also achieving an 83% accuracy. These results demonstrate the effectiveness of my machine learning techniques in providing personalized music recommendations based on user emotions and song attributes.
