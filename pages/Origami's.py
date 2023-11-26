import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define parameters
batch_size = 16
image_size = (224, 224)
# Define the dataset path
train_data_dir = 'C:/Users/LENOVO/Desktop/Untitled Folder/fyp_20230907-1/train/'
valid_data_dir = 'C:/Users/LENOVO/Desktop/Untitled Folder/fyp_20230907-1/valid/'


# Create ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',

)

import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('origami_shape_classifier.h5')

import streamlit as st

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)

try:
    #test_image_path = st.file_uploader("Choose a Image file", accept_multiple_files=False)
    img = tf.keras.preprocessing.image.load_img(picture, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    # Make predictions
    predictions = model.predict(img_array)

    # Map predicted class index to the actual class name
    class_names = train_generator.class_indices
    predicted_class = np.argmax(predictions)
    predicted_class_name = [k for k, v in class_names.items() if v == predicted_class][0]

    #st.image(test_image_path, width=300)
    st.write(f"Predicted class index: {predicted_class}")
    st.write(f"Predicted class name: {predicted_class_name}")
        
    import streamlit as st
    import requests
    
    if st.button("Fetch Youtube Video"):
        # Streamlit app
        st.title("YouTube Video")

        # YouTube video ID
        video_id = str(predicted_class_name)

        # Make a request to the YouTube API to get video details
        api_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={video_id}+origami&type=video&key=AIzaSyBZEO2RaoBgVKW_wJtYsvuP9EblzSVf_5k"
        response = requests.get(api_url)
        data = response.json()
        # Check if the API request was successful
        if "items" in data and data["items"]:
            video_id = data["items"][0]["id"]["videoId"]
            video_url = f"https://www.youtube.com/embed/{video_id}"
            video_details = data["items"][0]["snippet"]
            video_title = video_details["title"]
            video_description = video_details["description"]

            # Display video information
            st.write(f"**Title:** {video_title}")
            

            # Display the YouTube video using HTML iframe
            st.markdown(f'<iframe width="560" height="315" src="{video_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
        else:
            st.error("Error fetching video details. Please check the video ID and API key.")

    if st.button("Fetch Questions"):
        import streamlit as st


        if 'num' not in st.session_state:
            st.session_state.num = 0


        choices1 = ['Answer1', 'Answer2', 'Answer3', 'Answer4']
        choices2 = ['Answer1', 'Answer2', 'Answer3', 'Answer4']

        qs1 = [('Question 1', choices1),
            ('Question 1', choices1),
            ('Question 1', choices1)]
        qs2 = [('Question 2?', choices2),
            ('Question 1?', choices2),
            ('Question 1?', choices2)]
        


        def main():
            for _, _ in zip(qs1, qs2): 
                placeholder = st.empty()
                num = st.session_state.num
                with placeholder.form(key=str(num)):
                    st.radio(qs1[num][0], key=num+1, options=qs1[num][1])
                    st.radio(qs2[num][0], key=num+2, options=qs2[num][1])                               
                    if st.form_submit_button():
                        st.session_state.num += 1
                        if st.session_state.num >= 3:
                            st.session_state.num = 0 
                        placeholder.empty()
                    else:
                        st.stop()


        main()
except TypeError:
    pass
    
















