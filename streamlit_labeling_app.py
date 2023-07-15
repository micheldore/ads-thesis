from filestack import Client
import os
import cv2
import pandas as pd
import tempfile
import dropbox
import streamlit as st
from PIL import Image
from pathlib import Path
import dotenv
from dropbox import DropboxOAuth2FlowNoRedirect

dotenv.load_dotenv()


labels = [
    {
        "name": "Is het een bok of een geit?",
        "options": ["Bok", "Geit", "Weet niet"],
        "column": "buck"
    },
    {
        "name": "Is de foto realistisch genoeg?",
        "options": ["Ja", "Nee", "Weet niet"],
        "column": "realistic"
    }
]
folder = ''

client = Client(os.getenv("FILESTACK_API_KEY"))


def save_to_filestack():
    global df, client

    # Save dataframe to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        df.to_csv(temp_file, index=False)
        temp_file_path = temp_file.name

    with open(temp_file_path, 'rb') as src:
        new_filelink = client.upload(filepath=temp_file_path)

    # Remove the temporary file
    os.unlink(temp_file_path)


def get_filename_from_filepath(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_image_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG',
                                                                               '.gif', '.GIF', '.tif', '.tiff', '.TIF', '.TIFF',
                                                                               '.bmp', '.BMP', '.ico', '.ICO', '.eps', '.EPS',))]


def update_or_add_row(df, data):
    col_names = df.columns
    id_col = col_names[0]
    row_data = {col: val for col, val in zip(col_names, data)}

    if data[0] in df[id_col].values or int(data[0]) in df[id_col].values:
        df.loc[df[id_col] == data[0]] = row_data
    else:
        new_row = pd.DataFrame(row_data, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def save_to_csv(folder, filename, data):
    global df

    df = update_or_add_row(df, data)

    save_to_filestack()

    df.to_csv(os.path.join(folder, filename), index=False)


folder = os.getcwd() + '/selected_images'

df = pd.DataFrame(columns=['id'] + [label['column'] for label in labels])

csv_path = os.path.join(folder, 'labels.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df.to_csv(csv_path, index=False)

image_files = get_image_files(folder)

if "image_files" not in st.session_state:
    st.session_state.image_files = get_image_files(folder)

if len(st.session_state.image_files) > 0:
    current_image = st.session_state.image_files[0]
    img = Image.open(current_image)
    st.title("Labelen van geiten afbeeldingen")
    main_image = st.image(img, width=350)

    choices = {label["name"]: st.selectbox(
        label["name"], label["options"]) for label in labels}

    if st.button("Save"):
        save_to_csv(folder, 'labels.csv', [get_filename_from_filepath(
            current_image)] + [choices[label["name"]] for label in labels])

        st.session_state.image_files = st.session_state.image_files[1:]
        if len(st.session_state.image_files) > 0:
            current_image = st.session_state.image_files[0]
            img = Image.open(current_image)
            main_image.image(img, width=350)
        else:
            st.write("Geen afbeelding in deze map.")
else:
    st.write("Geen afbeelding in deze map.")
