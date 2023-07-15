import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import dotenv

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
    },
    # {
    #     "name": "Is de foto te gebruiken?",
    #     "options": ["Ja", "Nee"],
    #     "column": "usable"
    # },
]
folder = ''


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
        for key, value in row_data.items():
            df.loc[df[id_col] == data[0], key] = value
    else:
        df = df.append(row_data, ignore_index=True)

    return df


def save_to_csv(folder, filename, data):
    global df

    df = update_or_add_row(df, data)

    # save_to_dropbox(filename, data, os.getenv('DROPBOX_ACCESS_TOKEN'))

    df.to_csv(os.path.join(folder, filename), index=False)


def on_stop_click():
    global folder
    global current_image
    global choices
    save_to_csv(folder, 'labels.csv', [
                get_filename_from_filepath(current_image)] + choices)
    window.destroy()


def on_radio_click(radio_var):
    global choices
    choices = [rv.get() for rv in radio_vars]


def on_next_click():
    global folder
    global current_image
    global choices
    global image_files
    global img_label
    global radio_vars
    global visited_images

    save_to_csv(folder, 'labels.csv', [
        get_filename_from_filepath(current_image)] + choices)
    visited_images.append(current_image)
    image_files = image_files[1:]
    if len(image_files) > 0:
        current_image = image_files[0]
        img = cv2.imread(current_image)
        if img is not None:  # Check if the image has been read successfully
            img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            img_label.config(image=img)
            img_label.image = img
            for i, rv in enumerate(radio_vars):
                data = df.loc[df['id'] == os.path.splitext(
                    current_image)[0], df.columns[i + 1]].values
                if len(data) > 0:
                    on_radio_click(rv)
                    rv.set(data[0])
                else:
                    on_radio_click(rv)
                    rv.set(rv.choices[0])  # Reset to the first option
    else:
        window.destroy()


def on_prev_click():
    global folder
    global current_image
    global choices
    global image_files
    global img_label
    global radio_vars
    global visited_images
    if len(visited_images) > 0:
        current_image = visited_images.pop()
        image_files.insert(0, current_image)
        img = cv2.imread(current_image)
        if img is not None:  # Check if the image has been read successfully
            img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            img_label.config(image=img)
            img_label.image = img
            for i, rv in enumerate(radio_vars):
                data = df.loc[df['id'] == os.path.splitext(
                    current_image)[0], df.columns[i + 1]].values
                if len(data) > 0:
                    on_radio_click(rv)
                    rv.set(data[0])
                else:
                    on_radio_click(rv)
                    rv.set(rv.choices[0])  # Reset to the first option


def get_extension_from_file_name(file_name):
    return os.path.splitext(file_name)[1]


def get_file_name_from_image_id(image_id):
    # read the image_names.csv file
    image_names = pd.read_csv(os.path.join(folder, 'image_names.csv'))
    file_name = image_names.loc[image_names['id']
                                == image_id, 'file_name'].values[0]
    file_name = str(image_id) + get_extension_from_file_name(file_name)
    return file_name


window = tk.Tk()
window.title("Labellen van geiten afbeeldingen")

folder = os.getcwd() + '/selected_images'


df = pd.DataFrame(columns=['id'] + [label['column'] for label in labels])

csv_path = os.path.join(folder, 'labels.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df.to_csv(csv_path, index=False)


last_labeled_image_index = -1

image_files = get_image_files(folder)
print('This is how many images are in the folder: ' + str(len(image_files)))

if last_labeled_image_index >= 0:
    visited_images = image_files[:last_labeled_image_index]
    image_files = image_files[last_labeled_image_index:]
else:
    visited_images = []

for img_id in df['id']:
    # assuming .jpg extension, change accordingly
    img_file = os.path.join(folder, get_file_name_from_image_id(img_id))

    if img_file in image_files:
        last_labeled_image_index = max(
            last_labeled_image_index, image_files.index(img_file))


if len(image_files) > 0:
    if last_labeled_image_index >= 0:
        current_image = image_files[last_labeled_image_index]
    else:
        current_image = image_files[0]

    img = cv2.imread(current_image)
    img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(image=Image.fromarray(img))
    img_label = tk.Label(window, image=img)
    img_label.pack()

    choices = [l["options"][0] for l in labels]
    radio_vars = []

    for label in labels:
        frame = ttk.Frame(window)
        frame.pack(side=tk.TOP, pady=5)
        ttk.Label(frame, text=label["name"]).pack(side=tk.LEFT)
        radio_var = tk.StringVar(frame)
        radio_var.set(label["options"][0])
        radio_var.choices = label["options"]
        radio_vars.append(radio_var)
        for option in label["options"]:
            rb = ttk.Radiobutton(frame, text=option, variable=radio_var,
                                 value=option, command=lambda rv=radio_var: on_radio_click(rv))
            rb.pack(side=tk.LEFT)

    visited_images = []

    prev_button = ttk.Button(window, text="Vorige", command=on_prev_click)
    prev_button.pack(side=tk.LEFT, padx=10, pady=10)

    stop_button = ttk.Button(window, text="Stop", command=on_stop_click)
    stop_button.pack(side=tk.LEFT, padx=10, pady=10)

    next_button = ttk.Button(window, text="Volgende", command=on_next_click)
    next_button.pack(side=tk.RIGHT, padx=10, pady=10)

    window.mainloop()
else:
    print("Geen afbeelding in deze map.")
