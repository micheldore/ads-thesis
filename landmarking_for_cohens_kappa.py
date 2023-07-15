import cv2
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import distance
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle, Ellipse
from PIL import Image
import re


LANDMARKS_PER_IMAGE = 12


def calculate_pca(landmarks):
    # Flatten each set of landmarks into a 1D array
    # landmarks = np.array([l.flatten() for l in landmarks])
    landmarks = np.array(landmarks)
    landmarks = np.array([l.flatten() for l in landmarks])

    # Perform PCA
    pca = PCA(n_components=2)
    pca_landmarks = pca.fit_transform(landmarks)
    return pca_landmarks, pca


def calculate_pca_variance_explained(pca):
    var_exp = pca.explained_variance_ratio_
    print(f'Explained variance by PCA1 (PC1): {var_exp[0]}')
    print(f'Explained variance by PCA2 (PC2): {var_exp[1]}')
    total_variance_explained = sum(var_exp)
    print(f'Total explained variance by both PCs: {total_variance_explained}')
    return var_exp, total_variance_explained


def click_and_label(event, x, y, flags, param):
    # Define action on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        points, image, window_name = param
        points.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow(window_name, image)
        print(f'Point {len(points)}: x={x}, y={y}')


def landmark_images(folder):
    # Go through all images and let user landmark them
    landmarks_all_images = []
    reference_distances = []
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            # Read image
            image = cv2.imread(os.path.join(folder, file))
            landmarks_current_image = []
            window_name = f'Landmarking image {file}'
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, click_and_label, [
                                 landmarks_current_image, image, window_name])

            # Show image and wait for 'n' or 'q' key
            while True:
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n') or key == ord('q'):
                    # Check if correct number of landmarks have been placed
                    if len(landmarks_current_image) == LANDMARKS_PER_IMAGE:
                        break
                    else:
                        print(
                            f'Error: placed {len(landmarks_current_image)} landmarks, expected {LANDMARKS_PER_IMAGE}')
                elif key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    return None, None

            landmarks_all_images.append(landmarks_current_image)

            # Measure reference distance
            print('Please select two points to measure the reference distance.')
            reference_points = []
            cv2.setMouseCallback(window_name, click_and_label, [
                                 reference_points, image, window_name])
            while len(reference_points) < 2:
                cv2.imshow(window_name, image)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    cv2.destroyAllWindows()
                    return None, None
            reference_distances.append(np.linalg.norm(
                np.array(reference_points[0]) - np.array(reference_points[1])))

            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    return landmarks_all_images, reference_distances


def parse_tps_file(file):
    with open(file, 'r') as f:
        contents = f.read()

    blocks = contents.split('LM=')

    filenames = []
    landmarks = []
    for block in blocks:
        lines = [line for line in block.split(
            '\n') if line]  # Remove empty lines
        if not lines:  # Check if all lines were empty
            continue

        num_landmarks = int(lines[0])
        coordinates = lines[1:num_landmarks+1]
        image_name_line = [line for line in lines if 'IMAGE=' in line][0]
        image_name = image_name_line.split('=')[-1]

        filenames.append(image_name)

        # Parse the coordinates and append to the landmarks list
        landmarks_block = []
        for coordinate in coordinates:
            x, y = coordinate.split(' ')
            landmarks_block.append([float(x), float(y)])

        # Check if landmarks_block is not empty before appending
        if landmarks_block:
            landmarks.append(landmarks_block)

    return filenames, landmarks


def calculate_kappa(labels1, labels2):
    # Flatten the labels and calculate Cohen's kappa
    labels1_flat = labels1.reshape(-1)
    labels2_flat = labels2.reshape(-1)
    kappa = cohen_kappa_score(labels1_flat, labels2_flat)
    return kappa


def calculate_error_rates(labels1, labels2):
    error_rates = np.sqrt(np.sum(np.square(labels1 - labels2), axis=-1))
    return error_rates


def plot_landmarks(image, landmarks, error_rates):
    # Check if the image has a 4th channel (alpha)
    if image.shape[2] == 4:
        # If it does, create a new 3-channel image and fill the 4th channel with white
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        alpha_channel = image[:, :, 3]
        mask = alpha_channel == 0  # Wherever alpha channel is 0, make the image white
        # Assigning white to the masked region
        rgb_image[mask] = [255, 255, 255]
    else:
        # If the image does not have a 4th channel, just convert it to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_image)

    for i, (landmark, error_rate) in enumerate(zip(landmarks, error_rates)):
        # With transparent fill blue circle
        plt.gca().add_patch(
            Circle((landmark[0], landmark[1]), error_rate, color='b', fill=True, alpha=0.7))
        plt.scatter(landmark[0], landmark[1], color='r', s=2)
    plt.show()


def calculate_mahalanobis_distance(landmarks, consensus):
    # Flatten each set of landmarks into a 1D array
    landmarks = [l.flatten() for l in landmarks]
    consensus = consensus.flatten()

    cov = np.cov(np.array(landmarks).T)
    inv_cov = np.linalg.inv(cov)
    m_distances = [distance.mahalanobis(
        l, consensus, inv_cov) for l in landmarks]
    return m_distances


def calculate_mahalanobis_distance_tps_file(landmarks, consensus):
    # Convert each set of landmarks into a 1D numpy array
    landmarks = [np.array([coord for point in l for coord in point])
                 for l in landmarks]
    consensus = np.array([coord for point in consensus for coord in point])

    # Convert to 2D array and transpose
    landmarks_array = np.vstack(landmarks).T
    cov = np.cov(landmarks_array)
    inv_cov = np.linalg.inv(cov)
    m_distances = [distance.mahalanobis(
        l, consensus, inv_cov) for l in landmarks]
    return m_distances


def plot_mahalanobis_distances(folder, landmarks, m_distances, pca_landmarks):
    fig, ax = plt.subplots()

    # Get mean and standard deviations
    pca_mean = np.mean(pca_landmarks, axis=0)
    pca_std = np.std(pca_landmarks, axis=0)

    # Plot the first standard deviations ellipse
    ellipse = Ellipse(xy=pca_mean, width=pca_std[0] * 2,
                      height=pca_std[1] * 2, fill=False, color='r')

    # Plot the second standard deviations ellipse
    ellipse2 = Ellipse(xy=pca_mean, width=pca_std[0] * 4,
                       height=pca_std[1] * 2, fill=False, color='r')
    ax.add_patch(ellipse)
    ax.add_patch(ellipse2)

    ax.scatter(pca_landmarks[:, 0], pca_landmarks[:, 1], alpha=0)

    # Annotate with Mahalanobis distances
    for i, (d, file) in enumerate(zip(m_distances, os.listdir(folder))):
        if file.endswith(".jpg") or file.endswith(".png"):
            x, y = pca_landmarks[i, 0], pca_landmarks[i, 1]
            img = cv2.cvtColor(cv2.imread(
                os.path.join(folder, file)), cv2.COLOR_BGR2RGB)

            try:
                Image.open(os.path.join(folder, file))
            except IOError:
                print('Could not read:', file, '- it\'s ok, skipping.')

            img = cv2.resize(img, (400, 400))
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            imagebox = OffsetImage(img, zoom=0.1)
            ab = AnnotationBbox(imagebox, (x, y))
            ax.add_artist(ab)

    plt.show()


def load_or_label_images(folder):
    if os.path.exists('landmarks1.npy') and \
       os.path.exists('landmarks2.npy') and \
       os.path.exists('reference_distances1.npy') and \
       os.path.exists('reference_distances2.npy'):
        landmarks1 = np.load('landmarks1.npy')
        landmarks2 = np.load('landmarks2.npy')
        reference_distances1 = np.load('reference_distances1.npy')
        reference_distances2 = np.load('reference_distances2.npy')
    else:
        print('Label files not found. Please label the images.')
        landmarks1, reference_distances1 = landmark_images(folder)
        if landmarks1 is None:
            return

        print('Please label the images a second time.')
        landmarks2, reference_distances2 = landmark_images(folder)
        if landmarks2 is None:
            return

        np.save('landmarks1.npy', landmarks1)
        np.save('landmarks2.npy', landmarks2)
        np.save('reference_distances1.npy', reference_distances1)
        np.save('reference_distances2.npy', reference_distances2)

    return landmarks1, reference_distances1, landmarks2, reference_distances2


def main():
    folder = 'data/cohens_kappa'
    landmarks1, reference_distances1, landmarks2, reference_distances2 = load_or_label_images(
        folder)

    tps_file = 'final.tps'
    filenames, landmarks1 = parse_tps_file(tps_file)

    # Calculate the consensus shape
    consensus = np.mean(landmarks1, axis=0)

    # Calculate the Mahalanobis distances
    m_distances = calculate_mahalanobis_distance_tps_file(
        landmarks1, consensus)

    # Calculate PCA
    pca_landmarks, pca = calculate_pca(landmarks1)

    # Plot the Mahalanobis distances
    plot_mahalanobis_distances(
        'data/post_processing/final_resized/cropped', landmarks1, m_distances, pca_landmarks)

    # Calculate PCA
    pca_landmarks, pca = calculate_pca(landmarks1)

    # Calculate and print variance explained
    calculate_pca_variance_explained(pca)

    return

    kappa = calculate_kappa(landmarks1, landmarks2)
    print(f'Cohen\'s kappa: {kappa}')

    error_rates = calculate_error_rates(landmarks1, landmarks2)
    print(f'Error rates: {error_rates}')

    file_list = sorted(os.listdir(folder))
    file_list = sorted([f for f in os.listdir(folder) if f.endswith(
        ".jpg") or f.endswith(".png")], key=lambda x: int(''.join(filter(str.isdigit, x))))

    n_th_image = 4
    first_image_error_rates = error_rates[n_th_image]

    # Usage
    image = cv2.imread(os.path.join(
        folder, file_list[n_th_image]), cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 3:  # if the image doesn't have alpha channel, convert it from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plot_landmarks(image, landmarks1[n_th_image-2], first_image_error_rates)

    # Calculate consensus shape by averaging the two sets of landmarks
    consensus_shape = np.mean(landmarks1, axis=0)

    #  Calculate Mahalanobis distance to the consensus shape for each image
    m_distances = calculate_mahalanobis_distance(landmarks1, consensus_shape)

    # Calculate PCA
    pca_landmarks, pca = calculate_pca(landmarks1)

    # Plot Mahalanobis distances
    plot_mahalanobis_distances(folder, landmarks1, m_distances, pca_landmarks)


if __name__ == '__main__':
    main()
