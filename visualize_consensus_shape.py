import pandas as pd
import matplotlib.pyplot as plt

# Data from the consensus shape, taken from MorphoJ
data = {
    'Lmk.': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Axis 1 (x)': [-0.10063292, 0.03211212, 0.05966018, -0.06580345, -0.11484816, -0.14027212, 0.01662462, -0.04646961, -0.15503739, 0.13996176, 0.13968861, 0.23501636],
    'Axis 2 (y)': [-0.46045740, -0.51079875, -0.21938230, -0.00468729, 0.00733734, 0.15830178, 0.06384852, 0.14225758, 0.43048122, 0.05444769, 0.09468141, 0.24397020]
}

# Plot the average shape data
plt.figure(figsize=(10, 10))
plt.scatter(data['Axis 1 (x)'], data['Axis 2 (y)'],
            c='blue')  # Plot the points
plt.plot(data['Axis 1 (x)'], data['Axis 2 (y)'],
         'r-')  # Connect the points with a line

# Annotate the landmarks
for i, txt in enumerate(data['Lmk.']):
    plt.annotate(txt, (data['Axis 1 (x)'][i], data['Axis 2 (y)'][i]))


def parse_tps_data(tps_data):
    data = []
    for block in tps_data.split('LM=12')[1:]:
        lines = block.strip().split('\n')
        coords = [list(map(float, line.split())) for line in lines[:12]]
        image = lines[12].split('=')[1].strip()
        id_ = int(lines[13].split('=')[1].strip())
        for i, (x, y) in enumerate(coords, start=1):
            data.append({'ID': id_, 'Image': image,
                        'Landmark': i, 'X': x, 'Y': y})
    return pd.DataFrame(data)


# Read the tps file into a string
tps_data = open('final.tps').read()

df = parse_tps_data(tps_data)

# Plot the data in the same plot as the average shape, with a alpha=0.5
plt.scatter(df['X'], df['Y'], c='red', alpha=0.5)
for i, txt in enumerate(df['Landmark']):
    plt.annotate(txt, (df['X'][i], df['Y'][i]))


plt.xlabel('Axis 1 (x)')
plt.ylabel('Axis 2 (y)')
plt.title('Average Shape')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
