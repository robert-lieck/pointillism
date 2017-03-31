import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import numpy as np
from PIL import Image
import colorsys
from sklearn.neighbors import NearestNeighbors


def main():

    # img = Image.open('image.jpg').convert('LA')
    img = Image.open('image_3.jpg')

    min_radius = np.ceil(min(img.width, img.height)/200).astype(int)
    max_radius = np.ceil(max(img.width, img.height)/50).astype(int)

    values = []
    centers = []
    radii = []
    for point_idx in range(1000):

        # generate some candidate points
        n_tries = 1000
        n_neighbors = 50
        candidates = []
        for (x, y) in zip(np.random.randint(0, img.width, n_tries), np.random.randint(0, img.height, n_tries)):
            if point_idx > 0:
                # get the nearest neighbors and absolute distances
                nbrs = NearestNeighbors(n_neighbors=min(point_idx, n_neighbors), algorithm='auto').\
                        fit(np.array(centers)).\
                        kneighbors([[x, y]], return_distance=True)
                # print((x, y))
                # print("    ", nbrs[0].flatten())
                # print("    ", np.take(radii, nbrs[1].flatten()))
                # print("    ", nbrs[1].flatten())
                # print("    ", nbrs[0].flatten() / np.take(radii, nbrs[1].flatten()))
                # store point along with cost
                # candidates.append((- np.min(nbrs[0].flatten() / np.take(radii, nbrs[1].flatten())), (x, y)))
                candidates.append(((np.exp( - (nbrs[0].flatten() / np.take(radii, nbrs[1].flatten())) ** 2).mean(), (x, y))))
        if point_idx > 0:
            # select candidate with smallest cost
            # print(candidates)
            (x, y) = sorted(candidates, key=lambda x: x[0])[0][1]
            # print("chose:", (x, y))

        # determine radius via spread of colors
        current_radius = min_radius
        new_radius = current_radius
        variance = 0
        current_mean_color = [0, 0, 0]
        for radius_idx in range(100):
            rgb_colors = []
            hls_colors = []
            for x_idx in range(x-new_radius, x+new_radius+1):
                for y_idx in range(y-new_radius, y+new_radius+1):
                    if (x_idx >= 0 and y_idx >= 0 and x_idx < img.width and y_idx < img.height and np.sqrt((x - x_idx)**2 + (y - y_idx)**2) < new_radius):
                        color = np.array(img.getpixel((x_idx, y_idx)))
                        rgb_colors.append(color)
                        hls_colors.append(colorsys.rgb_to_hls(*list(color/255)))
            # compute mean and spread
            new_mean_color = np.round(np.array(rgb_colors).mean(axis=0)).astype(int) / 255
            spread = np.percentile(np.array(hls_colors)[:,1], [5, 95])
            if radius_idx == 0:
                # properly initialize current_mean_color
                current_mean_color = new_mean_color
            if spread[1] - spread[0] < 0.05 and new_radius < max_radius:
                # accept current values and try larger radius
                current_mean_color = new_mean_color
                current_radius = new_radius
                new_radius = np.ceil(current_radius * 1.1).astype(int)
            else:
                # break and use current values
                print("BREAK({}): {} {} {}".format(point_idx, current_radius, (x, y), current_mean_color, spread))
                break

        values.append((x, y, current_radius, current_mean_color))
        centers.append([x, y])
        radii.append(current_radius)

        fig, ax = plt.subplots()
        # ax.imshow(img)
        for x, y, radius, c in values:
            ax.add_artist(plt.Circle((x, y), radius, color=c))
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.set_aspect(1)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig('img_{}.jpg'.format(str(point_idx).zfill(4), dpi=600, bbox_inches='tight'))
        plt.close()
        fig.clear()

if __name__ == '__main__':
    main()
