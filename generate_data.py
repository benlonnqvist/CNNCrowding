import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def options():
    """Set options for data generation."""
    sizes = [20, 26]                                                    # Text size of letters.
    target_colors = ['rgb(200, 200, 200)', 'rgb(50, 50, 50)']           # RGB-colour of target letters.
    flanker_colors = ['rgb(200, 200, 200)', 'rgb(50, 50, 50)']          # RGB-colour of flanker letters.
    letters = ['A', 'B', 'C', 'E', 'G', 'M', 'Y', 'Q']                  # Target letters.
    flankers = ['A', 'B', 'C', 'E', 'G', 'H', 'M', 'Q', 'S', 'Y']       # Flanker letters.
    flanker_position1 = list(np.arange(0, 360, 18))                     # All possible degree positions of flankers.
    flanker_position2 = list(np.roll(flanker_position1, 10))            # Opposite side position for pair flankers.
    distances = [25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]            # Centre-centre distance of flankers
                                                                        # from the target.
    num_images_per_set = 1                                              # Used for generating multiple sets of
                                                                        # images at once.
    path = r'.\data\grey_bg\greybg.jpg'                                 # Path to grey background image. Set this to
                                                                        # wherever your desired target-flanker
                                                                        # background image is.

    return sizes, target_colors, flanker_colors, letters, flankers, \
            flanker_position1, flanker_position2, distances, num_images_per_set, path


def get_font(size):

    font = ImageFont.truetype('arialbd.ttf', size=size)

    return font


def get_xy(imagex, imagey, position_of_flanker=0, distance_from_target=0, flanker=False):
    """position_of_flanker in degrees of rotation. Function converts to radians on line 44."""

    # Right-side target
    x0 = imagex - int(imagex/4)
    y0 = int(imagey/2)

    if flanker:

        r = distance_from_target
        theta = position_of_flanker
        radians = theta * 0.0174532925

        x = int(x0 + r*np.cos(radians))
        y = int(y0 + r*np.sin(radians))

        return x, y

    return x0, y0


def draw_letter(image, font, letter, xy, color):
    """Function to draw a letter on an image. xy is a tuple of x and y-values."""

    x = xy[0]
    y = xy[1]

    draw = ImageDraw.Draw(image)
    draw.text((x, y), text=letter, fill=color, font=font, align='center')


def squish(path, n_steps, font, letter, target_color, flanker, position_of_flankers,
           distance_from_target, num_flankers, flanker_color, size):
    """Function to draw letters on an image and then logarithmically reduce image
    acuity.
    """

    # Shift letters to be centre-centred. This is a hack and could/should be refactored as sizes[0], sizes[1]
    if size == 26:
        xminus = 9
        yminus = 14

    if size == 20:
        xminus = 7
        yminus = 11

    # Assertion because if this condition does not hold, the code will run, but undesired effects will occur
    assert 0 < num_flankers <= 2, 'num_flankers must be > 0 and <= 2'

    original = Image.open(path)
    # This assertion can be made redundant, and exists only for purposes of confirmation
    assert original.size[1] == 224 and original.size[0] == 224, 'Wrong size: ' + path

    x, y = get_xy(original.size[0], original.size[1])
    flankerx0, flankery0 = get_xy(original.size[0], original.size[1], position_of_flankers[0], distance_from_target,
                                  flanker=True)

    draw_letter(image=original, font=font, letter=letter, xy=(x-xminus, y-yminus), color=target_color)

    # Create images for flankers and draw flankers
    flanker_image = original.copy()
    draw_letter(image=flanker_image, font=font, letter=flanker, xy=(flankerx0-xminus, flankery0-yminus), color=flanker_color)

    if num_flankers == 2:
        flankerx1, flankery1 = get_xy(original.size[0], original.size[1], position_of_flankers[1], distance_from_target,
                                      flanker=True)
        draw_letter(image=flanker_image, font=font, letter=flanker, xy=(flankerx1-xminus, flankery1-yminus), color=flanker_color)

    if n_steps:
        # Create a log-scale array of scalars to distribute a part of the
        # image to a given section of the image and normalise them to sum up to 1.
        # Finally, divide scalars by the first element to receive the size factor
        # for each image. np.geomspace == np.logspace, but start and end are not
        # raised to the base.
        scalars = np.geomspace(1, 0.2, num=n_steps)
        scalars = scalars / scalars.sum()
        scalars /= scalars[0]

        # Multiply scalars-values to get final image x and y squish dimensions.
        scalarsx = scalars * 224
        scalarsx = scalarsx.round().astype(int)
        scalarsy = scalars * 224
        scalarsy = scalarsy.round().astype(int)

        # Invert scalars-arrays to crop such that lowest acuity images have the least crop, etc.
        cropx = scalarsx[::-1]
        cropy = scalarsy[::-1]

        squishable = [flanker_image]
        final = []

        # Artefact from a time when sets of flankers were generated inside this function, but changed due
        # to a large increase in performance. Everything kept inside the single-pass loop to keep control
        # flow equal across experiments.
        for j in range(1):

            # Define array to store our images before pasting on top of each other.
            images = []

            # Resize images to lose quality using the nearest-neighbours technique (implicit Image.resize in PIL 4.2.x).
            for i in range(n_steps):
                squished = squishable[j].resize((scalarsx[i], scalarsy[i]))
                squished = squished.resize((original.size[0], original.size[1]))

                # Assign left, upper, right, lower dimensions to crop by
                left  = int((original.size[0]-cropx[i])/2)
                upper = int((original.size[1]-cropy[i])/2)
                right = int(original.size[0]-left)
                lower = int(original.size[1]-upper)
                # Pre-crop images to appropriate size, dropping unwanted pixels from the edges
                squished = squished.crop((left, upper, right, lower))

                images.append(squished)

            # Define final image on top of which we will paste more clear images
            final.append(images[n_steps-1])

            # Paste images on top of each other
            for i in range(n_steps):
                # Assign left, upper, right, lower dimensions to paste on
                left  = int((original.size[0] - scalarsx[i]) / 2)
                upper = int((original.size[1] - scalarsy[i]) / 2)
                final[j].paste(images[n_steps-1-i], box=(left, upper))

        #original.close()

    return final[0]
    #return flanker_image


def generate_sets():

    sizes, target_colors, flanker_colors, letters, flankers, \
     flanker_position1, flanker_position2, distances, num_images_per_set, path = options()

    letterkeys = {'A': 2, 'B': 3, 'C': 4, 'E': 5, 'G': 6, 'M': 7, 'Y': 8, 'Q': 9}
    for size in sizes:
        font = get_font(size)

        for target_color in target_colors:
            for flanker_color in flanker_colors:
                for letter in letters:
                    for flanker in flankers:
                        for p in range(int(len(flanker_position1))):
                            for distance in distances:
                                directory = r'./directory/{size}/{target_color}/{flanker_color}/{letter}'.format(
                                    size=size, target_color=target_color, flanker_color=flanker_color, letter=letter,
                                    flanker=flanker,
                                    p=flanker_position1[p], distance=distance)

                                for dir in range(len(letters)):
                                    classdir = directory + r'/{classdir}'.format(classdir=dir)
                                    if not os.path.exists(classdir):
                                        os.makedirs(classdir)

                                for i in range(num_images_per_set):

                                    savepath = directory + r'/{classdir}/{flanker}_{p}_{distance}_{i:03d}.jpg'.format(i=i, classdir=letterkeys[letter],
                                                                                                                      flanker=flanker, p=flanker_position1[p],
                                                                                                                      distance=distance)

                                    image = squish(path=path, n_steps=20, font=font, letter=letter,
                                                   target_color=target_color, flanker=flanker,
                                                   position_of_flankers=(flanker_position1[p], flanker_position2[p]),
                                                   distance_from_target=distance, num_flankers=1,
                                                   flanker_color=flanker_color, size=size)

                                    image.save(savepath, optimize=True)


def main():

    generate_sets()


if __name__ == '__main__':
    main()