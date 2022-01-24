from PIL import Image
import matplotlib.pyplot as plot
from scipy import signal

source = Image.open('green circles.png')
image = source.convert('RGB')
width, height = image.size

array_x = []
array_y = []
for i in range(0, width):
    sum = 0
    for j in range(0, height):
        red, green, blue = image.getpixel((i, j))
        sum += green
    array_x.append(i / 22)
    array_y.append(sum / height)

plot.plot(array_x, array_y)
plot.title('Функция интенсивности')
plot.xlabel('r, мм')
plot.show()

rv_x = []
rv_y = []
fl = 160
for i in range(160, 900 - 1):
    if (i < fl):
        continue

    if (array_y[i] - array_y[i + 1] > 3):
        mx = array_y[i]
        mn = 0
        for j in range(i, 900 - 1):
            if (array_y[j + 1] - array_y[j] > 3):
                mn = array_y[j]
                fl = j
                rv_x.append(j / 22)
                break
        rv_y.append((mx - mn) / (mx + mn))

rv_y = signal.savgol_filter(rv_y, len(rv_y) - 1, 3)

plot.plot(rv_x, rv_y)
plot.title('Функция видимости')
plot.xlabel('r, мм')
plot.show()