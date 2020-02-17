import cv2
import numpy as np
import pickle

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

digits = cv2.imread("digits.png", 0)
test_digits = cv2.imread("test_digits.png", 0)

rows = np.vsplit(digits, 50)
cells = []

for row in rows:
    row_cells = np.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)

cells = np.array(cells, dtype=np.float32)

k = np.arange(10)
cells_labels = np.repeat(k, 250)

test_digits = np.vsplit(test_digits, 50)
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)

test_cells = np.array(test_cells, dtype=np.float32)

test_img = cv2.imread("2.png", 0)
# test_img = np.vsplit(test_img, 2) # Split if there are more number of testing images in just one image.
test_cell = test_img.flatten()

# for d in test_img:    # Flatten and append for more than one test_image.
#     d = d.flatten()
#     test_cell.append(d)

test_cell = np.array(test_cell, dtype=np.float32)
test_cell = np.reshape(test_cell, (1, test_cell.shape[0]))  #Reshape while testing for only one image.

#KNN 
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cell, k=1)
print(result)
