from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob 


training_imgs_path = glob.glob("./p1_data/*[1-9].png")
testing_imgs_path = glob.glob("./p1_data/*10.png") 
# print(len(training_imgs_path[90]))
# print(training_imgs_path[90])

# print(imread("./p1_data/1_1.png").flatten().shape)
original_training_faces = np.array([imread(img).flatten() for img in training_imgs_path])
training_faces_label = np.array([img[10] if len(img) == 17 else img[10:12] for img in training_imgs_path])
# print(training_faces_label)
original_testing_faces = np.array([imread(img).flatten() for img in testing_imgs_path])
testing_faces_label = np.array([img[10] if len(img) == 17 else img[10:12] for img in testing_imgs_path])


pca = PCA(n_components = 360).fit(original_training_faces)


# problem 1
# eigen_faces = pca.components_[:4]
# mean_face = pca.mean_
# fig, ax = plt.subplots(1, 5, figsize = (8, 10))
# ax[0].imshow(mean_face.reshape(56,46), cmap = "gray")
# ax[0].set_title("mean face")
# for i in range(1, 5):
# 	ax[i].imshow(eigen_faces[i-1].reshape(56,46), cmap = "gray")
# 	ax[i].set_title(f'eigen face {i}')
# plt.show()



def get_projection_matrix(num):
	space = pca.components_[:num].T
	return np.matmul(np.matmul(space, np.linalg.inv(np.matmul(space.T, space))), space.T)
	# return space*np.linalg.inv(space.T*space)*space.T

def calc_proj(face, num):
	proj_matrix = get_projection_matrix(num)
	# print(proj_matrix.shape)
	return np.dot(proj_matrix, face.T)

dims = [3, 50, 170, 240, 345]
fig, ax = plt.subplots(1, 5, figsize = (8, 10))
face = imread("./p1_data/2_1.png").flatten()
projected_faces = np.array([calc_proj(face, dim) for dim in dims])

# problem 2
# for idx, dim in enumerate(dims):
# 	ax[idx].imshow(projected_faces[idx].reshape(56,46), cmap = "gray")
# 	ax[idx].set_title(f'eigen dim {dim}')
# plt.show()

# problem 3
# for idx, projected_face in enumerate(projected_faces):
# 	mse = mean_squared_error(face, projected_face)
# 	print(f"MSE bettween eigen face {idx} and original face = {mse}")

# problem 4 3-fold cross validation


def get_proj_faces(num, faces):
	return np.array([calc_proj(face, num) for face in faces])

# N = [3, 50, 170]
# K = [1, 3, 5]
# max_acc = 0
# hyperparameter = []
# for k in K:
# 	knn = KNeighborsClassifier(n_neighbors = k)
# 	for n in N:
# 		projected_training_faces = get_proj_faces(n, original_training_faces)
# 		scores = cross_val_score(knn, projected_training_faces, training_faces_label, cv = 3, scoring="accuracy")
# 		if(scores.mean() > max_acc):
# 			max_acc = scores.mean()
# 			hyperparameter = [k, n, max_acc]

# print(f"k = {hyperparameter[0]}, n = {hyperparameter[1]} results max_acc = {hyperparameter[2]}")


# problem 5

best_knn = KNeighborsClassifier(n_neighbors = 1)
best_projected_training_faces = get_proj_faces(50, original_training_faces)
best_projected_testing_faces = get_proj_faces(50, original_testing_faces)
best_knn.fit(best_projected_training_faces, training_faces_label)

predictd_testing_faces_label = best_knn.predict(best_projected_testing_faces)
acc = sum([1 if predicted_label == original_label else 0 for predicted_label, original_label in zip(predictd_testing_faces_label, testing_faces_label)]) / len(testing_faces_label)
print(f"accuracy = {acc*100}%")