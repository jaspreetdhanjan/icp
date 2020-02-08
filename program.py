import numpy as np

from scipy.spatial import cKDTree
from random import sample
import matplotlib.pyplot as plt
import copy


class Face:
    def __init__(self, size, p0, p1, p2):
        self.size = size
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return str(self.size) + " " + str(self.p0) + " " + str(self.p1) + " " + str(self.p2) + " \n"


class Model:
    def __init__(self, header, points, faces):
        self.header = header
        self.points = points
        self.faces = faces

    def apply_transformation(self, rotation_matrix, translation_vector):
        for i in range(0, len(self.points)):
            self.points[i] = (rotation_matrix.dot(self.points[i])) - translation_vector


class ICP:
    @staticmethod
    def align(source_model, dest_model, max_it):
        # Construct KD-tree from target points, so that we can search through it easy!

        correspondence_list = []

        tree = cKDTree(np.reshape(dest_model.points, (len(dest_model.points), 3)))

        for p in source_model.points:
            # Numpy is really awkward, converting this point to a column vector
            # Query and append to our correspondence list

            d, i = tree.query(np.reshape(p, (1, 3)), k=1, p=2)

            closest = dest_model.points[i[0]]
            correspondence_list.append(closest)

        # Create a copy of our model, we don't want to modify the existing point list

        new_model = copy.deepcopy(source_model)

        # Run ICP and apply translation

        it = 0
        errors = []

        while it < max_it:
            print("Beginning iterative closest point with " + str(max_it - it) + " remaining iterations...")

            r, t = ICP.calculate_translation(new_model.points, correspondence_list)

            new_model.apply_transformation(r, t)

            errors.append(ICP.calculate_error(new_model.points, correspondence_list, r, t))

            it += 1

        # Plot the error of error over iterations

        plt.plot(range(0, it), errors)
        plt.xlabel("Iterations")
        plt.ylabel("Magnitude of error")
        plt.show()

        return new_model

    @staticmethod
    def calculate_error(p_points, q_points, rotation_matrix, translation_vector):
        error_sum = 0

        for i in range(0, len(p_points)):
            error_sum += pow(np.linalg.norm(p_points[i] - rotation_matrix.dot(q_points[i]) - translation_vector), 2)

        return error_sum

    @staticmethod
    def calculate_translation(p_points, q_points):
        assert len(p_points) == len(q_points)
        sample_size = len(p_points)

        # Normalise to barycentric form

        normalised_points_of_p = []
        normalised_points_of_q = []

        mean_of_p = np.array([[0], [0], [0]])
        mean_of_q = np.array([[0], [0], [0]])

        for i in range(0, sample_size):
            mean_of_p = mean_of_p + p_points[i]
            mean_of_q = mean_of_q + q_points[i]

        mean_of_p = mean_of_p / sample_size
        mean_of_q = mean_of_q / sample_size

        for i in range(0, sample_size):
            normalised_points_of_p.append(p_points[i] - mean_of_p)
            normalised_points_of_q.append(q_points[i] - mean_of_q)

        # Multiply normalised barycenters together. Add to matrix.

        sum_of_products_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        for i in range(0, sample_size):
            product_matrix = normalised_points_of_p[i] * normalised_points_of_q[i].transpose()
            sum_of_products_matrix = sum_of_products_matrix + product_matrix

        # Get our orthonormal set and derive the rotation matrix. Remember: RR^T = 1!

        u, w, vh = np.linalg.svd(sum_of_products_matrix)

        rotation_matrix = vh.transpose() @ u.transpose()

        # Our formula: t = p - Rq

        translation_vector = mean_of_p - rotation_matrix.dot(mean_of_q)

        return rotation_matrix, translation_vector


class ModelLoader:
    @staticmethod
    def load(file_name):
        header = []
        points = []
        faces = []

        is_header = True

        with open(file_name, "r") as file_in:
            while True:
                line = file_in.readline()

                if not line:
                    break

                line = line.strip()

                if is_header:
                    header.append(line)

                    if line == "end_header":
                        is_header = False

                else:
                    tokens = line.split(" ", 4)

                    if len(tokens) == 3:
                        points.append(np.array([[float(tokens[0])], [float(tokens[1])], [float(tokens[2])]]))
                    elif len(tokens) == 4:
                        faces.append(Face(int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])))

        print("Loaded model -> " + file_name + " - with " + str(len(points)) + " points")

        return Model(header, points, faces)

    @staticmethod
    def save(file_name, model):
        with open(file_name, "w+") as file_out:
            for h in model.header:
                file_out.write(h + "\n")

            for p in model.points:
                line = str(p[0, 0]) + " " + str(p[1, 0]) + " " + str(p[2, 0]) + "\n"
                file_out.write(line)

            for f in model.faces:
                file_out.write(str(f))

        print("Saved model -> " + file_name)


def main():
    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun045_v2.ply")

    max_it = 5

    m3 = ICP.align(m2, m1, max_it)
    ModelLoader.save("output/q1.ply", m3)

    # for it in range(1, max_it):
    #     ModelLoader.save("output/" + str(it) + "-q1.ply", ICP.align(m2, m1, sample_size, it))

    # for it in range(0, 20):
    #    m3 = ICP.align(m2, m1, sample_size, it)

    #    ModelLoader.save("output/" + str(it) + "-q1.ply", m3)


if __name__ == "__main__":
    main()
