import copy
import matplotlib.pyplot as plt

from random import sample
from scipy.spatial import cKDTree
from time import time
from model_loader import *
from matrix_utils import *


class ICP:
    @staticmethod
    def align(source_model, dest_model, max_it, with_error=True):
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

        if with_error:
            plt.plot(range(0, it), errors)
            plt.xlabel("Iterations")
            plt.ylabel("Magnitude of error")
            plt.show()

        return new_model

    @staticmethod
    def subsample_align(source_model, dest_model, max_it, sample_size, with_error=True):
        # Get a random subsample of source and generate a correspondence for it

        tree = cKDTree(np.reshape(dest_model.points, (len(dest_model.points), 3)))

        new_model = copy.deepcopy(source_model)

        it = 0
        errors = []

        while it < max_it:
            subsample = sample(source_model.points, sample_size)
            correspondence_list = []

            for p in subsample:
                d, i = tree.query(np.reshape(p, (1, 3)), k=1, p=2)
                closest = dest_model.points[i[0]]
                correspondence_list.append(closest)

            print("Beginning iterative closest point with " + str(sample_size) + " samples and " + str(
                max_it - it) + " remaining iterations...")

            r, t = ICP.calculate_translation(subsample, correspondence_list)

            new_model.apply_transformation(r, t)

            errors.append(ICP.calculate_error(subsample, correspondence_list, r, t))

            it += 1

        # Plot the error of error over iterations

        if with_error:
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
            product_matrix = normalised_points_of_p[i] * (normalised_points_of_q[i].transpose())
            sum_of_products_matrix = sum_of_products_matrix + product_matrix

        # Get our orthonormal set and derive the rotation matrix. Remember: RR^T = 1!

        u, w, vh = np.linalg.svd(sum_of_products_matrix)

        rotation_matrix = vh.transpose() @ u.transpose()

        # Our formula: t = p - Rq

        translation_vector = mean_of_p - rotation_matrix.dot(mean_of_q)

        return rotation_matrix, translation_vector


def question_1():
    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun045_v2.ply")

    max_it = 50

    before = time()
    m3 = ICP.align(m1, m2, max_it)
    after = time()

    print("ICP took " + str(after - before))

    ModelLoader.save("output/q1.ply", m3)


def question_2():
    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun000_v2.ply")

    # Rotate M2 by some value in radians...

    perturb_deg = 5.0 * (pi / 180.0)
    perturb = MatrixUtils.construct_rotation_matrix(perturb_deg, np.array([0, 0, 1]))

    m2.apply_transformation(perturb)

    # Try to align...

    max_it = 50

    m3 = ICP.align(m2, m1, max_it)

    ModelLoader.save("output/q2-m1.ply", m1)
    ModelLoader.save("output/q2-m2.ply", m2)
    ModelLoader.save("output/q2-m3.ply", m3)


def question_3():
    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun000_v2.ply")

    # Add white noise to M2...

    m2.apply_zero_mean_noise(20)

    # Rotate M2 by some value in radians...

    perturb_deg = 20.0 * (pi / 180.0)
    perturb = MatrixUtils.construct_rotation_matrix(perturb_deg, np.array([0, 0, 1]))

    m2.apply_transformation(perturb)

    max_it = 50

    m3 = ICP.align(m2, m1, max_it)

    ModelLoader.save("output/q3-m1.ply", m1)
    ModelLoader.save("output/q3-m2.ply", m2)
    ModelLoader.save("output/q3-m3.ply", m3)


def question_4():
    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun045_v2.ply")

    max_it = 50
    sample_size = 1024

    before = time()
    m3 = ICP.subsample_align(m2, m1, max_it, sample_size)
    after = time()

    print("ICP with subsampling took " + str(after - before))

    ModelLoader.save("output/q4.ply", m3)


def question_5():
    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun045_v2.ply")
    m3 = ModelLoader.load("bunny/bun090_v2.ply")
    m4 = ModelLoader.load("bunny/bun180_v2.ply")
    m5 = ModelLoader.load("bunny/bun270_v2.ply")
    m6 = ModelLoader.load("bunny/bun315_v2.ply")

    m21 = ICP.align(m2, m1, 20)
    m31 = ICP.align(m3, m1, 20)
    m41 = ICP.align(m4, m1, 20)
    m51 = ICP.align(m5, m1, 20)
    m61 = ICP.align(m6, m1, 20)

    ModelLoader.save("output/q5-m21.ply", m21)
    ModelLoader.save("output/q5-m31.ply", m31)
    ModelLoader.save("output/q5-m41.ply", m41)
    ModelLoader.save("output/q5-m51.ply", m51)
    ModelLoader.save("output/q5-m61.ply", m61)


if __name__ == "__main__":
    question_5()
