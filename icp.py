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
    def align_point_to_plane(source_model, dest_model, max_it, with_error=True):
        correspondence_list = []

        tree = cKDTree(np.reshape(dest_model.points, (len(dest_model.points), 3)))
        normals = ICP.get_normals_map(dest_model.points)

        for p in source_model.points:
            d, i = tree.query(np.reshape(p, (1, 3)), k=1, p=2)

            closest = dest_model.points[i[0]]
            correspondence_list.append((closest, normals[closest]))

        # Create a copy of our model, we don't want to modify the existing point list

        new_model = copy.deepcopy(source_model)

        # Run ICP and apply translation

        it = 0
        # errors = []

        while it < max_it:
            print("Beginning iterative closest point with " + str(max_it - it) + " remaining iterations...")

            r, t = ICP.calculate_translation(new_model.points, correspondence_list)

            new_model.apply_transformation(r, t)

            # errors.append(ICP.calculate_error(new_model.points, correspondence_list, r, t))

            it += 1

        # Plot the error of error over iterations

        # if with_error:
        #    plt.plot(range(0, it), errors)
        #    plt.xlabel("Iterations")
        #    plt.ylabel("Magnitude of error")
        #     plt.show()

        return new_model

    @staticmethod
    def get_normals(model):
        tree = cKDTree(np.reshape(model.points, (len(model.points), 3)))

        # For each point in the mesh, get its three closest neighbours

        num_neighbours = 3

        d, i = tree.query(np.reshape(model.points, (len(model.points), 3)), num_neighbours)

        num_to_show = len(i)
        normals = ICP.lstsq_plane_fitting(model.points, i[:num_to_show], 3)

        normal = ICP.normalize_v3(normals)
        return normal

    @staticmethod
    def get_normals_map(model):
        tree = cKDTree(np.reshape(model.points, (len(model.points), 3)))

        # For each point in the mesh, get its three closest neighbours

        num_neighbours = 3

        d, i = tree.query(np.reshape(model.points, (len(model.points), 3)), num_neighbours)

        num_to_show = len(i)
        normals = ICP.lstsq_plane_fitting_map(model.points, i[:num_to_show], 3)

        # normal = ICP.normalize_v3(normals)
        return normals

    @staticmethod
    def normalize_v3(arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    @staticmethod
    def lstsq_plane_fitting(total, indices, k):
        # Adapted from tutorial:https://github.com/smartgeometry-ucl/COMPM080-Tutorials-2020/blob/master/tutorials/2_normal_estimation/python/normal%20estimation.ipynb

        normals = np.zeros((len(indices), 3))

        for point in range(len(indices)):
            dots = np.zeros((k, 3))

            for nei in range(k):
                loc = indices[point, nei]
                xyz = [total[loc][0], total[loc][1], total[loc][2]]
                dots[nei, :] = xyz

            (a, b, c), resid, rank, s = np.linalg.lstsq(dots[:, :3], np.ones_like(dots[:, 2]))
            normal = (a, b, c)
            nn = np.linalg.norm(normal)
            normal = normal / nn

            normals[point, :] = normal[:3]

            if point % 1000 == 0:
                print("Accomplishment of: " + str(point) + " points out of: " + str(len(indices)))

        return normals

    @staticmethod
    def lstsq_plane_fitting_map(total, indices, k):
        normal_point_map = {}

        for point in range(len(indices)):
            dots = np.zeros((k, 3))

            for nei in range(k):
                loc = indices[point, nei]
                xyz = [total[loc][0], total[loc][1], total[loc][2]]
                dots[nei, :] = xyz

            (a, b, c), resid, rank, s = np.linalg.lstsq(dots[:, :3], np.ones_like(dots[:, 2]))
            normal = (a, b, c)
            nn = np.linalg.norm(normal)
            normal = normal / nn

            normal_point_map[point] = normal[:3]

            if point % 1000 == 0:
                print("Accomplishment of: " + str(point) + " points out of: " + str(len(indices)))

        return normal_point_map

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
    # Basic point-to-point ICP alignment

    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun045_v2.ply")

    max_it = 50

    before = time()
    m3 = ICP.align(m1, m2, max_it)
    after = time()

    print("ICP took " + str(after - before))

    ModelLoader.save("output/q1.ply", m3)


def question_2():
    # Adding rotation to perturb

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
    # Adding noise to perturb

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
    # Subsampling

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
    # Global registration

    m1 = ModelLoader.load("bunny/bun000_v2.ply")
    m2 = ModelLoader.load("bunny/bun045_v2.ply")
    m3 = ModelLoader.load("bunny/bun090_v2.ply")
    m4 = ModelLoader.load("bunny/bun180_v2.ply")
    m5 = ModelLoader.load("bunny/bun270_v2.ply")
    m6 = ModelLoader.load("bunny/bun315_v2.ply")

    m21 = ICP.align(m1, m2, 20)
    m31 = ICP.align(m2, m3, 20)
    m41 = ICP.align(m3, m4, 20)
    m51 = ICP.align(m4, m5, 20)
    m61 = ICP.align(m5, m6, 20)
    m71 = ICP.align(m5, m1, 20)

    ModelLoader.save("output/q5-m21.ply", m21)
    ModelLoader.save("output/q5-m31.ply", m31)
    ModelLoader.save("output/q5-m41.ply", m41)
    ModelLoader.save("output/q5-m51.ply", m51)
    ModelLoader.save("output/q5-m61.ply", m61)
    ModelLoader.save("output/q5-m71.ply", m71)


def question_6():
    # Point to plane

    m1 = ModelLoader.load("bunny/bun000_v2.ply")

    # Output with normals so we can shade

    # m1_normals = ICP.get_normals(m1)
    # model_with_normals = ModelWithNormals(m1, m1_normals)

    # ModelLoader.save_to_obj("output/q6_bun000_v2_with_normals.obj", model_with_normals)

    # Perform point to plane

    m2 = ModelLoader.load("bunny/bun045_v2.ply")

    max_it = 50

    m3 = ICP.align_point_to_plane(m2, m1, max_it)

    ModelLoader.save_to_obj("output/q6_plane_alignment.obj", m3)


if __name__ == "__main__":
    question_6()
