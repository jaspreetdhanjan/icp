import numpy as np
import random


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

    def apply_transformation(self, rotation_matrix, translation_vector=np.array([[0], [0], [0]])):
        for i in range(0, len(self.points)):
            self.points[i] = (rotation_matrix.dot(self.points[i])) - translation_vector

    def apply_zero_mean_noise(self, variance):
        random.seed(1)

        for i in range(0, len(self.points)):
            if random.randint(0, variance) == 0:
                self.points[i] = self.points[i] + random.gauss(-0.01, +0.01)


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
