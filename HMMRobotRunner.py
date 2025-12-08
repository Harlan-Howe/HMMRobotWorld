import random
from sysconfig import expand_makefile_vars
from typing import Optional, List, Tuple

import cv2
import numpy as np
from HMMRobotWorldViewer import HMMRobotWorldViewer

OBSERVATION_ACCURACY_RATE = 1.0

class HMMRobotRunner:
    def __init__(self):
        self.world = self.load_file("HMM Robot World - Sheet1.tsv")
        self.num_open_squares = np.count_nonzero(self.world == 0)
        self.pi_matrix: Optional[np.ndarray] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self.observation_matrix: Optional[np.ndarray] = None
        self.viewer = None
        self.build_T_and_O()

        self.viewer = HMMRobotWorldViewer(self.world)
        self.viewer.display_world(wait_for_keystroke=False, dismiss=False)

        path, obs = self.generate_path_and_observations(60)


        ps = [1/self.num_open_squares for _ in range(self.num_open_squares)]
        self.viewer.set_probabilities_list(ps)
        self.viewer.display_observations(observation_list=obs, accuracy=OBSERVATION_ACCURACY_RATE)
        self.viewer.display_world()
        # self.test_viterbi(obs, path)

        # self.test_forward_backward(obs)

    def test_forward_backward(self, obs: list[int]):
        probabilities = self.calculate_probabilities(obs)
        for i in range(len(obs)):
            self.viewer.set_probabilities_list(probabilities[i])
            self.viewer.display_world(False, False)
            self.viewer.display_observations(observation_list=obs, accuracy=OBSERVATION_ACCURACY_RATE,
                                             highlight_obs_num=i)
            cv2.waitKey(500)

    def test_viterbi(self, obs: list[int], path: list[int]):
        predicted_path = self.calculate_viterbi(obs)

        missed = 0
        for i in range(len(path)):
            print(f"{chr(i + 65)}\t{path[i]}\t{predicted_path[i]}")
            if path[i] != predicted_path[i]:
                missed += 1
        print(f'Num missed = {missed}')
        print(f"Accuracy = {(1 - missed / len(path)) * 100:3.2f}%")
        print(f"Observtion Accuracy = {OBSERVATION_ACCURACY_RATE * 100:3.2f}")
        self.viewer.display_world()

        self.viewer.draw_path(predicted_path)

    def load_file(self, filename) -> np.ndarray:
        map_list = []
        try:
            with open(file=filename, mode='r') as f:
                for line in f:
                    parts = line.split("\t")
                    row = []
                    for p in parts:
                        row.append(int(p))
                    map_list.append(row)
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            f.close()

        return np.array(map_list)

    def build_T_and_O(self):
        numbered_map = self.world * -1
        count = 0
        for row in range(self.world.shape[0]):
            for col in range(self.world.shape[1]):
                if self.world[row, col] == 0:
                    numbered_map[row, col] = count
                    count += 1
        self.pi_matrix = np.ones(self.num_open_squares, dtype=float) / self.num_open_squares
        self.transition_matrix = np.zeros((self.num_open_squares, self.num_open_squares), dtype=float)
        self.observation_matrix = np.zeros((self.num_open_squares, 16), dtype=float)

        deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N/1, E/2, S/4, W/8
        for row in range(numbered_map.shape[0]):
            for col in range(numbered_map.shape[1]):
                index = numbered_map[row, col]
                if index > -1:
                    option_count = 0
                    observation_probabilities = [OBSERVATION_ACCURACY_RATE for _ in range(4)]  #  assume we likely see a wall in all directions.
                    dir_num = 0
                    for dr, dc in deltas:
                        val_at_delta = numbered_map[row+dr, col+dc]
                        if val_at_delta > -1:
                            self.transition_matrix[index, val_at_delta] = 0.25
                            option_count += 1
                            #  in this direction, we see no wall, so invert that probability.
                            observation_probabilities[dir_num] = 1 - observation_probabilities[dir_num]
                        dir_num += 1
                    self.transition_matrix[index, index] = (4 - option_count) * 0.25
                    # generate the 16 probabilities for observing walls in the four directions.
                    for n in range(2):
                        for e in range(2):
                            for s in range(2):
                                for w in range(2):
                                    value = ((2*n-1)*observation_probabilities[0]+1-n) * \
                                    ((2 * e - 1) * observation_probabilities[1] + 1 - e) * \
                                    ((2 * s - 1) * observation_probabilities[2] + 1 - s) * \
                                    ((2 * w - 1) * observation_probabilities[3] + 1 - w)
                                    loc = n + 2 * e + 4 * s + 8 * w
                                    self.observation_matrix[index,loc] = value

    def generate_path_and_observations(self, N: int) -> Tuple[List[int],List[int]]:
        hidden_path: List[int] = []
        observations: List[int] = []

        p = random.randint(0, self.num_open_squares-1)
        for n in range(N):
            sum = 0
            rnd = random.random()
            hidden_path.append(p)
            for i in range(16):
                sum+= self.observation_matrix[p][i]
                if rnd <= sum:
                    observations.append(i)
                    break
            rnd2 = random.random()
            sum = 0
            for i in range(self.num_open_squares):
                sum += self.transition_matrix[p][i]
                if rnd <= sum:
                    p = i
                    break
            if self.viewer is not None:
                self.viewer.display_observations(observations, accuracy=OBSERVATION_ACCURACY_RATE)


        return hidden_path, observations

    def calculate_viterbi(self, observations:List[int]) -> List[int]:
        """
        :param observations: a list of N observations (0-15 each) of what walls the robot saw
        :return: a list of the most likely path through the
        """
        N = len(observations)

        V = np.zeros((N+1, self.num_open_squares), dtype=float)
        V[0] = self.pi_matrix * self.observation_matrix[:,observations[0]]
        back = np.array([[0 for _ in range(self.num_open_squares)] for _  in range(N)])
        for step in range(1, N):
            for possible_current_loc in range(self.num_open_squares):
                mesh = V[step-1] * self.transition_matrix[:, possible_current_loc]
                mesh = mesh * self.observation_matrix[possible_current_loc, observations[step]]
                V[step, possible_current_loc] = np.max(mesh)
                back[step, possible_current_loc] = np.argmax(mesh)

        path = [np.argmax(back[N-1,:], axis=0),]
        i = N-1
        while i > 0:
            pos = path[0]
            path.insert(0, back[i, pos])
            i -= 1


        return path


    def forward(self, observations:List[int],
                pi_matrix:Optional[np.ndarray] = None,
                trans_matrix:Optional[np.ndarray] = None,
                obs_matrix:Optional[np.ndarray] = None) -> np.ndarray:
        N = len(observations)
        if pi_matrix is None:
            pi_matrix = np.ones(self.num_open_squares)/self.num_open_squares
        if trans_matrix is None:
            trans_matrix = self. transition_matrix
        if obs_matrix is None:
            obs_matrix = self.observation_matrix

        alpha = np.zeros((N, self.num_open_squares), dtype=float)
        alpha[0, :] = pi_matrix
        for i in range(1, N):
            alpha[i, :] = alpha[i-1, :].dot(trans_matrix) * obs_matrix[:, observations[i]]
        return alpha

    def backward(self, observations: List[int],
                 trans_matrix:Optional[np.ndarray] = None,
                 obs_matrix:Optional[np.ndarray] = None) -> np.ndarray:
        N = len(observations)
        if trans_matrix is None:
            trans_matrix = self. transition_matrix
        if obs_matrix is None:
            obs_matrix = self.observation_matrix
        beta = np.ones((N, self.num_open_squares), dtype=float)
        for i in range(N-2, -1, -1):
            beta[i, :] = trans_matrix.dot(beta[i+1, :] * obs_matrix[:, observations[i]])
            beta[i, :] /= np.sum(beta[i, :])
        return beta

    def calculate_probabilities(self, observations: List[int],
                                pi_matrix: Optional[np.ndarray] = None,
                                trans_matrix:Optional[np.ndarray] = None,
                                obs_matrix:Optional[np.ndarray] = None) -> np.ndarray:
        N = len(observations)
        if pi_matrix is None:
            pi_matrix = np.ones(self.num_open_squares)/self.num_open_squares
        if trans_matrix is None:
            trans_matrix = self. transition_matrix
        if obs_matrix is None:
            obs_matrix = self.observation_matrix
        alpha = self.forward(observations, pi_matrix, trans_matrix, obs_matrix)
        beta = self.forward(observations, trans_matrix, obs_matrix)
        gamma = alpha * beta
        normalize_totals = np.sum(gamma, axis=1)
        for i in range(len(observations)):
            gamma[i] = gamma[i, :]/normalize_totals[i]
        return gamma

    def baum_welch(self, observations:List[int]):
        pi_matrix = np.random.rand(self.num_open_squares)
        transition_matrix = np.random.rand(self.num_open_squares, self.num_open_squares)
        observation_matrix = np.random.rand(self.num_open_squares, 16)

        for i in range(10):
            gamma, xi = self.expectation(observations, pi_matrix, transition_matrix, observation_matrix)
            pi_matrix, transition_matrix, observation_matrix = self.maximization(gamma, xi, observations)



    def expectation(self, observations: List[int],
                    pi_matrix:np.ndarray,
                    trans_matrix:np.ndarray,
                    obs_matrix:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gamma = self.calculate_probabilities(observations, pi_matrix, trans_matrix, obs_matrix)
        xi = np.zeros((self.num_open_squares, self.num_open_squares, len(observations)-1), dtype=float)
        for i in range(len(observations)-1):
            xi[i] = gamma[i, :, np.newaxis] * gamma[i+1, :]
        return gamma, xi

    def maximization(self, gamma, xi, observations) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pi_matrix = gamma[0, :]
        trans_matrix = np.sum(xi, axis=2)/ np.sum(gamma, axis=1)
        obs_matrix = np.zeros((self.num_open_squares, 16), dtype=float)
        for i in range(self.num_open_squares):
            for t in range(gamma.shape[0]):
                obs_matrix[i, observations[t]] += gamma[t, i]
        obs_matrix /= np.sum(gamma, axis=1)

        return pi_matrix, trans_matrix, obs_matrix

if __name__ == "__main__":
    hmm_RR = HMMRobotRunner()


