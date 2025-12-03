import random
from typing import List, Tuple, Optional

import cv2
import numpy as np

CELL_SIZE = 40

class HMMRobotWorldViewer:

    def __init__(self, base_world: np.ndarray):
        self.base = base_world
        self.probabilities_list: Optional[List[float]] = None

    def set_probabilities_list(self, ps: Optional[List[float]]):
        self.probabilities_list = ps

    def display_world(self, wait_for_keystroke: bool = True, dismiss: bool = True):
        canvas = self.build_starter()

        cv2.imshow("World", canvas)
        cv2.moveWindow("World", 0, 3 * CELL_SIZE)
        if wait_for_keystroke:
            cv2.waitKey()
        if dismiss:
            cv2.destroyWindow("World")

    def display_observations(self, observation_list: List[int], accuracy: float = 1.0):
        canvas = np.ones((int(2*CELL_SIZE), int(1.75*CELL_SIZE*len(observation_list)),3), dtype=float) * 0.5
        canvas[:, :, 0:2] = 0.65

        for i in range(len(observation_list)):
            h_offset = int(0.5*CELL_SIZE+1.75*CELL_SIZE*i)
            self.draw_observation_box_at(canvas=canvas,
                                         topleft=(h_offset, CELL_SIZE//4),
                                         filled = observation_list[i] & 1 == 1,
                                         accuracy = accuracy)
            self.draw_observation_box_at(canvas=canvas,
                                         topleft=(h_offset+CELL_SIZE//2, 3*CELL_SIZE//4),
                                         filled=observation_list[i] & 2 == 2,
                                         accuracy = accuracy)
            self.draw_observation_box_at(canvas=canvas,
                                         topleft=(h_offset,5*CELL_SIZE//4),
                                         filled = observation_list[i] & 4 == 4,
                                         accuracy = accuracy)
            self.draw_observation_box_at(canvas=canvas,
                                         topleft=(h_offset - CELL_SIZE // 2, 3*CELL_SIZE // 4),
                                         filled=observation_list[i] & 8 == 8,
                                         accuracy = accuracy)
            cv2.circle(img=canvas,center=(h_offset+CELL_SIZE//4, CELL_SIZE),
                       radius = CELL_SIZE//4, color = (0.5, 0.5, 0.75), thickness = -1)
            cv2.putText(img=canvas, text=f"{chr(i+65)}", org=(h_offset+CELL_SIZE//4,CELL_SIZE+7), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale= 0.25, color = (0,0,0))
        cv2.imshow("Observations",canvas)
        cv2.waitKey(250)

    def draw_observation_box_at(self, canvas: np.ndarray, topleft:Tuple[int,int], filled:bool, accuracy:float = 1.0):
        if filled:
            cv2.rectangle(img=canvas,
                          pt1=topleft,
                          pt2 = (int(topleft[0]+CELL_SIZE/2), int(topleft[1]+CELL_SIZE/2)),
                          color= (0,0,0),
                          thickness= -1)
            if accuracy < 1:
                for x in range(topleft[0],topleft[0]+CELL_SIZE//2):
                    for y in range(topleft[1],topleft[1]+CELL_SIZE//2):
                        if random.random() > accuracy:
                            canvas[y,x] = (1, 1, 1)
        else:
            cv2.rectangle(img=canvas,
                          pt1=topleft,
                          pt2=(int(topleft[0] + CELL_SIZE / 2), int(topleft[1] + CELL_SIZE / 2)),
                          color=(1, 1, 1),
                          thickness=-1)
            if accuracy < 1:
                for x in range(topleft[0],topleft[0]+CELL_SIZE//2,2):
                    for y in range(topleft[1],topleft[1]+CELL_SIZE//2,2):
                        if random.random() > accuracy:
                            canvas[y,x] = (0, 0, 0)

    def build_starter(self) -> np.ndarray:
        space_num = 0
        result = np.zeros((self.base.shape[0]*CELL_SIZE, self.base.shape[1]*CELL_SIZE, 3), dtype=float)
        for row in range(self.base.shape[0]):
            for col in range(self.base.shape[1]):
                if self.base[row, col] == 0:
                    cv2.rectangle(img=result,
                                  pt1=(CELL_SIZE*col+1, CELL_SIZE*row+1),
                                  pt2=(CELL_SIZE*(col+1)-1, CELL_SIZE*(row+1)-1),
                                  color=(1.0, 1.0, 1.0),
                                  thickness=-1)
                    if self.probabilities_list is not None and len(self.probabilities_list)>space_num:
                        prob:float = self.probabilities_list[space_num]
                        cv2.circle(img=result,
                                   center=(CELL_SIZE * col + CELL_SIZE // 2, CELL_SIZE * row + CELL_SIZE // 2),
                                   radius= CELL_SIZE // 3,
                                   color=(0.5 - 0.5 * prob, 0.5 - 0.5 * prob, 1 - 0.5 * prob),
                                   thickness=int(1 + prob * 6))
                    cv2.putText(img=result,
                                text=f"{space_num}",
                                org=(int(CELL_SIZE * (col + 0.25))+1, int(CELL_SIZE * (row + 0.75))+1),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(1, 1, 1),
                                thickness=2)

                    cv2.putText(img=result,
                                text=f"{space_num}",
                                org=(int(CELL_SIZE*(col + 0.25)), int(CELL_SIZE*(row + 0.75))),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0.5, 0.25, 0),
                                thickness=2)
                    space_num += 1

        return result
