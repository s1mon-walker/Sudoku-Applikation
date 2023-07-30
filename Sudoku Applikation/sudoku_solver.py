import cv2
import numpy as np
import pytesseract
import time
import concurrent.futures

# ====================================================
# ===== Sudoku Solver Klasse für PyQT-Applikatio =====
# ===== S. Walker, Ph. Eberle, H. Mazlagic       =====
# ===== ABB Technikerschule                      =====
# ====================================================
# Quellen und Inspiration:
# https://github.com/05rs/Sudoku_AR
# https://stackoverflow.com/questions/57636399/how-to-detect-sudoku-grid-board-in-opencv


class SudokuSolver:
    def __init__(self, src):
        self.src = src
        self.src_type = 'image'
        self.state = 'INIT'
        self.state_update_callback = None
        self.img = None  # Original Image
        self.gray = None  # Image in grayscale
        self.blurred = None  # Blurred Image
        self.th_adaptive = None  # Adaptive Threshold of Image
        self.th_angle_corr = None  # perspective transformed Threshold
        self.img_angle_corr = None  # perspective transformed image
        self.cnts = None  # List of all Contours in image
        self.img_conts = None  # image showing the contours found
        self.img_corners = None  # List with corners of the Sudoku field
        self.boxes_list = None  # List of 81 boxes with numbers
        self.boxes_stacked = None  # boxes stacked after morphological operations
        self.box_mask = None  # Mask for removing sudoku grid from number images
        self.virtual_grid = None  # drawn version of original sudoku
        self.ar_grid = None  # original image with AR numbers filled in
        self.corners = None  # 4 corners of sudoku detected with approx
        self.corners_sorted = None  # corners sorted around center
        self.data = []  # empty list for sudoku data
        self.grid_valid = True  # if grid is invalid, solver does not attempt to do it
        self.solver = None  # iterable that returns all possible solutions
        self.solution = None  # solution for the sudoku
        self.solving_timelimit = 10  # sec
        self.timer = None  # timer to limit solvingprocess
        self.visualize_solver = False

        # configurable parameters
        self.blur_kernel = 3
        self.box_th_val = 180
        self.box_erode_iter = 1
        self.box_dilate_iter = 1
        self.min_num_area = 200
        self.th_blocksize = 11
        self.th_C = 3
        self.mask_size = 15

        self.cam_offline = cv2.imread('assets/image/camoffline.png')
        self.error_img = cv2.imread('assets/image/error_img.jpg')

        self.grid = [[None for _ in range(9)] for _ in range(9)]  # sudoku data for the solving algorithm
        self.counter = 0
        self.orig_grid_mask = [[None for _ in range(9)] for _ in range(9)]  # mask for th original numbers given

    def run(self):
        self.boxes_list = None
        self.data = []
        self.grid = [[None for _ in range(9)] for _ in range(9)]

        start_time = time.time()

        # get new frame based on source type
        self.__set_state('LOADING IMG')
        if self.src_type == 'image':
            self.img = cv2.imread(self.src)
        elif self.src_type == 'webcam':
            cap = cv2.VideoCapture(self.src)
            ret, self.img = cap.read()
        elif self.src_type == 'ipcam':
            cap = cv2.VideoCapture(self.src)
            ret, self.img = cap.read()
            if not ret:
                self.img = self.cam_offline
                self.gray = self.error_img
                self.th_adaptive = self.error_img
                self.img_conts = self.error_img
                self.img_corners = self.error_img
                self.img_angle_corr = self.error_img
                self.ar_grid = self.error_img
                self.build_virtual_grid()
                self.boxes_list = []
                return 1

        # convert to grayscale if color image
        if len(self.img.shape) == 3:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img
        self.blurred = cv2.medianBlur(self.gray, self.blur_kernel)
        self.th_adaptive = cv2.adaptiveThreshold(self.blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.th_blocksize, self.th_C)

        # find contours in the threshold image
        self.__set_state('FINDING CONTOURS')
        cnts, hir = cv2.findContours(self.th_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        self.img_conts = self.img.copy()
        cv2.drawContours(self.img_conts, cnts, -1, (0, 255, 0), 2)

        # approximate 4 cornered shape around sudoku grid
        self.__set_state('APPROXIMATE GRID')
        length = cv2.arcLength(self.cnts[0], True)
        self.corners = cv2.approxPolyDP(self.cnts[0], 0.015 * length, True)
        self.img_corners = self.img.copy()
        cv2.drawContours(self.img_corners, self.corners, -1, (0, 255, 0), 10)

        # grid can only be valid if the approximation has 4 corners
        if len(self.corners) == 4:
            self.sort_corners()
            self.angle_corr()

            self.devide_in_81_boxes()
            self.show_boxes()

            self.guess_all_numbers()
            self.build_virtual_grid(self.grid)

            self.check_sudoku()
            if self.grid_valid:
                self.get_solution()
                self.visualize_solver = False

                self.build_ar_grid()
                #self.transform_ar_grid()

                print('Solved Sudoku in {}s'.format(time.time()-start_time))
                self.__set_state('DONE')
                return 0
            else:
                self.build_ar_grid()
                self.ar_grid = self.error_img
                print('[SOLVER] Grid scan invalid')
                self.__set_state('ERROR Grid scan invalid')
                return 1
        else:
            print('[ERROR] no valid sudoku grid found!')
            self.__set_state('ERROR no valid grid found')
            #cv2.putText(error_img, 'no valid sudoku found!', (20, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.img_angle_corr = self.error_img
            self.ar_grid = self.error_img
            self.build_virtual_grid()
            return 1

    def set_source(self, path, type='image'):
        self.src = path
        self.src_type = type

    @staticmethod
    def distance(p1, p2):
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return dist

    def sort_corners(self):
        """
        Puts the detected corners in the right order by comparing each one to the centerpoint
        :return:
        """
        self.__set_state('SORTING CORNERS')
        corners = [[c[0][0], c[0][1]] for c in self.corners]  # reshape from (4,1,2) to (4,2)

        x, y = [], []
        for i in range(len(corners)):
            x.append(corners[i][0])
            y.append(corners[i][1])

        center = [sum(x) / len(x), sum(y) / len(y)]  # mean of x and y coordinates

        # sort based on relative position of the corners to the center
        for _, item in enumerate(corners):
            if item[0] < center[0]:
                if item[1] < center[1]:
                    top_left = item
                else:
                    bottom_left = item
            elif item[0] > center[0]:
                if item[1] < center[1]:
                    top_right = item
                else:
                    bottom_right = item

        self.corners_sorted = np.float32([top_left, top_right, bottom_right, bottom_left])
        print('[SOLVER] corners_sorted = ', self.corners_sorted, type(self.corners_sorted))
        return 0

    def angle_corr(self):
        """
        Calculates the perspective transform for the image and threshold
        :return:
        """
        self.__set_state('PERSPECTIVE TRANSFORM')
        top_left, top_right, bottom_right, bottom_left = self.corners_sorted

        width1 = self.distance(bottom_right, bottom_left)
        width2 = self.distance(top_right, top_left)
        height1 = self.distance(top_right, bottom_right)
        height2 = self.distance(top_left, bottom_right)
        width = max(int(width1), int(width2))
        height = max(int(height1), int(height2))

        dimensions = np.array([[0, 0], [width, 0], [width, width], [0, width]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(self.corners_sorted, dimensions)

        self.img_angle_corr = cv2.warpPerspective(self.img, matrix, (width, width))
        self.img_angle_corr = cv2.resize(self.img_angle_corr, (900, 900), interpolation=cv2.INTER_AREA)
        self.th_angle_corr = cv2.warpPerspective(self.th_adaptive, matrix, (width, width))
        self.th_angle_corr = cv2.resize(self.th_angle_corr, (900, 900), interpolation=cv2.INTER_AREA)
        print('[SOLVER] perspective transform done')
        return 0

    def devide_in_81_boxes(self):
        """
        Divides the threshold in 81 individual images and applies morphological transformations
        :return:
        """
        width, height = np.shape(self.th_angle_corr)
        x_devides = [0]
        y_devides = [0]
        for i in range(1, 10):
            x_devides.append(int(width/9*i))
            y_devides.append(int(height/9*i))

        boxes_list = [[None for _ in range(9)] for _ in range(9)]
        for y in range(9):
            for x in range(9):
                x1 = x_devides[x]
                x2 = x_devides[x+1]
                y1 = y_devides[y]
                y2 = y_devides[y+1]
                boxes_list[y][x] = self.th_angle_corr[y1:y2, x1:x2]

        self.box_mask = np.zeros((100, 100), dtype='uint8')
        self.box_mask = cv2.rectangle(self.box_mask, (self.mask_size, self.mask_size), (99-self.mask_size, 99-self.mask_size), 255, -1)

        for y in range(9):
            for x in range(9):
                box = boxes_list[y][x]
                T, thresh = cv2.threshold(box, self.box_th_val, 255, cv2.THRESH_BINARY)
                eroded = cv2.erode(thresh, None, iterations=self.box_erode_iter)
                masked = cv2.bitwise_and(eroded, self.box_mask)
                dilated = cv2.dilate(masked, None, iterations=self.box_dilate_iter)
                boxes_list[y][x] = dilated

                cv2.imwrite('resources/{}-{}.jpg'.format(x, y), boxes_list[y][x])

        self.boxes_list = boxes_list
        print('[SOLVER] created list of 81 boxes')
        return 0

    def show_boxes(self):
        rows = []
        for i in range(9):
            rows.append(np.hstack(self.boxes_list[i]))
        self.boxes_stacked = np.vstack(rows)
        print('[SOLVER] build boxes image')
        return 0

    def build_virtual_grid(self, data=None):
        grid = np.ones((900, 900), dtype='uint8') * 255
        for i in range(0, 1000, 100):
            cv2.line(grid, (0, i), (900, i), 0, 3)
            cv2.line(grid, (i, 0), (i, 900), 0, 3)
        for i in range(0, 1000, 300):
            cv2.line(grid, (0, i), (900, i), 0, 8)
            cv2.line(grid, (i, 0), (i, 900), 0, 8)

        if data:  # data = [[num, x, y]]
            for y in range(9):
                for x in range(9):
                    if data[y][x] != 0:
                        cv2.putText(grid, str(data[y][x]), (int(x*(900/9)+25), int(y*(900/9)+80)), cv2.FONT_HERSHEY_SIMPLEX,
                                    2.5, 0, 3)

        self.virtual_grid = grid
        print('[SOLVER] built virtual grid')
        return 0

    def calc_pos(self, x, y):
        """
        Calculates the position of a grid cell in the image based on the position
        of the 4 corners of the grid and x/y coordinates (0-8) of the cell.
        :param x: int - X-Coordinate
        :param y: int - Y-Coordinate
        :return: (int x, int y) position in the image
        """
        top_left, top_right, bottom_right, bottom_left = self.corners_sorted
        x_quotient = (2*x + 1) / 18  # |<----0.62--->|<-------0.38-------->| quotient
        y_quotient = (2*y + 1) / 18  # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | coordinate
        temp_upper_x = int(top_left[0] * (1 - x_quotient) + top_right[0] * x_quotient)
        temp_upper_y = int(top_left[1] * (1 - x_quotient) + top_right[1] * x_quotient)
        temp_lower_x = int(bottom_left[0] * (1 - x_quotient) + bottom_right[0] * x_quotient)
        temp_lower_y = int(bottom_left[1] * (1 - x_quotient) + bottom_right[1] * x_quotient)
        target_x = int(temp_upper_x * (1 - y_quotient) + temp_lower_x * y_quotient)
        target_y = int(temp_upper_y * (1 - y_quotient) + temp_lower_y * y_quotient)
        return target_x, target_y

    def build_ar_grid(self, pos_test_mode=False):
        """
        Generates image with AR-numbers filled in the original Sudoku image.
        Uses principle of bilinear interpolation to calculate the positions of
        the Numbers based of the position of the grid corners in the image
        :param pos_test_mode: False - if true the calculate center of every cell is shown
        :return: success
        """
        self.__set_state('VISUALIZING SOLUTION')
        self.ar_grid = self.img.copy()
        thickness = int((self.ar_grid.shape[0] + self.ar_grid.shape[1]) / 400)
        approx_grid_height = self.corners_sorted[2][1] - self.corners_sorted[0][1]
        approx_grid_width = self.corners_sorted[2][0] - self.corners_sorted[0][0]
        if pos_test_mode:
            for x in range(9):
                for y in range(9):
                    pos = self.calc_pos(x, y)
                    cv2.circle(self.ar_grid, pos, 5, (0,0,255), -1)
        else:
            font_size = 1 / 300 * approx_grid_height
            color = (0, 0, 255)
            x_offset = int(10 / 300 * approx_grid_width)
            y_offset = int(10 / 300 * approx_grid_height)
            for i in range(81):
                _, x, y = self.data[i]
                num = self.solution[y][x]
                if not self.orig_grid_mask[y][x]:
                    x_pos, y_pos = self.calc_pos(x, y)
                    x_pos -= x_offset
                    y_pos += y_offset
                    cv2.putText(self.ar_grid, str(num), (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, color, thickness)
        print('[SOLVER] built AR image')
        return 0

    def guess_all_numbers(self):
        self.__set_state('GUESSING NUMBERS')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for y in range(9):
                for x in range(9):
                    futures.append(executor.submit(self.guess_number, self.boxes_list[y][x], x, y))
            for future in concurrent.futures.as_completed(futures):
                self.data.append(future.result())

        for d in self.data:
            x, y = d[1], d[2]
            self.grid[y][x] = d[0]
            self.orig_grid_mask[y][x] = False if d[0] == 0 else True

        print(np.matrix(self.grid))

    def guess_number(self, box, x=None, y=None):
        number = 0
        cnts, hir = cv2.findContours(box.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for (i, c) in enumerate(cnts):
            area = cv2.contourArea(c)
            if area > self.min_num_area:
                options = "-l {} --psm {}".format('deu', 13)
                # options = r'--oem 3 --psm 6 outputbase digits'
                # options = r'-c tessedit_char_whitelist=123456789 --psm 6'
                time1 = time.time()
                text = pytesseract.image_to_string(box, config=options)
                print('guessd number in {}s'.format(time.time()-time1))
                #cv2.imshow('Box', box)
                #cv2.waitKey(0)

                if text[0] in 'BE':
                    number = 8
                elif text[0] in 'g':
                    number = 9
                elif text[0] in 'Iiı':
                    number = 1
                elif text[0] in 'T':
                    number = 7
                else:
                    try:
                        number = int(text[0])
                    except ValueError:
                        print('Cannot convert {} to int'.format(text[0]))

                print('{} -> {}'.format(text[0], number))
                return [number, x, y]
        return [0, x, y]

    def possible(self, y, x, n):
        """
        Checks if it is possible to place number n at spot [x,y] in self.grid
        :param y: int Y-Coordinate
        :param x: int X-Coordinate
        :param n: int Number to place
        :return: bool True if possible
        """
        for i in range(9):
            if self.grid[y][i] == n:
                return False
        for i in range(9):
            if self.grid[i][x] == n:
                return False
        x0 = (x//3) * 3
        y0 = (y//3) * 3
        for i in range(3):
            for j in range(3):
                if self.grid[y0+i][x0+j] == n:
                    return False
        return True

    def solve(self):
        """
        Recursive algorithm to solve the sudoku using backtracking.
        The Method uses self.grid and modifies it to create solutions.
        By using yield the method acts as iterable and can return every possible solution.
        Source: Computerphile - https://www.youtube.com/watch?v=G_UYXzGuqvM&t=540s
        :return: solution as 9 x 9 np array
        """
        self.counter += 1
        for y in range(9):
            for x in range(9):
                if self.grid[y][x] == 0:
                    for n in range(1, 10):
                        if self.possible(y, x, n):
                            if time.time() > self.timer + self.solving_timelimit and not self.visualize_solver:
                                print('[SOLVER] timeout: aborting solver')
                                return self.grid
                            self.grid[y][x] = n
                            if self.visualize_solver:
                                self.show_solver_step()
                            yield from self.solve()
                            self.grid[y][x] = 0
                    return self.grid
        yield np.asarray(self.grid)

    def show_solver_step(self):
        self.build_virtual_grid(self.grid)
        cv2.imshow('Visualisierung Lösungsalgorithmus', self.virtual_grid)
        cv2.waitKey(100)

    def check_sudoku(self):
        """
        This Method checks if the data generated from the image can be a valid sudoku
        by checking every given number follows the sudoku rules
        :return: Sudoku data valid
        """
        self.__set_state('CHECKING SUDOKU SCAN')
        self.grid_valid = True
        for y in range(9):
            for x in range(9):
                if self.grid[y][x] == 0:
                    continue
                num = self.grid[y][x]
                self.grid[y][x] = 0
                if self.possible(y, x, num):
                    self.grid[y][x] = num
                else:
                    self.grid_valid = False
                    return False
        return True

    def get_solution(self):
        self.__set_state('CALCULATING SOLUTION')
        try:
            self.timer = time.time()
            if not self.solver:
                self.counter = 0
                self.solver = self.solve()
            self.solution = next(self.solver)
            print('[SOLVER] got solution')
            return 0
        except StopIteration:
            print('[ERROR] no solution found')
            return 1

    def __set_state(self, state):
        self.state = state
        if self.state_update_callback:
            self.state_update_callback()


if __name__ == '__main__':
    solver = SudokuSolver('img/sudoku1.jpeg')
    #solver.set_source('http://192.168.1.120:8080/video', 'ipcam')
    solver.run()

    cv2.imshow('Virtueles Grid', solver.virtual_grid)
    cv2.imshow('AR Grid', solver.ar_grid)
    cv2.waitKey(0)



