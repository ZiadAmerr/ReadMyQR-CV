MASKS = {
    "000": lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
    "001": lambda i, j: (i // 2 + j // 3) % 2 == 0,
    "010": lambda i, j: ((i + j) % 2 + (i * j) % 3) % 2 == 0,
    "011": lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
    "100": lambda i, j: i % 2 == 0,
    "101": lambda i, j: (i + j) % 2 == 0,
    "110": lambda i, j: (i + j) % 3 == 0,
    "111": lambda i, j: j % 3 == 0,
}
UP8, UP4, DOWN8, DOWN4, CW8, CCW8 = range(6)

DIRECTION_OFFSETS = {
    UP8: {
        "row_offsets": [0, 0, -1, -1, -2, -2, -3, -3],
        "col_offsets": [0, -1, 0, -1, 0, -1, 0, -1],
    },
    UP4: {"row_offsets": [0, 0, -1, -1], "col_offsets": [0, -1, 0, -1]},
    DOWN8: {
        "row_offsets": [0, 0, 1, 1, 2, 2, 3, 3],
        "col_offsets": [0, -1, 0, -1, 0, -1, 0, -1],
    },
    DOWN4: {"row_offsets": [0, 0, 1, 1], "col_offsets": [0, -1, 0, -1]},
    CW8: {
        "row_offsets": [0, 0, 1, 1, 1, 1, 0, 0],
        "col_offsets": [0, -1, 0, -1, -2, -3, -2, -3],
    },
    CCW8: {
        "row_offsets": [0, 0, -1, -1, -1, -1, 0, 0],
        "col_offsets": [0, -1, 0, -1, -2, -3, -2, -3],
    },
}

N_DIM = 21


QR_READ_STEPS = [
    [-7, -1, UP8],
    [-11, -1, CCW8],
    [-10, -3, DOWN8],
    [-6, -3, DOWN8],
    [-2, -3, CW8],
    [-3, -5, UP8],
    [-7, -5, UP8],
    [-11, -5, CCW8],
    [-10, -7, DOWN8],
    [-6, -7, DOWN8],
    [-2, -7, CW8],
    [-3, -9, UP8],
    [-7, -9, UP8],
    [-11, -9, UP8],
    [-16, -9, UP8],
    [-20, -9, CCW8],
    [-19, -11, DOWN8],
    [-14, -11, DOWN4],
    [-12, -11, DOWN8],
    [-8, -11, DOWN8],
    [-4, -11, DOWN8],
    [-9, -13, UP8],
    [-12, -16, DOWN8],
    [-9, -18, UP8],
    [-12, -20, DOWN8],
]

QR_READ_STEPS = [[N_DIM + x, N_DIM + y, d] for x, y, d in QR_READ_STEPS]
