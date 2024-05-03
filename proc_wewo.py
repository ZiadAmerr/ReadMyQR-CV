import cv2
import numpy as np
import matplotlib.pyplot as plt


# define crucial functions to extract the qr code from the image
def get_start_and_end_points(img: np.ndarray) -> tuple:
    # define all starting points as -1 (not found yet)
    start_row = -1
    start_col = -1
    end_row = -1
    end_col = -1

    # loop through the image to find the start and end points of the qr code
    # what happens here is that, initially, the value of the pixel is 255 (white)
    # so we keep looping until we find a pixel that is not white, and thus, we found the start point
    # we do the same for the end point, but we loop from the end of the image to the start
    # to find the end point. The same goes for the columns using the transpose of the image
    # to loop through the columns

    # loop through each row of pixels
    for row_index, row in enumerate(img):
        # for each pixel in that row
        for pixel in row:
            # if there is a pixel that is not white
            if pixel != 255:
                # then we found the start row!
                start_row = row_index

                # break the loop
                break

        # if after the loop, the start row is still -1, then no non-white pixels were found, and thus we continue to the next row
        # otherwise, the next line checks if the start row was found, and if it was, it breaks the outer loop
        if start_row != -1:
            break

    # do the same for the end row, but loop through that row in reverse
    for row_index, row in enumerate(img[::-1]):
        for pixel in row:
            if pixel != 255:
                end_row = img.shape[0] - row_index
                break
        if end_row != -1:
            break

    # do the same for the columns, but using the transpose of the image
    for col_index, col in enumerate(cv2.transpose(img)):
        for pixel in col:
            if pixel != 255:
                start_col = col_index
                break
        if start_col != -1:
            break

    # do the same for the end column, but using the transpose of the image and looping in reverse
    for col_index, col in enumerate(cv2.transpose(img)[::-1]):
        for pixel in col:
            if pixel != 255:
                end_col = img.shape[1] - col_index
                break
        if end_col != -1:
            break

    # return the start and end points
    return start_row, start_col, end_row, end_col


def apply_kernel(img, kernel):
    # Apply filter
    filtered = cv2.filter2D(img, -1, kernel)

    # Convert the result back to uint8
    back_to_int = np.uint8(np.absolute(filtered))

    # Add the Laplacian result to the original image to sharpen it
    return cv2.add(img, back_to_int)


def get_grid_cell_size_and_num(qr_no_quiet_zone: np.ndarray) -> tuple:
    # get the size of the first non-white pixel in the qr code
    size = 0
    for pixel in qr_no_quiet_zone[0]:
        if pixel != 0:
            break
        size += 1

    # The size of the grid cell is the size of the qr code divided by 7, which is the width of the top-left border of the alignment pattern
    grid_cell_size = round(size / 7)

    # The number of grid cells is the size of the qr code divided by the size of the grid cell
    grid_cells_num = round(qr_no_quiet_zone.shape[0] / grid_cell_size)

    # return the size of the grid cell and the number of grid cells
    return grid_cell_size, grid_cells_num


def get_numeric_qr_cells(img: np.ndarray) -> np.ndarray:
    # get the start and end idxs of the qr code
    start_row, start_col, end_row, end_col = get_start_and_end_points(img)

    # get the qr code without the quiet zone
    qr_no_quiet_zone = img[start_row:end_row, start_col:end_col]

    # get the size of a grid cell and the number of grid cells
    grid_cell_size, grid_cells_num = get_grid_cell_size_and_num(qr_no_quiet_zone)

    # reshape the qr code to a 2D array of grid cells
    qr_cells = qr_no_quiet_zone.reshape(
        (
            grid_cells_num,
            grid_cell_size,
            grid_cells_num,
            grid_cell_size,
        )
    ).swapaxes(1, 2)

    # form an empty array to store the numeric values of the qr cells
    qr_cells_numeric = np.ndarray((grid_cells_num, grid_cells_num), dtype=np.uint8)

    # loop through the qr cells and get the median value of each cell
    for i, row in enumerate(qr_cells):
        for j, cell in enumerate(row):
            qr_cells_numeric[i, j] = np.median(cell) // 255

    # return the numeric qr cells
    return qr_cells_numeric


def give_me_circle_mask_nowww(mask_size, radius):
    mask = np.zeros(mask_size)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    return cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1).astype(np.uint8)


def try_lowpass(dft_img, limit, gaussian: bool = False):
    mask = give_me_circle_mask_nowww(dft_img.shape, limit)
    if gaussian:
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_lowpass = np.multiply(dft_img_shifted, mask)
    plot_shifted_fft_and_ifft(dft_img_shifted_lowpass)


def try_highpass(dft_img, limit, gaussian: bool = False, keep_dc: bool = False):
    mask = ~give_me_circle_mask_nowww(dft_img.shape, limit)
    if gaussian:
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    if keep_dc:
        mask[dft_img.shape[0] // 2, dft_img.shape[1] // 2] = 255
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_highpass = np.multiply(dft_img_shifted, mask)
    plot_shifted_fft_and_ifft(dft_img_shifted_highpass)

    return dft_img_shifted_highpass


def plot_shifted_fft_and_ifft(dft_img_shifted):
    img = np.fft.ifft2(np.fft.ifftshift(dft_img_shifted))
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax1.set(
        yticks=[0, img.shape[0] // 2, img.shape[0] - 1],
        yticklabels=[-img.shape[0] // 2, 0, img.shape[0] // 2 - 1],
    )
    ax1.set(
        xticks=[0, img.shape[1] // 2, img.shape[1] - 1],
        xticklabels=[-img.shape[1] // 2, 0, img.shape[1] // 2 - 1],
    )
    ax1.imshow(np.abs(dft_img_shifted) ** 0.1, cmap="gray")
    ax2.imshow(np.abs(img), cmap="gray")


def give_me_circle_mask_nowww(mask_size, radius):
    mask = np.zeros(mask_size)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    return cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1).astype(np.uint8)


def try_band_remove(
    dft_img,
    mask,
    gaussian: bool = False,
    keep_dc: bool = False,
    plot: bool = True,
    return_dft: bool = False,
):
    if gaussian:
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    if keep_dc:
        mask[dft_img.shape[0] // 2, dft_img.shape[1] // 2] = 255
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_band_removed = np.multiply(dft_img_shifted, mask)

    if plot:
        plot_shifted_fft_and_ifft(dft_img_shifted_band_removed)

    img = np.fft.ifft2(np.fft.ifftshift(dft_img_shifted_band_removed))

    if return_dft:
        return dft_img_shifted_band_removed, img

    return img


def get_rectangle_mask(mask_size, x1, y1, x2, y2):
    mask = np.zeros(mask_size)
    return cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1).astype(np.uint8)


def get_exclude_freq_mask(shape, center, x_region, y_region):
    large_mask = get_rectangle_mask(
        shape,
        center[1] - x_region[1] // 2,
        center[0] - y_region[1] // 2,
        center[1] + x_region[1] // 2,
        center[0] + y_region[1] // 2,
    )
    small_mask = ~get_rectangle_mask(
        shape,
        center[1] - x_region[0] // 2,
        center[0] - y_region[0] // 2,
        center[1] + x_region[0] // 2,
        center[0] + y_region[0] // 2,
    )
    return ~(large_mask & small_mask)


def preprocess_wewo(img_path, k_freq_to_eliminate):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    dft_img = np.fft.fft2(img)

    center = (dft_img.shape[0] // 2, dft_img.shape[1] // 2)

    mask = get_exclude_freq_mask(
        dft_img.shape, center, (k_freq_to_eliminate, k_freq_to_eliminate + 1), (-1, 1)
    )

    dft_img, img = try_band_remove(
        dft_img, mask, gaussian=False, keep_dc=True, plot=False, return_dft=True
    )

    x = np.abs(img)

    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

    final_processed_image = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1]

    return final_processed_image


def main():
    path = "./test_data/11_wewowewo.png"
    x = preprocess_wewo(path, 21)
    plt.imshow(x, cmap="gray")
    plt.savefig("output_wewo.png")
    


if __name__ == "__main__":
    main()
