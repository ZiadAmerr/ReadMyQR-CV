from typing import Dict, List, Tuple
import pickle
import argparse
import reedsolo as rs
import re
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
from consts import (
    MASKS,
    DIRECTION_OFFSETS,
    UP4,
    DOWN4,
    UP8,
    DOWN8,
    CW8,
    CCW8,
    QR_READ_STEPS,
    N_DIM,
)


def get_ec_level(inv_image: np.ndarray) -> np.ndarray:
    ec_level = inv_image[8, 0:2]  # upper left
    _ec_level = inv_image[-2:, 8][::-1]  # bottom left

    try:
        assert np.all(ec_level == _ec_level), "Error correction level not consistent"
    except AssertionError as e:
        ec_level = _ec_level  # bottom left

    return ec_level


def get_mask(inv_image: np.ndarray) -> np.ndarray:
    mask = inv_image[8, 2:5]
    _mask = inv_image[-5:-2, 8][::-1]

    try:
        assert np.all(mask == _mask), "Mask not consistent"
    except AssertionError as e:
        mask = _mask

    return mask


def get_fmt_ec(inv_image: np.ndarray) -> list:
    fmt_ec = []

    fmt_ec.append(inv_image[8, 5])
    fmt_ec.append(inv_image[8, 7])
    fmt_ec.extend(inv_image[0:6, 8])
    fmt_ec.extend(inv_image[7:9, 8])

    return fmt_ec


def get_qr_metadata(image: np.ndarray, inverted: bool) -> Dict[str, list]:
    # if not inverted, invert image
    inv_image = image if inverted else 1 - image

    # get error correction level
    ec_level = get_ec_level(inv_image)

    # get mask
    mask = get_mask(inv_image)

    # get format error correction
    fmt_ec = get_fmt_ec(inv_image)

    # xor data
    # ec_level[0] ^= 1
    # mask[0] ^= 1
    # mask[2] ^= 1
    # fmt_ec[5] ^= 1
    # fmt_ec[8] ^= 1

    # return metadata
    return {
        "ec_level": ec_level.tolist(),
        "mask": mask.tolist(),
        "fmt_ec": fmt_ec,
        "mask_str": "".join([str(c) for c in mask]),
    }


# A more general apply_mask function (still works the same way)
def apply_mask_general(
    data_start_i, data_start_j, data, direction, inverted=False, debug=False, plot=False
):
    result = []

    mask_str = get_qr_metadata(data, inverted=inverted)["mask_str"]

    offsets = DIRECTION_OFFSETS[direction]

    row_offsets = offsets["row_offsets"]
    col_offsets = offsets["col_offsets"]

    if not inverted:
        data = 1 - data

    for i, j in zip(row_offsets, col_offsets):
        x_idx = data_start_i + i
        y_idx = data_start_j + j

        if debug:
            print(f"({x_idx}, {y_idx}) -> {data[x_idx, y_idx]}")

        cell_bit = bool(data[x_idx, y_idx])
        mask_bit = MASKS[mask_str](x_idx, y_idx)
        # Modules corresponding to the dark areas of the mask are inverted.
        result.append(int(not cell_bit if mask_bit else cell_bit))
    if debug:
        print("\n\n\n")

    # show which bits were selected using the mask via matplotlib
    if plot:
        plt.imshow(1 - data, cmap="gray")
        for i, j in zip(row_offsets, col_offsets):
            x_idx = data_start_i + i
            y_idx = data_start_j + j
            plt.scatter(y_idx, x_idx, color="red")
        plt.title("Scanning with mask: " + mask_str)
        plt.show()

    return result[:4] if direction in [UP4, DOWN4] else result


def binary_to_alphanumeric(bin_str):
    alphanumeric_chars = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        " ",
        "$",
        "%",
        "*",
        "+",
        "-",
        ".",
        "/",
        ":",
    ]

    if len(bin_str) == 11:
        value = int(bin_str, 2)

        #  V = 45 * C1 + C2
        C1 = value // 45
        C2 = value % 45

        char1 = alphanumeric_chars[C1]
        char2 = alphanumeric_chars[C2]

        return char1, char2

    else:  # decode the last 6 bits
        value = int(bin_str, 2)

        char = alphanumeric_chars[value]

        return char, None


def extract_alphanum(data_bits):
    # split each 11 bits into 2 6-bit numbers
    data_chars = [data_bits[i : i + 11] for i in range(0, len(data_bits), 11)]

    all_chars = []

    all_chars = []
    for data_char in data_chars[:-1]:
        char1, char2 = binary_to_alphanumeric(data_char)
        all_chars.append(char1)
        all_chars.append(char2)

    # decode the last 6 bits
    char, _ = binary_to_alphanumeric(data_chars[-1])

    all_chars.append(char)

    return "".join(all_chars)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-n", "--n-test-case", type=int, default=0, help="Test case number"
    )
    args_parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    args = args_parser.parse_args()

    debug = args.debug
    n_tc = args.n_test_case

    images = pickle.load(
        open(
            "/Users/ziad/Desktop/Projects/CV/preprocessing_pipeline/read_images.pkl",
            "rb",
        )
    )
    image_name, image = images[n_tc]

    plt.imshow(image, cmap="gray")
    # plt.show()
    plt.savefig(f"image_{n_tc}_{image_name}.png")
    plt.close()

    if debug:
        print(f"Image name: {image_name}")

    meta = get_qr_metadata(image, inverted=False)

    mask = meta["mask"]

    if debug:
        print("Mask", mask)
        print("EC Level", meta["ec_level"])
        print("Format EC", meta["fmt_ec"])

    enc_bits = apply_mask_general(
        len(image) - 1, len(image) - 1, image, UP4, inverted=False
    )
    len_bits = apply_mask_general(
        len(image) - 3, len(image) - 1, image, UP8, inverted=False
    )
    # print("".join(chars))
    enc_str = "".join([str(bit) for bit in enc_bits])
    if enc_str == "0100":
        enc_type = "byte"
    elif enc_str == "0010":
        enc_type = "alphanum"
    else:
        enc_type = "byte"  # SHOULD BE CHANGED
        print("Invalid encoding type", end="\t")

    if debug:
        print("ENC_STR: ", enc_str)

    len_ = int("".join([str(bit) for bit in len_bits]), 2)

    if debug:
        print("Enc bits", enc_bits)
        print("Len bits", len_bits, len_)

    msg_bits = []
    msg_bits.extend(enc_bits)
    msg_bits.extend(len_bits)

    chars = []

    idx = 0

    # read the first len_ blocks
    for _ in range(min(len_, 18)):
        start_i, start_j, direction = QR_READ_STEPS[idx]
        bits = apply_mask_general(start_i, start_j, image, direction, inverted=False)
        msg_bits.extend(bits)
        bit_str = "".join([str(bit) for bit in bits])
        alpha_char = chr(int(bit_str, 2))
        chars.append(alpha_char)
        if debug:
            print(f"{bit_str} (={int(bit_str, 2):03d}) = {alpha_char}")
        idx += 1

    # read the end block
    start_i, start_j, direction = QR_READ_STEPS[idx]
    bits = apply_mask_general(start_i, start_j, image, direction, inverted=False)
    msg_bits.extend(bits)
    idx += 1

    # read the rest of the blocks
    for _ in range(len(QR_READ_STEPS) - len_ - 1):
        start_i, start_j, direction = QR_READ_STEPS[idx]
        bits = apply_mask_general(start_i, start_j, image, direction, inverted=False)
        bit_str = "".join([str(bit) for bit in bits])
        alpha_char = chr(int(bit_str, 2))
        if debug:
            print(f"{bit_str} (={int(bit_str, 2):03d}) = {alpha_char}")

        msg_bits.extend(bits)
        idx += 1

    bit_stream = "".join(str(b) for b in msg_bits)

    if debug:
        print("Msg bits: ", "".join(str(k) for k in msg_bits), end="\t")

    if enc_type == "alphanum":
        data_bits = bit_stream[13:]

        message = extract_alphanum(data_bits[: len_ * 11 + 6])

        print(f"Data in message = {message}")

        return

    message_bytes = [
        int("".join(map(str, msg_bits[i : i + 8])), 2)
        for i in range(0, len(msg_bits), 8)
    ]

    if debug:
        print("Message bytes", message_bytes)

    # Create the Reed-Solomon Codec for 7 ECC symbols (again, this is L)
    rsc = rs.RSCodec(nsym=7)

    # Decode the bytes with the 7-ECC RS Codec
    # find n errors
    try:
        message_decoded = rsc.decode(message_bytes)
        rsc.maxerrata(verbose=False)

        # In order to extract the actual data, need to convert back to bits
        # Then take as many bytes as indicated by the message length indicator
        # That is AFTER removing the first 12 bytes (of enc and len)
        data_bits = bin(int.from_bytes(message_decoded[0], byteorder="big"))[
            13 : 13 + len_ * 8
        ]

        # Now convert back to bytes and print it lol
        data_bytes = int(data_bits, 2).to_bytes((len(data_bits) + 7) // 8, "big")
        print(f'Data in message = "{data_bytes.decode(encoding="iso-8859-1")}"')
    except rs.ReedSolomonError as e:
        msg_str = "".join(chars)

        msg_removed_illegal_chars = re.sub(r"[^\x00-\x7F]", "", msg_str)

        msg_raw_new_lines_removed = re.sub(r"[\r\n]", "", msg_removed_illegal_chars)

        # remove linebreaks
        clean_msg = re.sub(r"[\r\n\t]", "", msg_raw_new_lines_removed)

        print(f"Error decoding message: {e}, message = {clean_msg}")


if __name__ == "__main__":
    main()
