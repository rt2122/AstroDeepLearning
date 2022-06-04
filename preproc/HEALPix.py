import numpy as np


def recursive_fill(matr: np.ndarray) -> None:
    """Fill matrix recursively to generate correspondence matrix for HEALPix nested scheme.

    :param matr: Input matrix.
    :type matr: np.ndarray
    :rtype: None
    """
    if matr.shape[0] == 1:
        return

    mid = matr.shape[0] // 2
    np.left_shift(matr, 1, out=matr)
    matr[mid:, :] += 1

    np.left_shift(matr, 1, out=matr)
    matr[:, mid:] += 1

    for i in [0, mid]:
        for j in [0, mid]:
            recursive_fill(matr[i:i+mid, j:j+mid])


def one_pixel_fragmentation(o_nside: int, o_pix: int, f_nside: int) -> np.ndarray:
    """Get correspondence matrix between two nsides for chosen pixel. Nested scheme HEALPix.

    :param o_nside: Larger partition nside.
    :type o_nside: int
    :param o_pix: Pixel index for larger nside.
    :type o_pix: int
    :param f_nside: Smaller partition nside.
    :type f_nside: int
    :rtype: np.ndarray
    """
    depth = int(np.log2(f_nside / o_nside))
    m_len = 2 ** depth
    f_matr = np.full((m_len, m_len), o_pix)
    recursive_fill(f_matr)
    return f_matr
