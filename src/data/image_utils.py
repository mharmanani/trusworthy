import numpy as np


def convert_physical_coordinate_to_pixel_coordinate(
    physical_coordinates,
    physical_extent,
    physical_origin,
    pixel_extent,
):
    # convert everything to numpy arrays if they are not already
    physical_coordinates = np.array(physical_coordinates)
    physical_extent = np.array(physical_extent)
    physical_origin = np.array(physical_origin)
    pixel_extent = np.array(pixel_extent)

    # Normalize physical coordinates to [0, 1]
    normalized_coordinates = (physical_coordinates - physical_origin) / physical_extent
    # Convert normalized coordinates to pixel coordinates
    pixel_coordinates = normalized_coordinates * pixel_extent
    return pixel_coordinates


def sliding_window_slice_coordinates(
    window_size, strides, image_size, discard_from=None
):
    """Returns the slice coordinates for a sliding window of size window_size over
    and img of size img_size with stride lengths step_size.
    Args:
        window_size (tuple): the size of the sliding window
        strides (tuple): the stride lengths
        image_size (tuple): the size of the image
        discard_from (tuple):
            whether to discard from the start or end of the image. If None, defaults to "end" for all dimensions.
            If specified, must be a tuple of length len(window_size) with entries "start" or "end".
    Returns:
        np.ndarray: an array whose i'th entry contains the array whose j'th entry contains the slice coordinates in that dimension

    Example:
        >>> sliding_window_slice_coords((3, 3), (3, 3), (10, 10), discard_from=["end", "end"])
        array([[[0, 3],
                [3, 6]],
                [6, 9]],

                [[0, 3],
                [3, 6]],
                [6, 9]]])
        >>> x, y = sliding_window_slice_coords((3, 3), (3, 3), (10, 10), discard_from=["end", "end"])
        >>> A = np.fromfunction(lambda i, j: i + j, (10, 10))
        >>> x1, x2 = x[0]
        >>> y1, y2 = y[0]
        >>> A[x1:x2, y1:y2]
        array([[0., 1., 2.],
                [1., 2., 3.],
                [2., 3., 4.]])
    """

    inds = []
    assert len(window_size) == len(strides) == len(image_size)
    if discard_from is None:
        discard_from = ["end"] * len(window_size)
    for i in range(len(window_size)):
        startpos = np.arange(0, image_size[i], strides[i])
        while startpos[-1] + window_size[i] > image_size[i]:
            startpos = startpos[:-1]
        if discard_from[i] == "start":
            while startpos[-1] + window_size[i] < image_size[i]:
                startpos = startpos + 1
        inds.append(np.array([[j, j + window_size[i]] for j in startpos]))
    return tuple(inds)
