def get_tcn_receptive_field_size(kernel_size: int,
                                 levels: int,
                                 dilation_exponential_base: int = 2):
    """Calculate the receptive field size of a TCN.

    Due to: https://github.com/locuslab/TCN/issues/44#issuecomment-677949937

    Args:
        kernel_size (int): Size of the kernel.
        levels (int): Number of levels in the TCN.
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        int: Receptive field size.
    """

    return sum([
        2 * dilation_exponential_base**(l - 1) * (kernel_size - 1)
        for l in range(levels, 0, -1)
    ]) + 1
