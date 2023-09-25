from typing import List, Union

import networkx as nx

def get_receptive_field_size(kernel_size: int,
                             num_layers: int,
                             dilation_exponential_base: int = 2):
    """Calculate the receptive field size of a TCN. We assume the TCN structure of the paper
    from Bai et al.

    Due to: https://github.com/locuslab/TCN/issues/44#issuecomment-677949937

    Args:
        kernel_size (int): Size of the kernel.
        num_layers (int): Number of layers in the TCN.
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        int: Receptive field size.
    """

    return sum([
        2 * dilation_exponential_base**(l - 1) * (kernel_size - 1)
        for l in range(num_layers, 0, -1)
    ]) + 1

def create_graph(kernel_size: int, num_layers: int, input_size: Union[int, None] = None):
    """Create computation graph of a TCN network. Assumes the structure of the TCN as proposed by Bai et al.

    Args:
        kernel_size (int): Size of the kernel
        num_layers (int): Total number of layers in a TCN
        input_size (Union[int, None], optional): Maximum length of sequence as input into the TCN. Defaults to None.

    Returns:
        Tuple[nx.DiGraph, Dict[(int, int), (int. int)]: Tuple of NetworkX graph of the TCN and a dictionary of
        the positions of the nodes in the graph
    """

    G = nx.DiGraph()
    color_map = []

    if input_size is None:
        input_size = get_receptive_field_size(kernel_size, num_layers, 2)

    # Add nodes
    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            G.add_node((i, j))

            # Make the first layer yellow, the rest blue
            if j == 0:
                color_map.append('yellow')
            else:
                color_map.append('blue')

    # Create pos dictionary
    pos = {}

    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            pos[(i, j)] = (i, j)

    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            # Add edges
            # Always add edge to previous layer with the same i value
            # Also add edge to previous layer with i value of i - kernel_size * n, where n is the dilation
            # This dilation is calculated by the formula: dilation ** (num_layers - j - 1)

            # Add residual connection
            if j % 2 == 0 and j > 0:
                G.add_edge((i, j - 2), (i, j))

            if j > 0:
                # Add connection to previous layer
                G.add_edge((i, j - 1), (i, j))

                dilation = 2**((j + 1) // 2 - 1)

                for k in range(1, kernel_size):
                    if i - k * dilation >= 0:
                        G.add_edge((i - k * dilation, j - 1), (i, j))

    # Find all ancestors of the last node in the last layer
    ancestors = nx.ancestors(
        G, (input_size - 1, num_layers * 2)) | {(input_size - 1, num_layers * 2)}
    

    # TODO: include coloring as an option in this function
    # # Make all ancestors red
    # for node in ancestors:
    #     i, j = node

    #     color_map[i * (num_layers * 2 + 1) + j] = 'red'

    # Remove all nodes that are not ancestors
    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            if (i, j) not in ancestors:
                G.remove_node((i, j))

    return G, pos


def calc_MACs(G: nx.DiGraph, kernel_size: int, input_channels: int, intermediate_channels: List[int], input_size: Union[int, None] = None):
    total_MACs = 0

    num_layers = len(intermediate_channels)

    if input_size is None:
        input_size = get_receptive_field_size(kernel_size, num_layers, 2)

    if type(intermediate_channels[0]) == int:
        intermediate_channels = [[a, a] for a in intermediate_channels]

    # Flatten list
    intermediate_channels = [item for sublist in intermediate_channels for item in sublist]

    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            # Check if node exists
            if (i, j) not in G.nodes:
                continue

            effective_kernel_size = len(list(G.predecessors((i, j))))

            if effective_kernel_size > 0:
                if j == 1:
                    total_MACs += input_channels * intermediate_channels[0] * (effective_kernel_size + 1)
                else:
                    total_MACs += intermediate_channels[j-2] * intermediate_channels[j-1] * effective_kernel_size

    return total_MACs

        

def get_max_buffer_size(kernel_size: int, num_layers: int) -> int:
    """Get the max required buffer size of the TCN. Buffer size means maximum amount of activations that need
    to be kept in memory at the same  time. This function assumes that only one input is processed at a time
    (i.e. batch size is 1). For higher batch size, multiply the buffer size by the batch size.

    Args:
        kernel_size (int): Kernel size.
        num_layers (int): Number of layers.

    Returns:
        int: Max required buffer size.
    """

    return (2*num_layers-1)*(kernel_size-1)+kernel_size


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    k = 3
    total_layers = 4

    # Plot for kernel size 5 and levels 1 to 30 (inclusive) the buffer size on the y-axis with the receptive field size on the x-axis.
    X, y = zip(*[(get_receptive_field_size(k, l, 2), get_max_buffer_size(k, 2*l)) for l in range(1, total_layers)])

    plt.xlabel("Receptive Field Size")
    plt.ylabel("Buffer Size (1/channels)")
    # Set x-axis to be log scale
    plt.xscale("log")

    plt.plot(X, y)
    plt.show()