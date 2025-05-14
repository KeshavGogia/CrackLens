import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize

def skeleton_to_graph(skeleton):
    G = nx.Graph()
    rows, cols = skeleton.shape
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    for x in range(rows):
        for y in range(cols):
            if skeleton[x, y] == 1:
                G.add_node((x, y))
                for dx, dy in directions:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < rows and 0 <= ny_ < cols and skeleton[nx_, ny_] == 1:
                        weight = np.sqrt(2) if abs(dx) + abs(dy) == 2 else 1
                        G.add_edge((x, y), (nx_, ny_), weight=weight)
    return G

def find_longest_path(G):
    if not G.nodes:
        return 0.0, []
    try:
        start_node = list(G.nodes())[0]
        lengths = nx.single_source_dijkstra_path_length(G, start_node, weight='weight')
        farthest_node = max(lengths, key=lengths.get)
        lengths = nx.single_source_dijkstra_path_length(G, farthest_node, weight='weight')
        other_farthest_node = max(lengths, key=lengths.get)
        path = nx.shortest_path(G, source=farthest_node, target=other_farthest_node, weight='weight')
        crack_length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        return crack_length, path
    except Exception as e:
        print(f"Error finding longest path: {str(e)}")
        return 0.0, []

def estimate_crack_width(skeleton):
    try:
        widths = []
        rows, cols = skeleton.shape
        for x in range(rows):
            for y in range(cols):
                if skeleton[x, y] == 1:
                    left, right = 0, 0
                    while y - left - 1 >= 0 and skeleton[x, y - left - 1] == 1:
                        left += 1
                    while y + right + 1 < cols and skeleton[x, y + right + 1] == 1:
                        right += 1
                    local_width = left + right + 1
                    widths.append(local_width)
        if not widths:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'range': (0.0, 0.0)
            }
        arr = np.array(widths)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'range': (float(np.mean(arr) - 2*np.std(arr)), float(np.mean(arr) + 2*np.std(arr)))
        }
    except Exception as e:
        print(f"Error estimating crack width: {str(e)}")
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'range': (0.0, 0.0)
        }

def analyze_crack(mask):
    try:
        _, binary_mask = cv2.threshold(mask.astype(np.uint8), 127, 1, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary_mask.astype(bool)).astype(np.uint8)
        if np.count_nonzero(skeleton) == 0:
            return {
                'length': 0.0,
                'width_stats': {
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'range': (0.0, 0.0)
                },
                'skeleton': skeleton * 255,
                'graph': None
            }
        G = skeleton_to_graph(skeleton)
        crack_length, _ = find_longest_path(G)
        width_stats = estimate_crack_width(skeleton)
        return {
            'length': float(crack_length),
            'width_stats': width_stats,
            'skeleton': skeleton * 255,
            'graph': G
        }
    except Exception as e:
        print(f"Crack analysis error: {str(e)}")
        return {
            'length': 0.0,
            'width_stats': {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'range': (0.0, 0.0)
            },
            'skeleton': np.zeros_like(mask),
            'graph': None
        }

def plot_mask_and_graph_to_base64(mask, graph, title="Predicted Graph"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    fig = plt.figure(figsize=(10, 5))
    # Left: mask
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(mask, cmap='gray')
    ax1.set_title("Image")
    ax1.axis('off')
    # Right: graph
    ax2 = fig.add_subplot(1, 2, 2)
    if graph is not None and len(graph.nodes) > 0:
        pos = {node: (node[1], -node[0]) for node in graph.nodes()}
        nx.draw(graph, pos, node_size=10, with_labels=False, edge_color='b', alpha=0.7, ax=ax2)
    ax2.set_title(title)
    ax2.axis('off')
    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
