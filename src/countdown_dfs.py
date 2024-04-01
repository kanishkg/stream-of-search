import itertools
import tiktoken

from countdown_utils import combine_nums, CountdownNode, sum_heuristic, mult_heuristic, metric_fn

def dfs(target, nums, heuristic=sum_heuristic, threshold=None, search_trace="", open_set=[]):
    if len(open_set) == 0:
        # Push the initial node with its index, heuristic value, and parent index
        open_set.append((heuristic(nums, target), CountdownNode(0, None, nums, [], heuristic(nums, target))))

    while open_set:
        # Sort open_set by heuristic value, then pop the best node (lowest heuristic)
        open_set.sort(key=lambda x: -x[0])
        _, current_node = open_set.pop()
        search_trace += f"Current State: {target}:{current_node.nums}, Operations: {current_node.operations}\n"

        # Generate successors for the current node
        generated_nodes = []
        for i, j in itertools.combinations(range(len(current_node.nums)), 2):
            node_index = 0
            for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                new_operations = current_node.operations + [operation]
                new_heuristic = heuristic(new_nums, target)
                new_node = CountdownNode(node_index, current_node, new_nums, new_operations, new_heuristic)
                generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes

        kept_nodes = []
        for g in generated_nodes:
            if threshold is None or g[0] <= threshold:
                kept_nodes.append(g)
            else:
                continue
        generated_nodes = kept_nodes
        generated_nodes.sort()

        node_index = 0
        for g, (_, new_node) in enumerate(generated_nodes):
            new_node.idx = f"{new_node.parent.idx},{node_index}"
            search_trace += f"Exploring Operation: {new_node.operations[-1]}, Resulting Numbers: {new_node.nums}\n"
            if len(new_node.nums) == 1 and new_node.nums[0] == target:
                search_trace += f"{new_node.nums[0]},{target} equal: Goal Reached\n"
                return search_trace
            elif len(new_node.nums) == 1:
                search_trace += f"{new_node.nums[0]},{target} unequal: No Solution\n"
            else:
                search_trace += f"Generated Node #{new_node.idx}: {target}:{new_node.nums} Operation: {new_node.operations[-1]}\n"
                new_set = [(new_heuristic, new_node)]
                search_trace += f"Moving to Node #{new_node.idx}\n"
                search_trace = dfs(target, nums, heuristic=heuristic, threshold=threshold, search_trace=search_trace, open_set=new_set)
                if "Goal Reached" in search_trace:
                    return search_trace
            node_index += 1
            if g < len(generated_nodes) - 1:
                next_index = new_node.parent.idx
                search_trace += f"Moving to Node #{next_index}\n"
                search_trace += f"Current State: {target}:{new_node.parent.nums}, Operations: {new_node.parent.operations}\n"

        # Backtracking trace
        if open_set:  # If there are still nodes to explore
            next_node = open_set[-1][1]  # Get the index of the next node to be explored
            next_index = next_node.idx
            search_trace += f"Moving to Node #{next_index}\n"

    return search_trace


if __name__ == "__main__":
    # Example usage
    target = 24
    nums = [8, 2, 3, 2, 1]
    search_path = dfs(target, nums, heuristic=mult_heuristic, threshold=target)
    print(search_path)
    print(len(search_path))
    print(metric_fn(search_path))
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(search_path)
    print(f"token length: {len(tokens)}")