import heapq
import itertools
import random

import tiktoken

from countdown_utils import combine_nums, CountdownNode, sum_heuristic, mult_heuristic, metric_fn

def bfs(target, nums, beam_size, heuristic=sum_heuristic):
    search_trace = ""
    open_set = []
    # Push the initial node with its index, heuristic value, and parent index
    heapq.heappush(open_set, (heuristic(nums, target), CountdownNode(0, None, nums, [], heuristic(nums, target))))
    node_index = 1  # Initialize node indexing 

    while open_set:
        # Get the top beam width nodes based on heuristic (pruning others)
        current_nodes = [heapq.heappop(open_set) for _ in range(min(beam_size, len(open_set)))]
        if not current_nodes:
            break  # Exit if no nodes are left to expand

        for idx, (_, current_node) in enumerate(current_nodes):
            search_trace += f"Current State: {target}:{current_node.nums}, Operations: {current_node.operations}\n"
            # Store nodes generated at this level for pruning
            generated_nodes = []
            # Generate successors for each node
            for i, j in itertools.combinations(range(len(current_node.nums)), 2):
                for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                    new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                    new_operations = current_node.operations + [operation]
                    new_heuristic = heuristic(new_nums, target)
                    new_node = CountdownNode(node_index, current_node, new_nums, new_operations, new_heuristic)
                    generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes
                    node_index += 1

            # Explicit Pruning: Keep only top 'beam_size' nodes, prune the rest
            generated_nodes.sort()  # Sort by heuristic value
            pruned_nodes = generated_nodes[beam_size:]  # Nodes that will be pruned
            generated_nodes = generated_nodes[:beam_size]  # Nodes that will be kept

           # shuffle the generated nodes so that the first node explored is not the same every time
            random.shuffle(generated_nodes)
            # Add remaining nodes back to open_set
            node_idx = 0
            for node_tuple in generated_nodes:
                node = node_tuple[-1]
                node.idx = f"{node.parent.idx},{node_idx}"
                operation = node.operations[-1]
                nums = node.nums
                search_trace += f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
                if len(node.nums) == 1 and node.nums[0] == target:
                    search_trace += f"{node.nums[0]},{target} equal: Goal Reached\n"
                    return search_trace
                elif len(new_node.nums) == 1:
                    search_trace += f"{node.nums[0]},{target} unequal: No Solution\n"
                else:
                    search_trace += f"Generated Node #{node.idx}: {target}:{node.nums} Operation: {operation}\n"
                    node_idx += 1
            generated_nodes.sort()
            for node_tuple in generated_nodes:
                heapq.heappush(open_set, node_tuple)
 
            # Note transition to the next node within the current set
            if idx < len(current_nodes) - 1:
                _, next_node = current_nodes[idx + 1]
                next_index = next_node.idx
                search_trace += f"Moving to Node #{next_index}\n"
            

        # Backtracking trace
        if current_nodes:
            # Find the next node to backtrack to (the next in the open set)
            next_node_to_explore = open_set[0] if open_set else None
            if next_node_to_explore:
                _, next_node = next_node_to_explore
                next_index = next_node.idx
                search_trace += f"Moving to Node #{next_index}\n"

    search_trace += "No solution found."
    return search_trace

if __name__ == "__main__":
    # Example usage
    random.seed(4)
    target = 24
    nums = [4,9,3]
    search_path = bfs(target, nums, 3, heuristic=mult_heuristic)
    print(search_path)
    print(len(search_path))
    print(metric_fn(search_path))

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(search_path)
    print(f"token length: {len(tokens)}")