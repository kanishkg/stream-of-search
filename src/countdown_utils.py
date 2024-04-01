'''
Utility functions for countdown.py
'''
import re
import json

def combine_nums(a, b):
    # Implicitly makes assumptions about the order of operations and valid operations
    a = int(a)
    b = int(b)
    possible = [[a+b, f"{a}+{b}={a+b}"], [a*b, f"{a}*{b}={a*b}"]]
    if a <= b:
        possible.append([b-a, f"{b}-{a}={b-a}"])
        if a != 0 and b % a == 0:
            possible.append([b//a, f"{b}/{a}={round(b//a,0)}"])
    else:
        possible.append([a-b, f"{a}-{b}={a-b}"])
        if b != 0 and a % b == 0:
            possible.append([a//b, f"{a}/{b}={round(a//b,0)}"])
    return possible

class CountdownNode:
    def __init__(self, idx, parent, nums, operations, heuristic):
        self.nums = nums
        self.operations = operations
        self.heuristic = heuristic
        self.parent = parent
        self.idx = idx
    
    def __lt__(self, other):
        return self.heuristic < other.heuristic


# Heuristics functions
def sum_heuristic(nums, target):
    if len(nums) == 1:
        return abs(nums[0] - target)
    return sum(abs(num - target) for num in nums) / len(nums)

def mult_heuristic(nums, target):
    # get closer to factors of target
    # return sum([1 if (nums[i] == 0 or target % nums[i] == 0 or nums[i] % target == 0) else 0 for i in range(len(nums))])
    # softer version, with distance to factors
    factors = [i for i in range(2, target+1) if target % i == 0]
    return sum([min(abs(num - factor) for factor in factors) for num in nums])
    
# prune functions
def great_prune(heuristic, target):
    # Simple pruning based on result magnitude
    return heuristic > target

def mult_prune(result, target):
    # Prune if result is not close to a factor of target
    factors = [i for i in range(1, target+1) if target % i == 0]
    return all(abs(result - factor) > target for factor in factors)

def simple_rating(search_path):
    # Simple rating function based on number of operations
    nodes_explored = search_path.count("Exploring Operation") + 1
    return nodes_explored


def get_target_nums(search_path, mode=""):
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]
    search_path = search_path.replace("<|endoftext|>","")
    if mode == "dt":
        first_line = first_line.split("->")[1]
    target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
    if not target_nums_match:
        return "Invalid input: Cannot find the initial state in the first line."
    
    target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
    return target, nums


def parse_trajectory(search_path, mode="dt"):
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]
    search_path = search_path.replace("<|endoftext|>", "")

    # if mode == "dt":
    #     first_line = first_line.split("->")[1]
    target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
    if not target_nums_match:
        return "Invalid input: Cannot find the initial state in the first line."

    target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]

    # Extract the operations from the line that claims the goal is reached.
    goal_lines = re.finditer(r"\d+,\d+ equal: Goal Reached", search_path)
    goal_lines = list(goal_lines)
    if not goal_lines:
        return "No goal reached statement found."

    goal_line = goal_lines[0]
    # get the last operation line before the goal reached statement
    operations = re.findall(r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]",
                            search_path[:goal_line.start()])
    if not operations:
        return "No operations found leading to the goal."

    final_operation = operations[-1][0]
    try:
        predicted_result = int(final_operation.split('=')[1])
    except:
        print("couldnt parse last op", final_operation)
        return "Couldnt parse last op"
    if predicted_result != target:
        return "Invalid path: Final operation does not result in target."

    # get the last current state, operations before the goal reached statement, and extract the operations
    operation_list = re.findall(r"Current State: \d+:\[.*?\], Operations: \[(.*?)\]", search_path[:goal_line.start()])[
        -1].split(', ')
    operation_list = [op.replace("'", "") for op in operation_list]
    operation_list += [final_operation]

    # Verify each operation and keep track of the numbers involved
    available_numbers = nums
    for operation in operation_list:
        # Verify the operation
        try:
            left, right = operation.split('=')
        except:
            return f"Could not split operation into lhs, rhs"
        try:
            if eval(left) != int(right):
                return f"Invalid operation: {operation}"
        except Exception as e:
            return f"Error in evaluating operation {operation}: {e}"
        # get the numbers involved
        used_numbers = re.findall(r"\d+", left)
        for n in used_numbers:
            if int(n) not in available_numbers:
                return f"Invalid operation: {operation}, number {n} not available in {available_numbers}"

        available_numbers = [n for n in available_numbers if n not in used_numbers]
        available_numbers.append(int(right))

    return "Valid path."


def metric_fn(search_path, mode="dt"):
    rating = parse_trajectory(search_path, mode=mode)
    if rating == "Valid path.":
        score = simple_rating(search_path)
        first_line = search_path.strip().split('\n')[0]
        if "->" in first_line:
            first_line = first_line.split("->")[1]
        target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
        target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
        if len(nums) == 2:
            # 2c2 x ops (4) = 4
            max_nodes = 4
        elif len(nums) == 3:
            # 3c2 x ops (4) x 2c2 x ops (4) = 48
            max_nodes = 48
        elif len(nums) == 4:
            # 4c2 x ops (4) x 3c2 x ops (4) x 2c2 x ops (4) = 1152
            max_nodes = 1152
        elif len(nums) == 5:
            # 5c2 x ops (4) x 4c2 x ops (4) x 3c2 x ops (4) x 2c2 x ops (4) = 46080
            max_nodes = 46080
        return (max(1. - score / max_nodes, 0.0), rating)
    return (0., rating)
"""
UTIL FUNCTIONS FOR PARSING TRAJECTORIES AND RETURNING SEARCH TREE OBJECT 
"""


class SearchTreeNode:
  def __init__(self, options, op, id):
    # what options remain at this node
    self.options = options
    # operation needed to get to this node from its parent
    self.op = op
    # id of each node. root node has id tuple([0])
    self.id = id


class SearchTree:
  """Current implementation works for correct trees"""

  def __init__(self):
    self.data = dict()
    self.target = -1
    # if no errors were made and the goal was reached correctly
    self.correctness = 0
    # if the "goal reached" statement is in the trajectory
    self.goal_reached = 0
    self.final_op_correct = 0

    self.num_total_nodes = 0
    self.correct_path = []
    self.num_nodes_in_correct_path = 0
    self.rating = 0
    # maximum number of nodes of full tree
    self.max_nodes = 0

    # characterizing failures
    self.errors = dict({
      "arithmetic": dict(),
      "exploration": dict(),
      "formatting": dict(),
      "other": dict()
    })

  def print_tree(self, data, indent):
    if data == self.data:
      print(self.data)
      print('Node 0:  Target: ', self.target, 'Nums: ', self.data[0][0].options, 'Correctness: ', self.correctness,
            'Rating: ', self.rating)
      self.print_tree(data[0][1], 4)
    else:
      for key, node_and_dict in data.items():
        print(' ' * indent + str(key), end=': ')  # print node id at this branch
        if bool(node_and_dict[1]):
          print(node_and_dict[0].options, node_and_dict[0].op)  # Print information about node
          self.print_tree(node_and_dict[1], indent + 4)
        else:
          print(node_and_dict[0].options, node_and_dict[0].op)

  def parse_search_trajectory(self, search_path, mode="sft"):
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]
    search_path = search_path.replace("<|endoftext|>", "")
    # trim to the first goal reached statement
    if "Goal Reached" in search_path:
      index = search_path.find("Goal Reached")
      search_path = search_path[:index+len("Goal Reached")+1]

    target_nums_match = re.search(r"Current State: (\d+):\[(.*?)\]", first_line.replace("<|endoftext|>", ""))
    # target_nums_match = re.search(r"Current State: (\d+):\[(.*?)\]", first_line.replace("<|endoftext|>", ""))
    if not target_nums_match:
      return "Invalid input: Cannot find the initial state in the first line."

    target, nums = int(target_nums_match.group(0).split(':')[1][1:]), json.loads(target_nums_match.group(0).split(':')[2])
    self.target = target
    self.data[0] = [SearchTreeNode(nums, None, tuple([0])), dict()]

    if len(nums) == 3:
      self.max_nodes = 48
    if len(nums) == 4:
      # 48 * 24
      self.max_nodes = 1152

    # Extract the operations from the line that claims the goal is reached.
    goal_lines = re.finditer(r"\d+,\d+ equal: Goal Reached", search_path)
    goal_lines = list(goal_lines)
    if goal_lines:
      self.goal_reached = 1

    if self.goal_reached == 1:
      goal_line = goal_lines[0]
      # get the last operation line before the goal reached statement
      operations = re.findall(r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]", search_path[:goal_line.start()])
      if not operations:
        self.errors["exploration"][tuple([0])] = "No operations found leading to the goal."
        return
    else:
      operations = re.findall(r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]", search_path)
      if not operations:
        self.errors["exploration"][tuple([0])] = "No operations found."
        return

    # get the last current state, operations before the goal reached statement, and extract the operations
    if self.goal_reached == 1:
      search_path = search_path[:goal_line.end()]

    node_list = re.findall(
      r"(Generated) Node #\d+((,\d+)*)(\n)*( )*(Current State)*: (\d*):\[(.*)?\](,)* Operation(s)*: (\[)*(.*)(\])*",
      search_path)
    exploration_list = re.findall(
      r"(Moving to) Node #\d+((,\d+)*)(\n)*( )*(Current State)*: (\d*):\[(.*)?\]\n(,)*.*Operation(s)*: (\d+)[-+\/*](\d+)=(\d+)",
      search_path)
    # example of element in exploration_list:
    # ('Moving to', ',0,0', ',0', '\n', ' ', 'Current State', '31', "33, 2], Operations: ['91-58=33', '54-52=2'", '', '33', '2', '31')
    # eg. Generated Node #0,0,0: 47:[61, 24] Operation: 66-42=24 gives
    # ('Generated', ',0,0', ',0', '', '', '', '47', '61, 24', '', '', '', '66-42=24', '')

    # add the root node
    self.num_total_nodes += 1

    for node in node_list:
      # eg. Node #0,0,0 produces ',0,0'
      node_id = [int(num) for num in node[1].split(',')[1:]]
      if node[0] == 'Generated' and node_id != []:
        self.num_total_nodes += 1
        operation_str = node[-2]
        operation = re.search(r"\d+[-+*\/]\d+=\d+", operation_str)
        if not operation:
          self.errors["formatting"][tuple([0] + node_id)] = f"Invalid operation {operation_str}."
          break
          # string of form '66-42=24'
        op = operation.group(0)
        # check for arithmetic errors
        try:
          left, right = op.split('=')
        except:
          self.errors["formatting"][tuple([0] + node_id)] = f"Could not parse or split operation {op} into lhs, rhs."
        try:
          if eval(left) != int(right):
            self.errors["arithmetic"][tuple([0] + node_id)] = f"Invalid operation {op}."
        except Exception as e:
          if tuple([0] + node_id) not in self.errors["other"]:
            self.errors["other"][tuple([0] + node_id)] = [f"Error in evaluating operation {op}: {e}."]
          else:
            self.errors["other"][tuple([0] + node_id)].append([f"Error in evaluating operation {op}: {e}."])
        exploration_error_on_this_node = 0
        subdict = self.data[0]
        for key in node_id[:-1]:
          if not key in subdict[1]:
            self.errors["exploration"][tuple([0] + node_id)] = f"Exploration is jumping steps. {key} does not exist in {subdict[1]}."
            exploration_error_on_this_node = 1
            break
          subdict = subdict[1][key]

        if exploration_error_on_this_node == 0:
          # check that it got the correct remaining nums from its parent node and used them
          used_numbers = re.findall(r"\d+", left)
          parent_options = subdict[0].options.copy()
          for used_num in used_numbers:
            if not int(used_num) in parent_options:
              if tuple([0] + node_id) not in self.errors["other"]:
                self.errors["other"][tuple([0] + node_id)] = [
                  f"Invalid operation: {op}, number {used_num} not available in {parent_options}."]
              else:
                self.errors["other"][tuple([0] + node_id)].append(
                  f"Invalid operation: {op}, number {used_num} not available in {parent_options}")

          for used_num in used_numbers:
            if int(used_num) in parent_options:
              parent_options.remove(int(used_num))
          try:
            remaining_options = list(int(num) for num in node[7].split(','))
          except:
            self.errors["formatting"][tuple([0] + node_id)] = f"Could not parse or split remaining options {node[7]}."
            continue
          if not set([number for number in parent_options if number not in used_numbers] + [int(right)]):
            if tuple([0] + node_id) not in self.errors["other"]:
              self.errors["other"][tuple([0] + node_id)] = [
                f"Error in remaining options: {subdict[0].options} should be {parent_options + [int(right)]} instead."]
            else:
              self.errors["other"][tuple([0] + node_id)].append(
                f"Error in remaining options: {subdict[0].options} should be {parent_options + [int(right)]} instead.")

          new_node = SearchTreeNode(list(int(num) for num in node[7].split(',')), op, tuple([0] + node_id))
          # trying to generate a node that has already been generated
          if node_id[-1] not in subdict[1]:
            subdict[1][node_id[-1]] = [new_node, dict()]
          else:
            self.errors["exploration"][tuple([0] + node_id)] = [
              f"Trying to generate node with index that has already been generated: {tuple([0] + node_id)} already exists."
            ]

    # add the unsuccessful explorations as nodes eg.
    # "Exploring Operation: 55-4=51, Resulting Numbers: [51]
    # 51,53 unequal: No Solution" as these are not associated with the phrase "Generated Node" and wouldn't be caught earlier
    failed_final_explorations = re.findall(
      r"Moving to Node \#((\d+,)*\d+)\n.*\nExploring Operation: (\d+)([-+\/*])(\d+)=(\d+).*\n.*unequal: No Solution",
      search_path)
    for exploration in failed_final_explorations:
      node_id = [int(num) for num in exploration[0].split(',')[1:]]
      op_num_1, operator, op_num_2, op_num_result = int(exploration[-4]), exploration[-3], int(exploration[-2]), int(exploration[-1])
      try:
        if (operator == "/" and op_num_2 == 0) or eval(f"{op_num_1}{operator}{op_num_2}") != int(op_num_result):
          self.errors["arithmetic"][tuple([0] + node_id)] = f"Invalid operation {op_num_1}{operator}{op_num_2}={op_num_result}."
      except Exception as e:
        if tuple([0] + node_id) not in self.errors["other"]:
          self.errors["other"][tuple([0] + node_id)] = [
            f"Error in evaluating operation {op_num_1}{operator}{op_num_2}={op_num_result}: {e}."]
        else:
          self.errors["other"][tuple([0] + node_id)].append([f"Error in evaluating operation {op_num_1}{operator}{op_num_2}={op_num_result}: {e}."])

      exploration_error = 0
      self.num_total_nodes += 1
      subdict = self.data[0]
      for key in node_id[:-1]:
        if not key in subdict[1]:
          self.errors["exploration"][tuple([0] + node_id)] = f"Exploration is jumping steps. {key} does not exist in {subdict[1]}."
          break
        subdict = subdict[1][key]
      if node_id:
        if node_id[-1] in subdict[1]:
          subdict = subdict[1][node_id[-1]]
        else:
          self.errors["exploration"][tuple([0] + node_id)] = \
            f"Exploration is jumping steps. {node_id[-1]} does not exist in {subdict[1]}."
          exploration_error = 1
      if exploration_error == 0:
        op = f'{op_num_1}{operator}{op_num_2}={op_num_result}'

        op_node = SearchTreeNode([op_num_result], op, tuple([0] + node_id))
        if subdict[1]:
          max_leaf_index = max(subdict[1].keys())
          subdict[1][max_leaf_index + 1] = [op_node, dict()]
          node_id = node_id + [max_leaf_index + 1]
        else:
          subdict[1][0] = [op_node, dict()]
          node_id = node_id + [0]
        if (operator == "/" and op_num_2 == 0) or eval(f"{op_num_1}{operator}{op_num_2}") != op_num_result:
          self.errors["arithmetic"][tuple([0] + node_id)] = f"Invalid operation: {op}."

        # check that it got the correct remaining nums from its parent node and used them
        used_numbers = [op_num_1, op_num_2]
        parent_options = subdict[0].options.copy()
        # print(f"node: {[0] + node_id},  used nums: {used_numbers},  parent options: {parent_options}")
        for used_num in used_numbers:
          if not int(used_num) in parent_options:
            if tuple([0] + node_id) not in self.errors["other"]:
              self.errors["other"][tuple([0] + node_id)] = [f"Invalid operation: {op}, number {used_num} not available in {parent_options}."]
            else:
              self.errors["other"][tuple([0] + node_id)].append(f"Invalid operation: {op}, number {used_num} not available in {parent_options}")

        for used_num in used_numbers:
          if int(used_num) in parent_options:
            parent_options.remove(int(used_num))
        # options should be empty at this point
        if parent_options:
          if tuple([0] + node_id) not in self.errors["other"]:
            self.errors["other"][tuple([0] + node_id)] = [
              f"Error in remaining options: remaining nums should be [] instead."]
          else:
            self.errors["other"][tuple([0] + node_id)].append(
              f"Error in remaining options: remaining nums should be [] instead.")

    # add a goal reached node if exists and is correct
    if exploration_list and self.goal_reached:
      last_exploration = exploration_list[-1]
      node_id = [int(num) for num in last_exploration[1].split(',')[1:]]
      self.num_total_nodes += 1
      subdict = self.data[0]
      exploration_error_on_final_node = 0
      for key in node_id[:-1]:
        if not key in subdict[1]:
          self.errors["exploration"][tuple([0] + node_id)] \
            = f"Exploration is jumping steps. {key} does not exist in {subdict[1]}."
          exploration_error_on_final_node = 1
          break
        subdict = subdict[1][key]
      if node_id:
        if node_id[-1] in subdict[1]:
          subdict = subdict[1][node_id[-1]]
        else:
          self.errors["exploration"][tuple([0] + node_id)] \
            = f"Exploration is jumping steps. {node_id[-1]} does not exist in {subdict[1]}."
          exploration_error_on_final_node = 1
      if exploration_error_on_final_node == 0:
        final_operation = operations[-1][0]
        predicted_result = int(final_operation.split('=')[1])
        matches = re.search(r"(\d+)([-+/*])(\d+)", final_operation.split('=')[0])
        if subdict[1]:
          max_leaf_index = max(subdict[1].keys())
          self.path = [0] + node_id + [max_leaf_index + 1]
          if matches.group(2) == '/' and matches.group(3) == '0':
            self.errors["arithmetic"][tuple(self.path)] = f"Invalid final operation: {final_operation}."
            return
          if predicted_result == eval(final_operation.split('=')[0]) and predicted_result == self.target:
            self.final_op_correct = 1
          if predicted_result != eval(final_operation.split('=')[0]) and predicted_result == self.target:
            self.errors["arithmetic"][tuple(self.path)] = f"Invalid final operation: {final_operation}."
            return
          if self.final_op_correct == 1:
            final_op_node = SearchTreeNode([target], final_operation, tuple(self.path))
            subdict[1][max_leaf_index + 1] = [final_op_node, dict()]
        else:
          self.path = [0] + node_id + [0]
          if matches.group(2) == '/' and matches.group(3) == '0':
            self.errors["arithmetic"][tuple(self.path)] = f"Invalid final operation: {final_operation}."
            return
          if predicted_result == eval(final_operation.split('=')[0]) and predicted_result == self.target:
            self.final_op_correct = 1
          if predicted_result != eval(final_operation.split('=')[0]) and predicted_result == self.target:
            self.errors["arithmetic"][tuple(self.path)] = f"Invalid final operation: {final_operation}."
            return
          if self.final_op_correct == 1:
            final_op_node = SearchTreeNode([target], final_operation, tuple(self.path))
            subdict[1][0] = [final_op_node, dict()]
            # print("added final op node", final_op_node.id)

    if self.goal_reached == 1 and self.final_op_correct == 1:
      # verify that each operation in path used is correct
      nodes_in_path = []
      for i in range(1, len(self.path)):
        nodes_in_path.append(self.path[:i + 1])
      for subdict in self.errors.values():
        for node_id in nodes_in_path:
          if tuple(node_id) in subdict:
            self.correctness = -1
      if self.correctness == -1:
        self.correctness = 0
      else:
        self.correctness = 1

    if self.correctness == 1:
      self.num_nodes_in_correct_path = len(self.path)
      self.rating = self.correctness - (self.num_total_nodes / self.max_nodes)

    return

  def get_all_valid_nodes(self, data, set_valid_nodes):
    # used in get_node_alignment
    if data == self.data:
      # root node is counted
      set_valid_nodes.add(tuple(sorted(list(data[0][0].options))))
      self.get_all_valid_nodes(data[0][1], set_valid_nodes)
    else:
      for key, node_and_dict in data.items():
        # check node is valid
        if not any(node_and_dict[0].id in error_type for error_type in self.errors.values() if
                   isinstance(error_type, dict)):
          if type(node_and_dict[0].options) is list:
            set_valid_nodes.add(tuple(sorted(node_and_dict[0].options)))
          else:
            set_valid_nodes.add(tuple(sorted([node_and_dict[0].options])))
        if bool(node_and_dict[1]):
          self.get_all_valid_nodes(node_and_dict[1], set_valid_nodes)

def get_node_alignment(search_tree, symbolic_search_tree) -> float:
  # expects 2 SearchTree objects. checks how many of the same nodes have been visited in 2 trajectories
  # Alignment = nodes in both (search_tree, symbolic_search_tree) / max_nodes(search_tree, symbolic_search_tree)
  # erroneous nodes in search_tree are thrown out
  # 2 nodes are considered the same if the list of numbers in that state are the same
  valid_nodes_t1 = set()
  valid_nodes_t2 = set()
  search_tree.get_all_valid_nodes(search_tree.data, valid_nodes_t1)
  symbolic_search_tree.get_all_valid_nodes(symbolic_search_tree.data, valid_nodes_t2)
  # for debugging
  # print("Search trace nodes:", valid_nodes_t1)
  # print("Symbolic search trace nodes:", valid_nodes_t2)
  intersection = valid_nodes_t1.intersection(valid_nodes_t2)
  max_nodes = max(len(valid_nodes_t1), len(valid_nodes_t2))
  return len(intersection) / max_nodes
    
if __name__ == "__main__":
    trajectory = """Current State: 47:[42, 61, 66], Operations: []
Exploring Operation: 66-42=24, Resulting Numbers: [61, 24]
Generated Node #0,0: 47:[61, 24] Operation: 66-42=24
Moving to Node #0,0
Current State: 47:[61, 24], Operations: ['66-42=24']
Exploring Operation: 61-24=37, Resulting Numbers: [37]
37,47 unequal: No Solution
Moving to Node #0,0
Current State: 47:[61, 24], Operations: ['66-42=24']
Exploring Operation: 61+24=85, Resulting Numbers: [85]
85,47 unequal: No Solution
Moving to Node #0
Current State: 47:[42, 61, 66], Operations: []
Exploring Operation: 61-42=19, Resulting Numbers: [66, 19]
Generated Node #0,1: 47:[66, 19] Operation: 61-42=19
Moving to Node #0,1
Current State: 47:[66, 19], Operations: ['61-42=19']
Exploring Operation: 66-19=47, Resulting Numbers: [47]
47,47 equal: Goal Reached47,47 equal: Goal Reached"""
    result = parse_trajectory(trajectory, mode="sft")
    print(result)