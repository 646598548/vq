import torch
from collections import OrderedDict, defaultdict
from collections import deque, defaultdict
#LRU
class cache_sim:
    def __init__(self, mode, linesize, cachesize, num_lin, codebook):
        self.linesize = linesize
        self.num_lin = num_lin
        self.cache = torch.full((num_lin, linesize), -1)
        self.codebook = codebook
        
        # 预建 codebook 值到行的快速映射 (值 -> 行索引)
        self.codebook_value_map = {}
        for row_idx, row in enumerate(codebook):
            for val in row:
                val = val.item()
                if val not in self.codebook_value_map:
                    self.codebook_value_map[val] = row_idx
        
        # 使用 OrderedDict 实现高效 LRU 机制
        self.lru = OrderedDict()
        
        # 值到缓存行的反向映射 (值 -> 所在行集合)
        self.value_to_lines = defaultdict(set)
        # 缓存行到值的反向映射 (行 -> 存储的值集合)
        self.line_to_values = [set() for _ in range(num_lin)]

    def maintain_record(self, line):
        """ 更新 LRU 记录，O(1) 时间复杂度 """
        if line in self.lru:
            self.lru.move_to_end(line)
        else:
            self.lru[line] = None

    def is_empty(self):
        """ 返回可替换的行或 -1 """
        if len(self.lru) < self.num_lin:
            return -1
        else:
            # 弹出 LRU 行并清理其反向映射
            lru_line, _ = self.lru.popitem(last=False)
            for val in self.line_to_values[lru_line]:
                self.value_to_lines[val].remove(lru_line)
                if not self.value_to_lines[val]:
                    del self.value_to_lines[val]
            self.line_to_values[lru_line].clear()
            return lru_line

    def load_data(self, data):
        """ 加载数据到缓存，向量化操作 """
        data_val = data.item()
        if data_val not in self.codebook_value_map:
            raise ValueError(f"Value {data_val} not in codebook")
        
        code_row = self.codebook[self.codebook_value_map[data_val]]
        target_line = self.is_empty()
        
        if target_line == -1:
            # 寻找未使用的行
            for line in range(self.num_lin):
                if line not in self.lru:
                    target_line = line
                    break
            # 如果所有行已使用，触发 LRU 替换
            if target_line == -1:
                target_line = self.is_empty()
        # print(target_line)
        # 更新缓存行数据
        self.cache[target_line] = code_row
        # 更新反向映射
        new_values = set(code_row.tolist())
        for val in new_values:
            self.value_to_lines[val].add(target_line)
        self.line_to_values[target_line] = new_values
        self.maintain_record(target_line)

    def sim(self, read_data):
        """ 批量查询优化，O(1) 时间复杂度 """
        data_val = read_data.item()
        
        # 
        if data_val in self.value_to_lines:
            # 取第一个关联行并更新 LRU
            line = next(iter(self.value_to_lines[data_val]))
            self.maintain_record(line)
            return True
        else:
            self.load_data(read_data)
            # print('not hit')
            return False
#fifo
class cache_sim_fifo:
    def __init__(self, mode, linesize, cachesize, num_lin, codebook):
        self.linesize = linesize
        self.num_lin = num_lin
        self.cache = torch.full((num_lin, linesize), -1)
        self.codebook = codebook
        
        # 预建 codebook 值到行的快速映射 (值 -> 行索引)
        self.codebook_value_map = {}
        for row_idx, row in enumerate(codebook):
            for val in row:
                val = val.item()
                if val not in self.codebook_value_map:
                    self.codebook_value_map[val] = row_idx
        
        # FIFO 队列 (核心改动)
        self.fifo_queue = deque(maxlen=num_lin)  # 自动淘汰最早进入的项
        
        # 值到缓存行的反向映射 (值 -> 所在行集合)
        self.value_to_lines = defaultdict(set)
        # 缓存行到值的反向映射 (行 -> 存储的值集合)
        self.line_to_values = [set() for _ in range(num_lin)]

    def maintain_record(self, line):
        """ FIFO 只在加载时记录新行，不调整顺序 (核心改动) """
        if line not in self.fifo_queue:
            self.fifo_queue.append(line)

    def is_empty(self):
        """ 返回可替换的行或 -1 """
        if len(self.fifo_queue) < self.num_lin:
            return -1
        else:
            # 弹出 FIFO 队列最旧的行 (核心改动)
            lru_line = self.fifo_queue.popleft()
            # 清理反向映射
            for val in self.line_to_values[lru_line]:
                self.value_to_lines[val].remove(lru_line)
                if not self.value_to_lines[val]:
                    del self.value_to_lines[val]
            self.line_to_values[lru_line].clear()
            return lru_line

    def load_data(self, data):
        """ 加载数据到缓存 """
        data_val = data.item()
        if data_val not in self.codebook_value_map:
            raise ValueError(f"Value {data_val} not in codebook")
        
        code_row = self.codebook[self.codebook_value_map[data_val]]
        target_line = self.is_empty()
        
        if target_line == -1:
            # 寻找未使用的行 (新逻辑)
            for line in range(self.num_lin):
                if line not in self.fifo_queue:
                    target_line = line
                    break
            # 如果所有行已使用，触发 FIFO 替换
            if target_line == -1:
                target_line = self.is_empty()

        # 更新缓存行数据
        self.cache[target_line] = code_row
        # 更新反向映射
        new_values = set(code_row.tolist())
        for val in new_values:
            self.value_to_lines[val].add(target_line)
        self.line_to_values[target_line] = new_values
        self.maintain_record(target_line)  # 仅在此处更新队列

    def sim(self, read_data):
        """ 查询优化 (命中时不更新队列顺序) """
        data_val = read_data.item()
        if data_val in self.value_to_lines:
            # FIFO 不更新访问顺序 (核心改动)
            return True
        else:
            self.load_data(read_data)
            return False
        
#associative fifo
class cache_sim_set_associative_fifo:
    def __init__(self, mode, linesize, cachesize, num_lin, codebook, associativity=4):
        # 参数校验
        assert num_lin % associativity == 0, "Cache lines must be divisible by associativity"
        assert associativity >= 1, "Associativity must be at least 1"
        
        # 基础参数
        self.linesize = linesize
        self.num_lin = num_lin
        self.associativity = associativity
        self.groups = num_lin // associativity  # 计算组数
        self.cache = torch.full((num_lin, linesize), -1)
        self.codebook = codebook

        # 预建codebook值到行的映射
        self.codebook_value_map = {}
        for row_idx, row in enumerate(codebook):
            for val in row:
                val = val.item()
                if val not in self.codebook_value_map:
                    self.codebook_value_map[val] = row_idx

        # 为每个组维护FIFO队列
        self.group_queues = [deque(maxlen=associativity) for _ in range(self.groups)]

        # 反向映射数据结构
        self.value_to_lines = defaultdict(set)
        self.line_to_values = [set() for _ in range(num_lin)]

    def _get_replacement_line(self, group):
        """ 在指定组中获取需要替换的缓存行 """
        # 计算组内起始行号
        start = group * self.associativity
        # 优先寻找空闲行
        for line_offset in range(self.associativity):
            current_line = start + line_offset
            if current_line not in self.group_queues[group]:
                return current_line

        # 执行FIFO替换
        victim = self.group_queues[group].popleft()
        
        # 清理反向映射
        for val in self.line_to_values[victim]:
            self.value_to_lines[val].remove(victim)
            if not self.value_to_lines[val]:
                del self.value_to_lines[val]
        self.line_to_values[victim].clear()
        
        return victim

    def load_data(self, data):
        """ 加载数据到缓存 """
        data_val = data.item()
        if data_val not in self.codebook_value_map:
            raise ValueError(f"Value {data_val} not in codebook")

        # 获取对应的codebook行和组号
        code_row_idx = self.codebook_value_map[data_val]
        code_row = self.codebook[code_row_idx]
        group = code_row_idx % self.groups  # 关键组映射逻辑

        # 获取目标缓存行
        target_line = self._get_replacement_line(group)

        # 更新缓存内容
        self.cache[target_line] = code_row

        # 维护组队列
        self.group_queues[group].append(target_line)

        # 更新反向映射
        new_values = set(code_row.tolist())
        self.line_to_values[target_line] = new_values
        for val in new_values:
            self.value_to_lines[val].add(target_line)

    def sim(self, read_data):
        """ 模拟缓存访问 """
        data_val = read_data.item()
        if data_val in self.value_to_lines:
            return True  # 命中不更新队列
        else:
            self.load_data(read_data)
            return False