# Type-1 克隆示例：精确克隆

# 原始代码片段
def calculate_sum(a, b):
    """计算两个数的和"""
    result = a + b
    return result

# 克隆代码片段1（完全相同）
def calculate_sum(x, y):
    """计算两个数的和"""
    result = x + y
    return result

# 克隆代码片段2（仅注释和空格不同）
def calculate_sum(a, b):
    """计算两个数的和"""
    result = a + b
    return result

# 克隆代码片段3（格式略有不同）
def calculate_sum(a,b):
    """计算两个数的和"""
    result = a + b
    return result

# 这些都是Type-1克隆，因为除了空白字符和注释外，代码结构完全相同
