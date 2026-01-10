# Type-2 克隆示例：语法克隆（重命名）

# 原始代码片段
def calculate_area(length, width):
    """计算矩形面积"""
    area = length * width
    return area

# 克隆代码片段1（变量重命名）
def compute_area(l, w):
    """计算矩形面积"""
    result = l * w
    return result

# 克隆代码片段2（函数名和变量重命名）
def get_rectangle_area(width, height):
    """计算矩形面积"""
    rectangle_area = width * height
    return rectangle_area

# 克隆代码片段3（参数名重命名）
def calculate_area(rect_length, rect_width):
    """计算矩形面积"""
    final_area = rect_length * rect_width
    return final_area

# 这些都是Type-2克隆，因为通过重命名标识符得到，语法结构相同
