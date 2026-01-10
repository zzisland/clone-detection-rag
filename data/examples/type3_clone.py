# Type-3 克隆示例：语义克隆（结构修改）

# 原始代码片段：使用for循环求和
def sum_array_for(arr):
    """使用for循环计算数组元素和"""
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# 克隆代码片段1：使用while循环
def sum_array_while(arr):
    """使用while循环计算数组元素和"""
    total = 0
    i = 0
    while i < len(arr):
        total += arr[i]
        i += 1
    return total

# 克隆代码片段2：使用for-each循环
def sum_array_foreach(arr):
    """使用for-each循环计算数组元素和"""
    total = 0
    for element in arr:
        total += element
    return total

# 克隆代码片段3：使用内置函数
def sum_array_builtin(arr):
    """使用内置函数计算数组元素和"""
    return sum(arr)

# 克隆代码片段4：使用递归
def sum_array_recursive(arr, index=0):
    """使用递归计算数组元素和"""
    if index >= len(arr):
        return 0
    return arr[index] + sum_array_recursive(arr, index + 1)

# 这些都是Type-3克隆，因为功能相同但实现方式不同
