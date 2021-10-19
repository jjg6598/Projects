
def binary_search(list_, element):

    first = 0 
    last = len(list_) - 1
    index = -1

    while first <= last and index == -1:
        mid = (first + last) // 2
        if list_[mid] == element:
            index = mid
        else:
            if element < list_[mid]:
                last = mid - 1
            else:
                first = mid + 1
    return index

nums = [10, 20, 30, 40, 50]
print(binary_search(nums, 20))

nums = [1,1,1,1,1]
print(binary_search(nums, 1))