
import math 
import random

def jump_search(list_, element):

    list_len = len(list_)   #List length
    jump_size = int(math.sqrt(list_len))    #Jumpsize 
    left, right = 0, 0 #Initialize left, right bounds

    #print(jump_size)

    while left < list_len and list_[left] <= element:
        #increment right bound
        right = min(list_len - 1, left + jump_size)

        #check if element between left/right
        if list_[left] <= element and list_[right] >= element:
            break 
        
        #increment left bound
        left += jump_size

    #if left reaches list_len, element not in list
    if left >= list_len or list_[left] > element:
        return -1 

    #Assign final left/right values
    right = min(right, list_len - 1)
    i = left

    #Linear Search between final left/right values
    while i <= right and list_[i] <= element:
        if list_[i] == element:
            return i
        i += 1

    return False

random.seed(42)
nums = sorted(random.sample(list(range(100)), 16))
#print(nums)
results = jump_search(nums, 35)
print(f'At index {results} you can find {nums[results]}!')

