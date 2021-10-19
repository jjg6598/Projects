
def linear_search(list_, element):
    for i in range(len(list_)):
        if list_[i] == element:
            return i 

    return False


numbers = [1,2,3,42,3,4,21,1,2,4,32,42,34]
print(linear_search(numbers, 3))