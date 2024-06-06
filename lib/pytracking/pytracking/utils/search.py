
def interpolation_search(arr, target):
    """returns the left boundary of target"""
    target = target
    low = 0
    high = len(arr) - 1
    # pos = 0
    
    while low <= high and arr[low] <= target <= arr[high]:
        pos = int(low + (target - arr[low]) * (high - low) // (arr[high] - arr[low]))
        # if arr[pos] == target:
        #     return pos
        if arr[pos] <= target:
            low = pos + 1
        else:
            high = pos - 1
            
    return pos

if __name__=='__main__':
    arr = [0,0,1,1,2,3,4,5,6,7,8,8,8,8,8,8,8,9,10,10,12,12,13]
    target = 15
    res = interpolation_search(arr, target)
    print(res)
    print(arr[res])