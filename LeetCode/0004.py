'''寻找两个正序数组的中位数

https://leetcode.cn/problems/median-of-two-sorted-arrays/description/

'''
def findMedianSortedArrays(nums1, nums2) -> float:
    # 计算两个列表的长度
    m = len(nums1)
    n = len(nums2)

    # 合并nums1和nums2
    for i in nums2:
        nums1.append(i)

    # 对合并和的列表进行排序
    nums1 = sorted(nums1)

    # 计算两个列表的长度
    m = len(nums1)
    n = len(nums2)

    # 合并nums1和nums2
    for i in nums2:
        nums1.append(i)

    # 对合并和的列表进行排序
    nums1 = sorted(nums1)
    
    # 求中位数
    lenN1N2 = m+n
    if (lenN1N2)%2 == 0:
        midNum = (nums1[lenN1N2//2 - 1] + nums1[lenN1N2//2])/2
    else:
        midNum = nums1[(lenN1N2+1)//2 - 1]
    
    return midNum

