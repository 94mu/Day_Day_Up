'''两数之和

https://leetcode.cn/problems/two-sum/description/

'''

def twoSum(nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)-1):
            for x in range(i+1, len(nums)):
                if target == nums[i] + nums[x] :
                    return [i, x]