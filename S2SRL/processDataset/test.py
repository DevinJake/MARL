#-*-coding:utf-8-*-
"""
This file is used to test certain functions or methods.
"""
def test():
    a = [2 - 1.5] * 10
    print (a)

    actions_t = slice(3)
    log_prob_v = [[1,2,3],[4,5,6],[7,8,9]]
    a = log_prob_v[actions_t]
    print(a)

    answer = 111
    orig_response = 222
    print (str(answer) + "::" + str(orig_response))

if __name__ == "__main__":
    test()