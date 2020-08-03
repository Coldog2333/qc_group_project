import numpy as np

def record_test_result_for_kaggle(y_pred, submission_file):
    start = 892
    end = 1309
    result = "PassengerId,Survived\n"
    for i in range(start, end + 1):
        result += "%d,%d\n" % (i, y_pred[i - 892])
    with open(submission_file, "w", encoding="utf-8") as writer:
        writer.write(result)

def convert_to_angle(b_st):
    if b_st[0] == 1:
        theta = np.arccos(1/np.sqrt(3))
    else:
        theta = np.arccos(-1/np.sqrt(3))

    if b_st[1] == 1 and b_st[2] == 1:
        varphi = np.pi / 4

    if b_st[1] == 1 and b_st[2] == 0:
        varphi = 3 * np.pi / 4

    if b_st[1] == 0 and b_st[2] == 0:
        varphi = -3 * np.pi / 4

    if b_st[1] == 0 and b_st[2] == 1:
        varphi = -np.pi / 4

    return [theta, varphi]

