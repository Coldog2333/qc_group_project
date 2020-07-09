import numpy as np

def record_test_result_for_kaggle(y_pred, submission_file):
    start = 892
    end = 1309
    result = "PassengerId,Survived\n"
    for i in range(start, end + 1):
        result += "%d,%d\n" % (i, y_pred[i - 892])
    with open(submission_file, "w", encoding="utf-8") as writer:
        writer.write(result)


