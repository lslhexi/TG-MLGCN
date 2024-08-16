from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def precision_recall_f1_at_k(outputs, labels, k=5):
    """
    计算precision@k, recall@k, f1-score@k
    """
    topk_indices = np.argsort(outputs, axis=1)[:, -k:]

    # 生成与输出形状相同的0矩阵
    topk_outputs = np.zeros_like(outputs)

    # 将top-k位置的值设置为1
    for i in range(outputs.shape[0]):
        topk_outputs[i, topk_indices[i]] = 1

    precision = precision_score(labels, topk_outputs, average='samples')
    recall = recall_score(labels, topk_outputs, average='samples')
    f1 = f1_score(labels, topk_outputs, average='samples')

    return precision, recall, f1