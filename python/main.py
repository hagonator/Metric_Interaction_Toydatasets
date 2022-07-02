import torch

from utils import initialize, generate
from experimental_setup \
    import table_model_architectures, table_datasets, table_explanation_methods, table_evaluation_metrics

path = 'test2.pt'
generate(path)
# t, check = torch.load(path)
# torch.save([t, ['evaluating', [1, 1, 5, 0, 0]]], path)

# initialize(
#     path=path,
#     architectures=table_model_architectures,
#     datasets=table_datasets,
#     explanation_methods=table_explanation_methods,
#     evaluation_metrics=table_evaluation_metrics
# )
# t, check = torch.load(path)
# for i in range(6):
#     for j in range(8):
#         t[5][0][0][i][j] = [None for k in range(len(t[4][0][0]))]
#         t[5][1][0][i][j] = [None for k in range(len(t[4][1][0]))]
# check = ['evaluating', [0, 0, 0, 0, 0]]
# torch.save([t, check], path)
# train(t, [1, 1])
# torch.save([t, check], path)
# generate(path)
#
# t, check = torch.load(path)
# print(check)
# check = ['evaluating', [1,0,5,1,0]]
# torch.save([t,check], path)
# t[2] = table_explanation_methods
# t[5] = t[6]
# t = t[:6]
# torch.save([t, ['evaluating', [0, 0, 0, 0, 0]]], path)
# print(check)
# torch.save([t, ['explaining', [1, 1, 8, 0]]], path)
