from utils import initialize, generate
from experimental_setup \
    import table_model_architectures, table_datasets, table_explanation_methods, table_evaluation_metrics

path = 'test.pt'
initialize(
    path=path,
    architectures=table_model_architectures,
    datasets=table_datasets,
    explanation_methods=table_explanation_methods,
    evaluation_metrics=table_evaluation_metrics
)
generate(path)
#
# t, check = torch.load(path)
# print(check)
# torch.save([t, ['explaining', [1, 1, 8, 0]]], path)
