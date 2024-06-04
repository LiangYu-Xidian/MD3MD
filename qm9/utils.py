import torch


def compute_mean_mad(dataset, properties, dataset_name):
    if dataset_name == 'qm9' or dataset_name == 'qm9_second_half' or dataset_name == 'the_main_dataset':
        return compute_mean_mad_from_dataloader(dataset, properties)
    # elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
    #     return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataset, properties):
    property_norms = {}
    for property_key in properties:
        values = dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        max_value = torch.max(values)
        min_value = torch.min(values)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
        property_norms[property_key]['max'] = max_value
        property_norms[property_key]['min'] = min_value
    return property_norms


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    # batch_size = minibatch['batch'][-1] + 1
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # Global feature.
            # assert properties.size() == (batch_size,)
            properties = properties.index_select(0, minibatch['batch'])
            context_list.append(properties.unsqueeze(1))
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            # assert properties.size(0) == batch_size

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=1)
    # Mask disabled nodes!
    assert context.size(1) == context_node_nf
    return context
def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):

    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

