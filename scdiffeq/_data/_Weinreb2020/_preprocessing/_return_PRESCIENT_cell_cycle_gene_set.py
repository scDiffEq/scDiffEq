
def _return_PRESCIENT_cell_cycle_gene_set(add=False):

    """"""

    cell_cycle_genes = [
        "Ube2c",
        "Hmgb2",
        "Hmgn2",
        "Tuba1b",
        "Ccnb1",
        "Tubb5",
        "Top2a",
        "Tubb4b",
    ]

    if add:
        if type(add) == list:
            for gene in add:
                cell_cycle_genes.append(gene)
        else:
            cell_cycle_genes.append(add)

    return cell_cycle_genes