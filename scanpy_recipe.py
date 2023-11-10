# This module provides fast and easy used methods to do single cell preprocessing
# and analysis based on scanpy basic function
import click
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import warnings
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
# from bbknn import bbknn
import celltypist
import scanorama


def mark_doublet(adata):
    """
    run scrublet to mark doublets
    raw counts is wanted in adata.X
    """
    adata = adata.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sc.external.pp.scrublet(adata, batch_key='Batch', verbose=False)
    adata.obs["doublet_tag"] = adata.obs["predicted_doublet"].map(
        lambda b: "doublet" if b else "singlet")

    return adata


def create_raw_ad(adata, use_raw=None, layer=None):
    if layer:
        raw_ad = ad.AnnData(
            X=adata.layers[layer], obs=adata.obs, var=adata.var)
        raw_ad.obs_names = adata.obs_names
        raw_ad.var_names = adata.var_names
    elif use_raw:
        raw_ad = adata.raw.to_adata()
    else:
        raise Exception("Either use_raw or layer is needed!")

    return raw_ad


def plot_hvg(adata, markers):
    """
    Highly variable genes plot for residual variances method
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    hvgs = adata.var["highly_variable"]

    ax.scatter(
        adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none"
    )
    ax.scatter(
        adata.var["mean_counts"][hvgs],
        adata.var["residual_variances"][hvgs],
        c="tab:red",
        label="selected genes",
        s=3,
        edgecolor="none",
    )
    ax.scatter(
        adata.var["mean_counts"][np.isin(adata.var_names, markers)],
        adata.var["residual_variances"][np.isin(adata.var_names, markers)],
        c="k",
        label="known marker genes",
        s=10,
        edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("mean expression")
    ax.set_yscale("log")
    ax.set_ylabel("residual variance")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.legend()


def sc_setting(figdir=None, dpi=150, script=True):
    """
    basic settings for scanpy
    """
    if script:
        import matplotlib as mpl
        mpl.use("Agg")
    if figdir:
        sc.settings.figdir = figdir

    sc.set_figure_params(dpi=dpi, facecolor="white")


def basic_filter(adata: ad.AnnData,
                 remove_rp_mt=True,
                 pct_mt_cut=20,
                 save=None,
                 species="hsapiens",
                 filter_count=True,
                 high_count_threshold=None,
                 min_genes=200,
                 min_cells=3,
                 remove_rbc=None,
                 chromosome="MT",
                 show_fig=False,
                 use_raw=None,
                 layer=None,
                 ):
    """
    Usage:
    basic filter for single cell data
    raw counts need to be in the adata.X ,or indicate use_raw or layer 
    to mark where raw counts matrix is stored
    Args:
        adata: anndata
        remove_rp_mt: whether or not to remove RPL/RPS and MT genes
        pct_mt_cut: percentage to cut high mt genes' counts cell
        save: output figure if indicated, which can be suffix of picture name like _tag.png or None
        species: ensembl species name like [hsapiens,mmusculus], other species's MT can be searched 
                 by species and chromosome
        remove_rbc: remove red blood cell [None or number of percent: 5]

    Return:
        cleaned anndata
    """
    if use_raw or layer:
        adata = create_raw_ad(adata, use_raw=use_raw, layer=layer)

    adata = adata.copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # annotate the group of mitochondrial genes as 'mt'
    if species == "hsapiens":
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    elif species == "mmusculus":
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    else:
        mito_gene_names = sc.queries.mitochondrial_genes(
            species, chromosome=chromosome)
        adata.var['mt'] = adata.var_names.map(
            lambda g: True if (g in mito_gene_names) else False)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if show_fig:
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                     jitter=0.4, multi_panel=True, save=save)
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save=save)
        sc.pl.scatter(adata, x='total_counts',
                      y='n_genes_by_counts', save=save)
    if filter_count:
        count_99 = adata.obs['total_counts'].quantile(0.99)
        adata = adata[adata.obs.n_genes_by_counts < count_99, :]
    if high_count_threshold:
        adata = adata[adata.obs.n_genes_by_counts < high_count_threshold, :]

    adata = adata[adata.obs.pct_counts_mt < pct_mt_cut, :]

    if remove_rp_mt and (species in ["hsapiens", "mmusculus"]):
        adata.var["rp"] = [True if g.startswith("RPL") or g.startswith(
            "RPS") else False for g in adata.var_names]
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=['rp'], percent_top=None, log1p=False, inplace=True)
        if species == "hsapiens":
            # remove rpl/rps/mt
            filtered_genes = [False if (g.startswith("RPL") or g.startswith(
                "RPS") or g.startswith("MT-")) else True for g in adata.var_names]
            adata = adata[:, filtered_genes]
        elif species == "mmusculus":
            filtered_genes = [False if (g.startswith("rpl") or g.startswith(
                "rps") or g.startswith("mt-")) else True for g in adata.var_names]
            adata = adata[:, filtered_genes]
    if remove_rbc:
        if species in ["hsapiens", "mmusculus"]:
            filtered_genes = [
                False if (g.upper() == "HBB") else True for g in adata.var_names]
            adata = adata[:, filtered_genes]

    adata.layers["counts"] = csc_matrix(adata.X.copy())
    click.secho("save filtered counts to adata.layers['counts']", fg="green")
    print(adata)
    return adata


def remove_batch_effect(adata,
                        batch_col,
                        dataset_col,
                        batch_effect_method,
                        n_pcs,
                        neighbors_within_batch=3,
                        annoy_n_trees=10):
    """
    function to remove batch effect one or two times
    """
    if batch_effect_method == "harmony-bbknn":
        if (not batch_col) or (not dataset_col):
            click.secho("batch_col or dataset_col is None!", fg="red")
            exit(1)
        adata, use_rep = run_harmony_bbknn(
            adata, batch_name=batch_col, dataset_name=dataset_col, n_pcs=n_pcs, neighbors_within_batch=neighbors_within_batch,
            annoy_n_trees=annoy_n_trees)
    elif batch_effect_method == "harmony":
        adata, use_rep = run_harmony(
            adata, batch_name=batch_col, dataset_name=dataset_col)
    elif batch_effect_method == "scanorama":
        adata, use_rep = run_scanorama(
            adata, batch_name=batch_col, dataset_name=dataset_col)
    elif batch_effect_method == "bbknn":
        if dataset_col:
            click.secho(
                "dataset_col is not None,only batch_col will be used in bbknn method!", fg="red")
        adata, use_rep = run_bbknn(adata, batch_name=batch_col, n_pcs=n_pcs, neighbors_within_batch=neighbors_within_batch,
                                   annoy_n_trees=annoy_n_trees)

    return adata, use_rep


def run_bbknn(adata, batch_name, n_pcs, neighbors_within_batch=3,
              annoy_n_trees=10):
    # bbknn(adata, batch_key=batch_name, n_pcs=n_pcs)
    sc.external.pp.bbknn(adata, batch_key=batch_name, n_pcs=n_pcs, neighbors_within_batch=neighbors_within_batch,
                         annoy_n_trees=annoy_n_trees)
    # sc.tl.umap(adata)
    # sc.tl.leiden(adata, resolution=1)
    return adata, None


def run_harmony(adata, batch_name, dataset_name=None):
    use_rep = 'X_pca_harmony'
    if dataset_name:
        sc.external.pp.harmony_integrate(adata,
                                         dataset_name,
                                         basis='X_pca',
                                         adjusted_basis='X_pca_harmony_dataset')
        sc.external.pp.harmony_integrate(adata,
                                         batch_name,
                                         basis='X_pca_harmony_dataset',
                                         adjusted_basis=use_rep)
    else:
        sc.external.pp.harmony_integrate(adata,
                                         batch_name,
                                         basis='X_pca',
                                         adjusted_basis=use_rep)

    return adata, use_rep


def single_batch_effect_remove_by_scanorama(adata, batch_name):
    adatas = []
    for batch in adata.obs[batch_name].unique():
        subset_ad = adata[adata.obs[batch_name] == batch]
        adatas.append(subset_ad)
    scanorama.integrate_scanpy(adatas)
    concat_ad = ad.concat(adatas, merge='same')
    return concat_ad


def run_scanorama(adata, batch_name, dataset_name=None):
    use_rep = 'X_scanorama'

    if dataset_name:
        adata = single_batch_effect_remove_by_scanorama(adata, dataset_name)
        adata = single_batch_effect_remove_by_scanorama(adata, batch_name)
    else:
        adata = single_batch_effect_remove_by_scanorama(adata, batch_name)

    return adata, use_rep


def run_harmony_bbknn(adata, batch_name, dataset_name, n_pcs, neighbors_within_batch=3,
                      annoy_n_trees=10):
    """
    Run harmony between datasets and bbknn between samples
    """
    print("run harmony...")
    sc.external.pp.harmony_integrate(
        adata, dataset_name, basis='X_pca', adjusted_basis='X_pca_harmony')
    print("run bbknn...")
    sc.external.pp.bbknn(adata, batch_key=batch_name,
                         use_rep='X_pca_harmony',
                         n_pcs=n_pcs,
                         neighbors_within_batch=neighbors_within_batch,
                         annoy_n_trees=annoy_n_trees)
    # bbknn(adata, batch_key=batch_name,
    #       use_rep='X_pca_harmony', n_pcs=n_pcs)

    return adata, None


def basic_prep(adata: ad.AnnData,
               run_basic_filter=True,
               markers=[],
               pct_mt_cut=20,
               save=None,
               species="hsapiens",
               filter_count=True,
               high_count_threshold=None,
               min_genes=200,
               min_cells=3,
               remove_rp_mt=True,
               remove_rbc=None,
               chromosome="MT",
               regress_to=[],
               n_top_genes=2000,
               pcs=50,
               neighbors=15,  # knn neighbors range 1~100
               resolution=1,
               n_genes=25,
               flavor="seurat_v3",
               hvg_batch_key=None,
               batch_col=None,
               dataset_col=None,
               batch_effect_method="harmony",
               legend_loc="on data",
               raw_layer="counts",
               neighbors_within_batch=3,
               annoy_n_trees=10,  # bbknn neighbors
               DGE_method="wilcoxon",
               show_fig=False,
               use_raw=None,
               layer=None,
               ):
    """
    Usage:
    Input attention:
        raw counts need to be in the adata.X or indicate use_raw or layer 
        to mark where raw counts matrix is stored
    Neighbor set attention:
        If you need to run BBKNN, set annoy_n_trees to control the precision of KNN.Higher score
        with higher precison.
        Otherwise, set neighbors to control precision of KNN , range 1~100, higher score with
        lower precision
    Batch effect remove attention:
        Specify batch_col only if you are combining samples in one dataset
        Specify both batch_col and dataset_col simultaneously if you are combining different datasets, each containing multiple samples.
    Args:
    run_basic_filter: whether or not to remove low quality cells and genes 
    markers: other marker genes to plot umap 
    pct_mt_cut: cut off percentage of Mitochondria 
    save: suffix of figures to save
    species: species to find mitochondria genes
    chromosome: mitochondria chromosome name
    filter_count: whether or not to filter cells which counts higher than 99 percent 
    high_count_threshold:  filter cells higher than threshold by counts
    min_genes: filter cells lower than threshold by number of genes
    min_cells: filter genes lower than threshold
    remove_rp_mt: whether or not to remove RPS and Mitochondria genes
    regress_to: default not to run in case of overfit, e.g:['total_counts', 'pct_counts_mt']
    flavor: seurat_v3 / seurat
    batch_col: remove batch effect if provided
    dataset_col: remove batch effect twice with batch_col and dataset_col
    batch_effect_method: <harmony|bbknn|harmony-bbknn|scanorama>
    hgv_batch_key:  If specified, highly-variable genes are selected within each batch separately and merged. 
                    This simple process avoids the selection of batch-specific genes and acts as a lightweight 
                    batch correction method. For all flavors, genes are first sorted by how many batches they 
                    are a HVG. For dispersion-based flavors ties are broken by normalized dispersion. 
                    If flavor = 'seurat_v3', ties are broken by the median (across batches) rank based 
                    on within-batch normalized variance.
    neighbors_within_batch: BBKNN argument, How many top neighbours to report for each batch; total number of neighbours in the 
                            initial k-nearest-neighbours computation will be this number times the number of batches.
                             This then serves as the basis for the construction of a symmetrical matrix of connectivities.
    annoy_n_trees: BBKNN argument, Only used with annoy neighbour identification. The number of trees to construct in 
                   the annoy forest. More trees give higher precision when querying, at the cost of increased run time and resource intensity.
    """
    if use_raw or layer:
        adata = create_raw_ad(adata, use_raw=use_raw, layer=layer)
    adata = adata.copy()

    if dataset_col and (not batch_col):
        click.secho(
            f"batch_col should be indicated if dataset_col is not None!", fg="red")
        exit(1)
    if run_basic_filter:
        adata = basic_filter(adata,
                             remove_rp_mt=remove_rp_mt,
                             pct_mt_cut=pct_mt_cut,
                             save=save,
                             species=species,
                             filter_count=filter_count,
                             high_count_threshold=high_count_threshold,
                             min_genes=min_genes,
                             min_cells=min_cells,
                             remove_rbc=remove_rbc,
                             chromosome=chromosome,
                             show_fig=show_fig,)
    else:
        adata.layers["counts"] = csc_matrix(adata.X.copy())
        click.secho(
            "save filtered counts to adata.layers['counts']", fg="green")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata, flavor=flavor, n_top_genes=n_top_genes, batch_key=hvg_batch_key, layer=raw_layer)
    else:
        sc.pp.highly_variable_genes(
            adata, flavor=flavor, n_top_genes=n_top_genes, batch_key=hvg_batch_key)

    if show_fig:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sc.pl.highly_variable_genes(adata, save=save)
    # remove rpl/rps/mt by regress out
    if regress_to:
        sc.pp.regress_out(adata, regress_to)

    # adata.raw = adata
    adata.layers["norm"] = csc_matrix(adata.X.copy())

    sc.pp.scale(adata)

    sc.pp.pca(adata, n_comps=pcs)
    # default use_rep
    use_rep = "X_pca"
    if batch_col:
        if batch_effect_method in ["bbknn", "harmony-bbknn"]:
            sample_counts = adata.obs.value_counts([batch_col]).reset_index()
            sample_counts.columns = [batch_col, "counts"]
            small_samples = sample_counts[batch_col][sample_counts["counts"]
                                                     < neighbors_within_batch].to_list()
            if small_samples:
                click.secho(
                    f"bbknn will be used and samples with less than neighbors_within_batch={neighbors_within_batch} : {small_samples} will be removed!")
                adata = adata[~adata.obs[batch_col].isin(small_samples)]

        adata, use_rep = remove_batch_effect(adata,
                                             batch_col,
                                             dataset_col,
                                             batch_effect_method,
                                             n_pcs=pcs,
                                             neighbors_within_batch=neighbors_within_batch,
                                             annoy_n_trees=annoy_n_trees
                                             )
    if use_rep:
        sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep=use_rep)
    sc.tl.umap(adata)
    if isinstance(resolution, list):
        if 1 not in resolution:
            resolution.append(1)
        for r in resolution:
            if r == 1:
                sc.tl.leiden(adata, resolution=r, key_added=f"leiden")
            else:
                sc.tl.leiden(adata, resolution=r, key_added=f"leiden{r}")
    else:
        sc.tl.leiden(adata, resolution=resolution)
    if show_fig:
        sc.pl.umap(adata,
                   color=["leiden"] + markers,
                   legend_loc=legend_loc,
                   layer="norm",
                   ncols=4,
                   save=save)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sc.tl.rank_genes_groups(adata, groupby="leiden",
                                layer="norm", method=DGE_method, use_raw=False)
        if show_fig:
            sc.pl.rank_genes_groups(adata, n_genes=n_genes)

    return adata


def pr_prep(adata: ad.AnnData,
            run_basic_filter=True,
            remove_rp_mt=True,
            markers=[],
            theta=100,
            pct_mt_cut=20,
            n_top_genes=2000,
            pcs=50,
            neighbors=15,
            resolution=1,
            n_genes=25,
            save=None,
            species="hsapiens",
            filter_count=True,
            high_count_threshold=None,
            min_genes=200,
            min_cells=3,
            hvg_batch_key=None,
            batch_col=None,
            dataset_col=None,
            batch_effect_method="harmony",
            legend_loc="on data",
            raw_layer="counts",
            neighbors_within_batch=3,
            annoy_n_trees=10,
            DGE_method="wilcoxon",
            show_fig=False,
            use_raw=None,
            layer=None,
            ):
    """
    Pearson residuals preprocess
    arguments detail to see basic_prep
    """
    if use_raw or layer:
        adata = create_raw_ad(adata, use_raw=use_raw, layer=layer)
    adata = adata.copy()
    if run_basic_filter:
        adata = basic_filter(adata,
                             remove_rp_mt=remove_rp_mt,
                             pct_mt_cut=pct_mt_cut,
                             save=save,
                             species=species,
                             filter_count=filter_count,
                             high_count_threshold=high_count_threshold,
                             min_genes=min_genes,
                             min_cells=min_cells,
                             show_fig=show_fig,)
    else:
        adata.layers["counts"] = csc_matrix(adata.X.copy())
        click.secho(
            "save filtered counts to adata.layers['counts']", fg="green")

    adata.layers["norm"] = sc.pp.log1p(
        sc.pp.normalize_total(adata, inplace=False)["X"])
    click.secho("save normalized data to adata.layers['norm']", fg="green")
    # sc.experimental.pp.normalize_pearson_residuals(adata)
    sc.experimental.pp.highly_variable_genes(
        adata,
        flavor="pearson_residuals",
        n_top_genes=n_top_genes,
        theta=theta,
        batch_key=hvg_batch_key, layer=raw_layer)
    if show_fig:
        plot_hvg(adata, markers)

    sc.pp.pca(adata, n_comps=pcs)
    # default use_rep
    use_rep = "X_pca"
    if batch_col:
        if batch_effect_method in ["bbknn", "harmony-bbknn"]:
            sample_counts = adata.obs.value_counts([batch_col]).reset_index()
            sample_counts.columns = [batch_col, "counts"]
            small_samples = sample_counts[batch_col][sample_counts["counts"]
                                                     < neighbors_within_batch].to_list()
            if small_samples:
                click.secho(
                    f"bbknn will be used and samples with less than neighbors_within_batch={neighbors_within_batch} : {small_samples} will be removed!")
                adata = adata[~adata.obs[batch_col].isin(small_samples)]

        adata, use_rep = remove_batch_effect(adata,
                                             batch_col,
                                             dataset_col,
                                             batch_effect_method,
                                             n_pcs=pcs,
                                             neighbors_within_batch=neighbors_within_batch,
                                             annoy_n_trees=annoy_n_trees
                                             )
    if use_rep:
        sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep=use_rep)
    sc.tl.umap(adata)
    if isinstance(resolution, list):
        if 1 not in resolution:
            resolution.append(1)
        for r in resolution:
            if r == 1:
                sc.tl.leiden(adata, resolution=r, key_added=f"leiden")
            else:
                sc.tl.leiden(adata, resolution=r, key_added=f"leiden{r}")
    else:
        sc.tl.leiden(adata, resolution=resolution)
    if show_fig:
        sc.pl.umap(adata, color=["leiden"] + markers,
                   legend_loc=legend_loc, layer="norm", ncols=4, save=save)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sc.tl.rank_genes_groups(adata, groupby="leiden",
                                layer="norm", method=DGE_method, use_raw=False)
        if show_fig:
            sc.pl.rank_genes_groups(adata, n_genes=n_genes, save=save)

    return adata


def basic_paga(adata: ad.AnnData,
               group="leiden",
               markers=[],
               layer=None,
               use_raw=None,
               threshold=0.1):
    """
        markers = ["CD34", "AVP", "CD79B", "CD14", "GATA1", "CD3E"]

    """
    sc.set_figure_params(facecolor="white")
    adata = adata.copy()
    sc.tl.paga(adata, groups=group)
    sc.pl.paga(adata, color=[group] + markers, layout="fa",
               threshold=threshold)
    sc.tl.draw_graph(adata, init_pos='paga')
    sc.pl.draw_graph(adata, color=[group] +
                     markers, legend_loc='on data', layout="fa", layer=layer, use_raw=use_raw)
    sc.pl.paga_compare(adata, threshold=threshold)

    return adata


def run_celltypist(adata, model, over_clustering, save=None):
    """
    adata.X should be normalized matrix
    """
    prediction = celltypist.annotate(
        adata, model=model, majority_voting=True, over_clustering=over_clustering)
    predicted_ad = prediction.to_adata()
    sc.pl.umap(predicted_ad, color=[
               over_clustering, "predicted_labels", "majority_voting"], ncols=2, legend_loc="on data", save=save)
    return prediction, predicted_ad


def celltypist_marker_plot(adata, model_name="Immune_All_High.pkl", cluster="leiden", n_top_genes=5, search_top_genes=50, save=None):
    """
    ploting marker genes of predicted celltypes in celltypist model
    """
    models = celltypist.Model.load(model_name)
    ct_dict = {}
    for ct in adata.obs["majority_voting"].unique():
        markers = models.extract_top_markers(ct, top_n=search_top_genes)
        top_markers = []
        for m in markers:
            if m in adata.var_names:
                if len(top_markers) < n_top_genes:
                    top_markers.append(m)
        ct_dict[ct] = top_markers

    sc.pl.dotplot(adata, ct_dict, cluster, dendrogram=True, save=save)
    return ct_dict
