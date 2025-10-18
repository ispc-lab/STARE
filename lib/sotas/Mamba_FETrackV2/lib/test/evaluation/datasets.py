from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),

    vot18=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
    coesot=DatasetInfo(module=pt % "coesot", class_name="COESOTDataset", kwargs=dict(split='test')),
    fe108=DatasetInfo(module=pt % "fe108", class_name="FE108Dataset", kwargs=dict(split='test')),
    felt=DatasetInfo(module=pt % "felt", class_name="FELTDataset", kwargs=dict(split='test')),

    esot500s=DatasetInfo(module=pt % "esot500Stream", class_name="ESOT500DatasetStream", kwargs=dict(split='test')),
    esot500hs=DatasetInfo(module=pt % "esot500Stream", class_name="ESOT500HDatasetStream", kwargs=dict(split='test')),

    esot500=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset", kwargs=dict(split='test', variant='500')),

    esot_500_2=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=500, window=2)),
    esot_500_4=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=500, window=4)),
    esot_500_8=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=500, window=8)),
    esot_500_20=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=500, window=20)),
    esot_500_50=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=500, window=50)),
    esot_500_100=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                             kwargs=dict(split='test', fps=500, window=100)),
    esot_500_150=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                             kwargs=dict(split='test', fps=500, window=150)),

    esot_250_2=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=250, window=2)),
    esot_250_4=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=250, window=4)),
    esot_250_8=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=250, window=8)),
    esot_250_20=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=250, window=20)),
    esot_250_50=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=250, window=50)),
    esot_250_100=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                             kwargs=dict(split='test', fps=250, window=100)),
    esot_250_150=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                             kwargs=dict(split='test', fps=250, window=150)),

    esot_125_2=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=125, window=2)),
    esot_125_4=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=125, window=4)),
    esot_125_8=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=125, window=8)),
    esot_125_20=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=125, window=20)),
    esot_125_50=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=125, window=50)),

    esot_50_2=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                          kwargs=dict(split='test', fps=50, window=2)),
    esot_50_4=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                          kwargs=dict(split='test', fps=50, window=4)),
    esot_50_8=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                          kwargs=dict(split='test', fps=50, window=8)),
    esot_50_20=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=50, window=20)),
    esot_50_50=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=50, window=50)),

    esot_20_2=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                          kwargs=dict(split='test', fps=20, window=2)),
    esot_20_4=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                          kwargs=dict(split='test', fps=20, window=4)),
    esot_20_8=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                          kwargs=dict(split='test', fps=20, window=8)),
    esot_20_20=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=20, window=20)),
    esot_20_50=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                           kwargs=dict(split='test', fps=20, window=50)),
    esot_20_100=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=20, window=100)),
    esot_20_150=DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",
                            kwargs=dict(split='test', fps=20, window=150)),

    # esoth_500_2=DatasetInfo(module=pt % "esot500", class_name="ESOT500HDataset",
    #                         kwargs=dict(split='fan_new_2', fps=500, window=2)),
    # esoth_500_20=DatasetInfo(module=pt % "esot500", class_name="ESOT500HDataset",
    #                          kwargs=dict(split='batch-3-20ms', fps=500, window=20)),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset