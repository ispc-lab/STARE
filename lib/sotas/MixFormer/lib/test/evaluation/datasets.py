from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    fe240 = DatasetInfo(module=pt % "fe240", class_name="FE240Dataset", kwargs=dict(split='test')),
    visevent = DatasetInfo(module=pt % "visevent", class_name="VisEventDataset", kwargs=dict(split='test')),
    visevent_stnet = DatasetInfo(module=pt % "visevent", class_name="VisEventDataset", kwargs=dict(split='test',version='stnet')),
    visevent_presence = DatasetInfo(module=pt % "visevent", class_name="VisEventDataset", kwargs=dict(split='test',version='presence')),
    visevent_172 = DatasetInfo(module=pt % "visevent", class_name="VisEventDataset", kwargs=dict(split='test',version='172')),
    eventcarla = DatasetInfo(module=pt % "eventcarla", class_name="EventCarlaDataset",  kwargs=dict(split='test')),
    esot_500_2 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=500,window=2)),
    esot_500_4 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=500,window=4)),
    esot_500_8 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=500,window=8)),
    esot_500_20 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=500,window=20)),
    esot_500_50 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=500,window=50)),

    esot_250_2 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=250,window=2)),
    esot_250_4 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=250,window=4)),
    esot_250_8 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=250,window=8)),
    esot_250_20 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=250,window=20)),
    esot_250_50 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=250,window=50)),

    esot_125_2 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=125,window=2)),
    esot_125_4 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=125,window=4)),
    esot_125_8 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=125,window=8)),
    esot_125_20 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=125,window=20)),
    esot_125_50 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=125,window=50)),
    
    esot_50_2 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=50,window=2)),
    esot_50_4 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=50,window=4)),
    esot_50_8 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=50,window=8)),
    esot_50_20 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=50,window=20)),
    esot_50_50 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=50,window=50)),

    esot_20_2 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=20,window=2)),
    esot_20_4 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=20,window=4)),
    esot_20_8 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=20,window=8)),
    esot_20_20 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=20,window=20)),
    esot_20_50 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=20,window=50)),

    esot500 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',variant='500')),

    esot500s = DatasetInfo(module=pt % "esot500Stream", class_name="ESOT500DatasetStream",  kwargs=dict(split='test')),
    esot2s = DatasetInfo(module=pt % "esot2Stream", class_name="ESOT2DatasetStream",  kwargs=dict(split='test')),

    # esot250 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',variant='250')),
    # esot125 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',variant='125')),
    # esot050 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',variant='050')),
    # esot020 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',variant='020')),
    esoth = DatasetInfo(module=pt % "esot500V", class_name="ESOT500VDataset",  kwargs=dict(split='test',variant='VariableH')),
    esotm = DatasetInfo(module=pt % "esot500V", class_name="ESOT500VDataset",  kwargs=dict(split='test',variant='VariableM')),
    esotl = DatasetInfo(module=pt % "esot500V", class_name="ESOT500VDataset",  kwargs=dict(split='test',variant='VariableL')),

    esot2_default = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=1, window=0)),
    esot2_2_20 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=2, window=20)),
    esot2_5_20 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=5, window=20)),
    esot2_10_20 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=10, window=20)),

    esot2_2_50 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=2, window=100)),
    esot2_5_50 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=5, window=100)),
    esot2_10_50 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=10, window=100)),

    esot2_2_100 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=2, window=100)),
    esot2_5_100 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=5, window=100)),
    esot2_10_100 = DatasetInfo(module=pt % "esot2", class_name="ESOT2Dataset",  kwargs=dict(split='test',interpolate=10, window=100)),
    
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
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict())
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