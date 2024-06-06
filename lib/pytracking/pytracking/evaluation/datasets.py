from collections import namedtuple
import importlib
from pytracking.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

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

    esot500 = DatasetInfo(module=pt % "esot500", class_name="ESOT500Dataset",  kwargs=dict(split='test',fps=500)),

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
    tpl=DatasetInfo(module=pt % "tpl", class_name="TPLDataset", kwargs=dict()),
    tpl_nootb=DatasetInfo(module=pt % "tpl", class_name="TPLDataset", kwargs=dict(exclude_otb=True)),
    vot=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    trackingnetvos=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict(vos_mode=True)),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    got10kvos_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val', vos_mode=True)),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_train=DatasetInfo(module=pt % "lasot", class_name="LaSOTTrainSequencesDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
    lasotvos=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(vos_mode=True)),
    oxuva_dev=DatasetInfo(module=pt % "oxuva", class_name="OxUvADataset", kwargs=dict(split='dev')),
    oxuva_test=DatasetInfo(module=pt % "oxuva", class_name="OxUvADataset", kwargs=dict(split='test')),
    avist=DatasetInfo(module=pt % "avist", class_name="AVisTDataset", kwargs=dict()),
    dv2017_val=DatasetInfo(module="ltr.dataset.davis", class_name="Davis", kwargs=dict(version='2017', split='val')),
    dv2016_val=DatasetInfo(module="ltr.dataset.davis", class_name="Davis", kwargs=dict(version='2016', split='val')),
    dv2017_test_dev=DatasetInfo(module="ltr.dataset.davis", class_name="Davis",
                                kwargs=dict(version='2017', split='test-dev')),
    dv2017_test_chal=DatasetInfo(module="ltr.dataset.davis", class_name="Davis",
                                 kwargs=dict(version='2017', split='test-challenge')),
    yt2019_test=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                            kwargs=dict(version='2019', split='test')),
    yt2019_valid=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                             kwargs=dict(version='2019', split='valid')),
    yt2019_valid_all=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                                 kwargs=dict(version='2019', split='valid', all_frames=True)),
    yt2018_valid_all=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                                 kwargs=dict(version='2018', split='valid', all_frames=True)),
    yt2018_jjval=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                             kwargs=dict(version='2018', split='jjvalid')),
    yt2019_jjval=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                             kwargs=dict(version='2019', split='jjvalid', cleanup=['starts'])),
    yt2019_jjval_all=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                                 kwargs=dict(version='2019', split='jjvalid', all_frames=True, cleanup=['starts'])),
)


def load_dataset(name: str, **kwargs):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs, **kwargs)  # Call the constructor
    return dataset


def get_dataset(*args, **kwargs):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name, **kwargs).get_sequence_list())
    return dset


def get_dataset_attributes(name, mode='short', **kwargs):
    """ Get a list of strings containing the short or long names of all attributes in the dataset. """
    dset = load_dataset(name , **kwargs)
    dsets = {}
    if not hasattr(dset, 'get_attribute_names'):
        dsets[name] = get_dataset(name)
    else:
        for att in dset.get_attribute_names(mode):
            dsets[att] = get_dataset(name, attribute=att)

    return dsets