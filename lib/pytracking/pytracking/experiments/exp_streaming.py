from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting

def esot2_streaming_1():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('kys', 'default') + \
                trackerlist('rts','rts50')
    dataset = get_dataset('esot2s')
    stream_setting = load_stream_setting('s1')
    return trackers, dataset, stream_setting

def esot2_streaming_2():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('kys', 'default') + \
                trackerlist('rts','rts50') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('kys', 'fe240') + \
                trackerlist('tomp','tomp50_fe240')
    dataset = get_dataset('esot2s')
    stream_setting = load_stream_setting('s2')
    return trackers, dataset, stream_setting

def streaming_3():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s3')
    return trackers, dataset, stream_setting

def streaming_4():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s4')
    return trackers, dataset, stream_setting

def streaming_24():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('kys', 'default') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s24')
    return trackers, dataset, stream_setting
def streaming_5():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s5')
    return trackers, dataset, stream_setting

def streaming_6():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s6')
    return trackers, dataset, stream_setting

def streaming_7():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s7')
    return trackers, dataset, stream_setting

def streaming_8():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s8')
    return trackers, dataset, stream_setting

def streaming_9():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'prdimp18_fe240') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s9')
    return trackers, dataset, stream_setting

def streaming_14():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s14')
    return trackers, dataset, stream_setting

def streaming_18():
    trackers =  trackerlist('atom', 'default') + \
                trackerlist('atom', 'fe240') + \
                trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_fe240') +\
                trackerlist('dimp', 'prdimp18') +\
                trackerlist('dimp', 'dimp50') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'fe240') + \
                trackerlist('rts','rts50') + \
                trackerlist('keep_track','default') + \
                trackerlist('tomp','tomp50')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s18')
    return trackers, dataset, stream_setting

trackers_range_default = trackerlist('atom', 'default',range(3)) + \
            trackerlist('dimp', 'dimp18',range(3)) + \
            trackerlist('dimp', 'prdimp18',range(3)) +\
            trackerlist('dimp', 'dimp50',range(3)) + \
            trackerlist('kys', 'default',range(3)) + \
            trackerlist('rts','rts50',range(3)) + \
            trackerlist('keep_track','default',range(3)) + \
            trackerlist('tomp','tomp50',range(3))

trackers_range = trackers_range_default + \
            trackerlist('atom', 'fe240',range(3)) + \
            trackerlist('dimp', 'dimp18_fe240',range(3)) +\
            trackerlist('kys', 'fe240',range(3))  +\
            trackerlist('ostrack', 'baseline',range(3))  +\
            trackerlist('ostrack', 'aug',range(3))   +\
            trackerlist('mixformer_convmae_online', 'baseline',range(3))   +\
            trackerlist('stark_s', 'baseline', range(3)) 


trackers_single =  trackerlist('atom', 'default') + \
            trackerlist('atom', 'fe240') + \
            trackerlist('dimp', 'dimp18') + \
            trackerlist('dimp', 'dimp18_fe240') +\
            trackerlist('dimp', 'prdimp18') +\
            trackerlist('dimp', 'dimp50') + \
            trackerlist('kys', 'default') + \
            trackerlist('kys', 'fe240') + \
            trackerlist('rts','rts50') + \
            trackerlist('keep_track','default') + \
            trackerlist('tomp','tomp50')

def streaming_18_range():
    trackers =  trackers_range
    # dataset = get_dataset('esot500s','esot2s')
    dataset = get_dataset('esot500s')
    stream_setting = load_stream_setting('s18')
    return trackers, dataset, stream_setting

def streaming_31():
    trackers =  trackers_single
    # dataset = get_dataset('esot500s','esot2s')
    dataset = get_dataset('esot500s')
    stream_setting = load_stream_setting('s31')
    return trackers, dataset, stream_setting

def streaming_32():
    trackers =  trackers_single
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s32')
    return trackers, dataset, stream_setting

def streaming_33():
    trackers =  trackers_single
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s33')
    return trackers, dataset, stream_setting

def streaming_34():
    trackers =  trackers_single
    dataset = get_dataset('esot500s')
    stream_setting = load_stream_setting('s34')
    return trackers, dataset, stream_setting

def streaming_35():
    trackers =  trackers_single
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s35')
    return trackers, dataset, stream_setting

# def streaming_36():
#     trackers =  trackers_range
#     dataset = get_dataset('esot500s','esot2s')
#     stream_setting = load_stream_setting('s34')
#     return trackers, dataset, stream_setting

def streaming_egt():
    trackers =  trackerlist('egt', 'egt') 
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('egt')
    return trackers, dataset, stream_setting

def streaming_egt_2():
    trackers =  trackerlist('egt', 'egt') 
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('egt_sim')
    return trackers, dataset, stream_setting

def streaming_egt_3():
    trackers =  trackerlist('egt', 'egt') 
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('egt_fast')
    return trackers, dataset, stream_setting

# =============  sotas ============


def streaming_sotas():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline')
    dataset = get_dataset('esot2s')
    stream_setting = load_stream_setting('s2')
    return trackers, dataset, stream_setting
def streaming_sotas_s3():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s3')
    return trackers, dataset, stream_setting
def streaming_sotas_s4():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s4')
    return trackers, dataset, stream_setting
def streaming_sotas_s5():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s5')
    return trackers, dataset, stream_setting
def streaming_sotas_s6():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s6')
    return trackers, dataset, stream_setting
def streaming_sotas_s7():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s7')
    return trackers, dataset, stream_setting
def streaming_sotas_s8():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s8')
    return trackers, dataset, stream_setting
def streaming_sotas_s9():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                trackerlist('ostrack', 'trial6_ostrack256_aug1')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s9')
    return trackers, dataset, stream_setting
def streaming_sotas_s18():
    trackers =  trackerlist('stark_s', 'baseline') + \
                trackerlist('mixformer_convmae_online', 'baseline') + \
                trackerlist('ostrack', 'baseline') + \
                trackerlist('ostrack', 'aug')
                # trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
                # trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
                # trackerlist('ostrack', 'trial6_ostrack256_aug1') + \
                
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s18')
    return trackers, dataset, stream_setting
def streaming_sotas_s14():
    # trackers =  trackerlist('stark_s', 'baseline') + \
                #trackerlist('mixformer_convmae_online', 'baseline') + \
                #trackerlist('ostrack', 'vitb_256_mae_ce_32x4_ep300') + \
    # trackers =  trackerlist('ostrack', 'trial4_vitb_256_mae_ce_32x4_aligned') + \
    #             trackerlist('ostrack', 'trial6_ostrack256_aug1') + \
    trackers = trackerlist('ostrack', 'trial8_ostrack256') + \
               trackerlist('ostrack', 'trial9_ostrack256') + \
               trackerlist('ostrack', 'baseline') + \
               trackerlist('ostrack', 'aug')
    dataset = get_dataset('esot500s','esot2s')
    # dataset = get_dataset('esot2s')
    # stream_setting = load_stream_setting('s18')
    stream_setting = load_stream_setting('s14')
    return trackers, dataset, stream_setting
def streaming_sotas_s15():
    trackers = trackerlist('ostrack', 'baseline') + \
               trackerlist('ostrack', 'aug') + \
               trackerlist('ostrack', 'aug_2x')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s15')
    return trackers, dataset, stream_setting
def streaming_sotas_s16():
    trackers = trackerlist('ostrack', 'baseline') + \
               trackerlist('ostrack', 'aug')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s16')
    return trackers, dataset, stream_setting
def streaming_sotas_s17():
    trackers = trackerlist('ostrack', 'baseline') + \
               trackerlist('ostrack', 'aug') + \
               trackerlist('ostrack', 'aug_2x')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s17')
    return trackers, dataset, stream_setting
def streaming_sotas_s19():
    trackers = trackerlist('ostrack', 'baseline') + \
               trackerlist('ostrack', 'aug')
    dataset = get_dataset('esot500s','esot2s')
    stream_setting = load_stream_setting('s19')
    return trackers, dataset, stream_setting

def streaming_sotas_ostrack_std():
    trackers = trackerlist('ostrack', 'esot500_baseline')
    dataset = get_dataset('esot500s')
    stream_setting = load_stream_setting('s14')
    return trackers, dataset, stream_setting
    
# python streaming_eval_v3.py  exp_streaming streaming_sotas_s14
