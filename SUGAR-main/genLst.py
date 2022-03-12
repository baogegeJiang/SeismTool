from glob import glob
import obspy
#NM.WLT._remove_resp_VEL.BHZ
tmpDir = '/media/jiangyr/1TSSD/eventSac/'
file = 'CEAAfter.lst'
#conda activate SUGAR
#python 01_apply_ann.py --filelist_name CEA.lst --output_dir CEA/ --data_dir / --model_name trained_model/ann_202110131500.hdf5
with open(file,'w+') as f:
    for eventDir in glob(tmpDir+'*/'):
        baseTime = float( eventDir.split('/')[-2].split('_')[0])
        if baseTime >= obspy.UTCDateTime(2010,1,1).timestamp:
            continue
        print(eventDir)
        for sacfile in glob(eventDir+'/*remove_resp_DISP.BHZ'):
            f.write('%s\n'%sacfile)
            #print(sacfile)
