#!/bin/csh
#run corsac_1s merge_sac_1s clipped_1s resp_all_1s
#source ~/.bashrc
#set datadir = /home/GroupNing/tianyuan/data/done_northern
set pdir = /home/GroupNing/tianyuan/surface_wave_tomography
set programdir = $pdir/procedure/pre_processing/pre_proc
set dataname = NChina_pre
cd $pdir
csh $programdir/corsac_1s -data $dataname
echo "finish corsac_1s"
csh $programdir/merge_sac_1s -data $dataname
echo "finish merge_sac_1s"
csh $programdir/clipped_1s -data $dataname
echo "finish clipped_1s"
csh $programdir/resp_all_1s -data $dataname
echo "finish resp_all_1s"


