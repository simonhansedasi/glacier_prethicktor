#!/bin/bash

# # python workflow_step0_merge_training_data.py 
# python workflow_step1_run_bootstrap_and_ensemble.py 
# # <<EOF
# # '1'
# # EOF

# python workflow_step2_calculate_training_stats.py
# # <<EOF
# # sm9
# # EOF

# python workflow_step3_predict_rgi_thicknesses.py
# # <<EOF
# # sm9
# # EOF

# python workflow_step4_calculate_rgi_thickness_statistics.py
# # <<EOF
# # sm9
# # EOF

python full_workflow.py<<EOF
1
EOF