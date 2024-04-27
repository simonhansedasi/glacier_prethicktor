import os
home_path = None

def set_paths():
    global home_path
    if home_path is None:
        raise ValueError('Home path is not set. Please set the home path before using set_paths')
        
    data_path = f'{home_path}/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    RGI_path = f'{data_path}/RGI'
    if not os.path.exists(RGI_path):
        os.makedirs(RGI_path)
    glathida_path = f'{data_path}/glathida'
    if not os.path.exists(glathida_path):
        os.makedirs(glathida_path) 
    ref_path = f'{data_path}/ref'
    if not os.path.exists(ref_path):
        os.makedirs(ref_path) 
        
        
    model_path = f'{home_path}/models'
    if not os.path.exists(model_path):
        os.makedirs(model_path) 
    coregistration_testing_path = f'{model_path}/coregistration_testing'
    if not os.path.exists(coregistration_testing_path):
        os.makedirs(coregistration_testing_path) 
    arch_test_path = f'{model_path}/arch_test'
    if not os.path.exists(arch_test_path):
        os.makedirs(arch_test_path) 
    LOO_path = f'{model_path}/LOO'
    if not os.path.exists(LOO_path):
        os.makedirs(LOO_path)    
    
    return home_path, data_path, RGI_path, glathida_path, ref_path, coregistration_testing_path, arch_test_path, LOO_path

