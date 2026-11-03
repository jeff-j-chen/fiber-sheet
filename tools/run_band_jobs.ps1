$py = 'C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe'
Set-Location 'C:\Users\ChenJeff\.VC3D\afv_model'
& $py -u afv_infer_band.py 'C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths\auto_grown_20261012065803387_straightened2_abf' --band 40 --around 3845 3521.9 12936.1 --radius 900 *> 'C:\Users\ChenJeff\.VC3D\afv_predictions\band_r900.log'
& $py -u afv_infer_band.py 'C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths\auto_grown_20261012065803387_straightened2_abf' --band 40 *> 'C:\Users\ChenJeff\.VC3D\afv_predictions\band_full.log'
