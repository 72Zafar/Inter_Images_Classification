import zipfile
zip_path = r'D:\My_Folder\New_Mlpos_Coures_projects\Inter_Images_Classification\Data\seg_pred.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    for name in z.namelist()[:100]:
        print(name)
