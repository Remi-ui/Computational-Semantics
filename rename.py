import os

def file_rename(old_type, new_type, filenames):
        for filename in filenames:
                tem = os.path.splitext(filename)
                print(tem)
                if tem[1] == old_type:
                        new_filename = tem[0] + new_type
                        os.rename(filename, new_filename)
		
filenames = os.listdir(os.getcwd())
file_rename(".conll", ".txt", filenames)
