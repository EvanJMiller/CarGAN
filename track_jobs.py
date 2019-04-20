import os
import re

def show_job_files(job):

    files = os.listdir('./')
    pattern = r'output_files(?P<job>[\w]+)\.sub\.(?P<type>(e|o))(?P<id>[\w]+)'
    output_files = []
    error_files = []

    print('files...')
    for file in files:
        
	result = re.match(pattern, file)
        
	if result is not None:
           #print(result.groupdict()) 
       
	   if result.groupdict()['type'] == 'e':
           	error_files.append(file)
           elif result.groupdict()['type'] == 'o':
           	output_files.append(file)

    print('Error files')
    print('--------------------------')
    for file in error_files:
        print(file)
        os.rename(file, 'error_files/' + file)

    print('\nOutput files')
    print('--------------------------')
    for file in output_files:
        print(file)
        os.rename(file, 'output_files/' + file)

if __name__ == "__main__":
	show_job_files('my_job')

