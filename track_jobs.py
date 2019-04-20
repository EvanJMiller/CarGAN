import os
import re

def show_job_files(job):
    output_files = []
    error_files = []

    files = os.listdir('/')

    pattern = r'(?P<job>[\w\_]+).sub.(?P<id>(?P<type>(e|o))[\w]+)'

    for file in files:
        result = re.match(pattern, job)
        if result is not None:
            if result.groupdict()['type'] == 'e':
                error_files.append(file)
            if result.groupdict()['type'] == 'o':
                output_files

    print('Error files')
    print('--------------------------')
    for file in error_files:
        print(file)
        #os.rename('/error_files/' + file)

    print('\nOutput files')
    print('--------------------------')
    for file in output_files:
        print(file)
        #os.rename('/output_files' + file)


