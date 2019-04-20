
import os
import re

def show_job_files(job):
    filtered = []
    files = os.listdir()

    pattern = r'(?P<job>[\w\_]+).sub.(?P<id>(?P<type>(e|o))[\w]+)'

    result = re.match(pattern, job)

    print(result.groupdict())

    for file in files:
        if job in file:
            filtered.append(file)

    for file in filtered:
        print(file)

if __name__ == "__main__":

    ex1 = 'my_job.sub.e36849'
    ex2 = 'my_job.sub.o36848'

    show_job_files(ex1)
    show_job_files(ex2)
