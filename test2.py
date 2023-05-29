import subprocess

proc = subprocess.Popen(["python", "test.py"], stdout=subprocess.PIPE)
out = proc.communicate()[0]
print(out.upper())