import os

def main():
    os.system("python myTF.py | tee -a terminalOutput.txt")
    # execfile("python myTF.py |& tee terminalOutput.txt")

main()
