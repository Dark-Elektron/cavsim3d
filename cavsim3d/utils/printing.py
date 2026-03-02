
def error(*arg):
    print(f'\x1b[31mERROR:: {arg[0]}\x1b[0m')


def warning(*arg):
    print(f'\x1b[33mWARNING:: {arg[0]}\x1b[0m')


def running(*arg):
    print(f'\x1b[36m{arg[0]}\x1b[0m')


def info(*arg):
    print(f'\x1b[34mINFO:: {arg[0]}\x1b[0m')


def done(*arg):
    print(f'\x1b[32mDONE:: {arg[0]}\x1b[0m')