def cyan(*args):
    print(f"\033[96m{' '.join(map(str, args))}\033[0m")

def orange(*args):
    print(f"\033[93m{' '.join(map(str, args))}\033[0m")

def green(*args):
    print(f"\033[92m{' '.join(map(str, args))}\033[0m")

def red(*args):
    print(f"\033[91m{' '.join(map(str, args))}\033[0m")
