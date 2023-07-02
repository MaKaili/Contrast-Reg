import argparse

parser = argparse.ArgumentParser(description='manual to this script')


def test(x, a, b):
    print(x)
    print(a)
    print(b)


if __name__ == "__main__":
    kwargs = {"a": 1, "b": 2}
    test(1, **kwargs)
