import sys


def print(*args, **kwargs):
    """
    sys.stdoutでラップしました
    基本的に普通のprintと一緒ですがflushだけは絶対に指定しましょう
    """
    sys.stdout.write(
        kwargs.pop("sep", " ").join([str(x) for x in args]) + kwargs.pop("end", "\n")
    )

    if kwargs.pop("flush", False):
        sys.stdout.flush()
