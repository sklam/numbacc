import tempfile
from nbcc.frontend import frontend


def _compile(src: str):
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".spy", prefix="test", delete=False
    ) as tmpfile:
        tmpfile.write(src)
        tmpfile.flush()
        path = tmpfile.name
        return frontend(path)


def test_basic_if():
    source = """
def main() -> None:
    a: i32
    b: i32
    c: i32

    a = 1
    b = 2
    if a < b:
        c = b
    else:
        c = a

    print(c)
"""
    _compile(source)
